import copy
import os
import ray
from loguru import logger

from smallpond.common import DEFAULT_MAX_RETRY_COUNT
from smallpond.execution.task import Task
from smallpond.execution.workqueue import WorkStatus
from smallpond.io.filesystem import dump, load
from smallpond.logical.dataset import DataSet


def run_on_ray(task: Task) -> ray.ObjectRef:
    """
    Run the task on Ray.
    Return an `ObjectRef`, which can be used with `ray.get` to wait for the output dataset.
    A `_dataset_ref` attribute is added to the task to store the reference.
    """
    if task._dataset_ref is not None:
        # already started
        return task._dataset_ref

    # read the output dataset if the task has already finished
    if os.path.exists(task.ray_dataset_path):
        logger.info(f"task {task.key} already finished, skipping")
        output = load(task.ray_dataset_path)
        task._dataset_ref = ray.put(output)
        return task._dataset_ref

    task = copy.copy(task)
    task.input_deps = {dep_key: None for dep_key in task.input_deps}

    @ray.remote
    def exec_task(task: Task, *inputs: DataSet) -> DataSet:
        import multiprocessing as mp
        import os
        from pathlib import Path

        from loguru import logger

        # ray use a process pool to execute tasks
        # we set the current process name to the task name
        # so that we can see task name in the logs
        mp.current_process().name = task.key

        # probe the retry count
        task.retry_count = 0
        while os.path.exists(task.ray_marker_path):
            task.retry_count += 1
            if task.retry_count > DEFAULT_MAX_RETRY_COUNT:
                raise RuntimeError(
                    f"task {task.key} failed after {task.retry_count} retries"
                )
        if task.retry_count > 0:
            logger.warning(
                f"task {task.key} is being retried for the {task.retry_count}th time"
            )
        # create the marker file
        Path(task.ray_marker_path).touch()

        # put the inputs into the task
        assert len(inputs) == len(task.input_deps)
        task.input_datasets = list(inputs)
        # execute the task
        status = task.exec()
        if status != WorkStatus.SUCCEED:
            raise task.exception or RuntimeError(
                f"task {task.key} failed with status {status}"
            )

        # dump the output dataset atomically
        os.makedirs(os.path.dirname(task.ray_dataset_path), exist_ok=True)
        dump(task.output, task.ray_dataset_path, atomic_write=True)
        return task.output

    # this shows as {"name": ...} in timeline
    exec_task._function_name = repr(task)

    remote_function = exec_task.options(
        # ray task name
        # do not include task id so that they can be grouped by node in ray dashboard
        name=f"{task.node_id}.{task.__class__.__name__}",
        num_cpus=task.cpu_limit,
        num_gpus=task.gpu_limit,
        memory=int(task.memory_limit),
        # note: `exec_on_scheduler` is ignored here,
        #       because dataset is distributed on ray
    )
    try:
        task._dataset_ref = remote_function.remote(
            task, *[run_on_ray(dep) for dep in task.input_deps.values()]
        )
    except RuntimeError as e:
        if (
            "SimpleQueue objects should only be shared between processes through inheritance"
            in str(e)
        ):
            raise RuntimeError(
                f"Can't pickle task '{task.key}'. Please check if your function has captured unpicklable objects. {task.location}\n"
                f"HINT: DO NOT use externally imported loguru logger in your task. Please import it within the task."
            ) from e
        raise e
    return task._dataset_ref
