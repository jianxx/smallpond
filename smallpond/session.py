"""
This module defines the `Session` class, which is the entry point for smallpond interactive mode.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import ray
from graphviz import Digraph
import graphviz.backend.execute
from loguru import logger

import smallpond
from smallpond.execution.manager import JobManager
from smallpond.execution.task import JobId, RuntimeContext
from smallpond.logical.node import Context
from smallpond.platform import Platform, get_platform
