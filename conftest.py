"""Pytest configuration: ensure the workspace root is on sys.path
so `import qhomogenize` works whether or not the package is pip-installed."""

import os
import sys

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)
