"""pytest configuration: ensure vendored packages are on sys.path."""
import sys
import os

_vendored = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".venv_packages"
)
if os.path.isdir(_vendored) and _vendored not in sys.path:
    sys.path.insert(0, _vendored)
