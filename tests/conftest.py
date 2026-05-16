"""
Pytest configuration for forgety tests.

Sets rootdir to /tests and ensures /src is on sys.path so that
'app.main' and 'libs.*' are importable from the test environment.
"""
import os
import sys

# Ensure /src is on the path so 'from app.main import ...' works
if '/src' not in sys.path:
    sys.path.insert(0, '/src')
