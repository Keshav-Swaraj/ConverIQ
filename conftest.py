# conftest.py
# Ensures the project root is on the Python path for all tests.
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
