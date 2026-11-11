"""Compatibility entrypoint for src.detection.import_reviewed."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("src.detection.import_reviewed", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.detection.import_reviewed")
