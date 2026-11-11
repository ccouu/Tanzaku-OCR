"""Compatibility entrypoint for src.detection.prepare_training."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("src.detection.prepare_training", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.detection.prepare_training")
