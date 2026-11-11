"""Compatibility entrypoint for src.detection.verify_stage2."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("src.detection.verify_stage2", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.detection.verify_stage2")
