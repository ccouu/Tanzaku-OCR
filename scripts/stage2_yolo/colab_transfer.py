"""Compatibility entrypoint for src.detection.colab_transfer."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("src.detection.colab_transfer", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.detection.colab_transfer")
