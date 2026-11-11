
import os

dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "version.txt")
with open(version_file, "r") as f:
    __version__ = f.read().strip()

import importlib
import pkgutil

def import_submodules(package, recursive=True):
    if isinstance(package, str):
        try:
            package = importlib.import_module(package)
        except Exception as e:

            print(f"Failed to import {package}: {e}")
    results = {}
    if not isinstance(package, str):
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                results[name] = importlib.import_module(name)
            except Exception as e:

                print(f"Failed to import {name}: {e}")
            if recursive and is_pkg:
                results.update(import_submodules(name))
    return results

import_submodules(__name__)

from voxcore.auto.auto_model import AutoModel

os.environ["HYDRA_FULL_ERROR"] = "1"

from voxcore.inference import VoxMel
