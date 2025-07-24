import argparse
import hashlib
import os.path
import re
import shutil
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Execute Jupyter notebooks.")
parser.add_argument(
    "--run-all",
    action="store_true",
    help="Execute all notebooks regardless of hash status.",
)
args = parser.parse_args()

doc_path = Path(__file__).parents[1]

print(f"executing notebooks found within {doc_path}")

exclusions = ["reading-data"]

for path in sorted(doc_path.rglob("*.ipynb")):
    if ".ipynb_checkpoints" in path.parts:
        # skip any checkpoints
        continue
    if path.name.endswith(".nbconvert.ipynb"):
        # this is an execution output, don't nest
        continue
    exclude = False
    for e in exclusions:
        if re.search(e, str(path)):
            exclude = True
            # simple copy instead of execution
            shutil.copyfile(path, path.with_suffix(".nbconvert.ipynb"))
            continue
    if exclude:
        continue

    print(f"📓 reading {path}")
    with open(path, "rb") as f:
        content = f.read()
        hash_value = hashlib.sha256(content).hexdigest()
        print(f"  SHA-256 hash of the notebook content: {hash_value}")

    skip_execution = False
    if not args.run_all:
        if path.with_suffix(".hash.txt").exists() and path.with_suffix(".nbconvert.ipynb").exists():
            with open(path.with_suffix(".hash.txt"), encoding="utf-8") as f:
                if f.read() == hash_value:
                    print("  🌟 The notebook content has not changed. Skipping execution.")
                    skip_execution = True
                else:
                    print("  💥 The notebook content has changed. Re-executing.")
                    os.remove(path.with_suffix(".hash.txt"))

    if skip_execution:
        continue

    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600)
        clean_execution = True
        try:
            ep.preprocess(nb, {"metadata": {"path": path.parent}})
        except CellExecutionError as err:
            print(f'Error executing the notebook "{path}".')
            print(f"\033[1;32m{err}\033[0m")
            clean_execution = False
        finally:
            print(f"  writing to {path.with_suffix('.nbconvert.ipynb')}")
            with open(path.with_suffix(".nbconvert.ipynb"), "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            if clean_execution:
                with open(path.with_suffix(".hash.txt"), "w", encoding="utf-8") as f:
                    f.write(hash_value)
