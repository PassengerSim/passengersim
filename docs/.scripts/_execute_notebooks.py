"""Execute notebooks."""

import re
import shutil
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

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

    print(f"converting {path} to {path.with_suffix('.nbconvert.ipynb')}")
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600)
        try:
            ep.preprocess(nb, {"metadata": {"path": path.parent}})
        except CellExecutionError:
            print(f'Error executing the notebook "{path}".')
        finally:
            with open(path.with_suffix(".nbconvert.ipynb"), "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
