"""Generate the code reference pages."""

import re
from pathlib import Path

import mkdocs_gen_files

import passengersim

public_module_display_name = "PassengerSim"

sw_path = Path(passengersim.__path__[0]).parent

for path in sorted(sw_path.joinpath("passengersim/config").rglob("*.py")):
    module_path = path.relative_to(sw_path).with_suffix("")
    doc_path = path.relative_to(sw_path.joinpath("passengersim")).with_suffix(".md")
    full_doc_path = Path("API", doc_path)

    title = None
    classes = []
    skip = False
    with open(sw_path.joinpath(module_path).with_suffix(".py")) as text_file:
        file_lines = text_file.readlines()
        for line in file_lines:
            if line.startswith("# SKIP-DOC"):
                skip = True
                break
            if line.startswith("# TITLE:"):
                title = line[8:].strip()
            if line.startswith("# DOC-NAME:"):
                doc_name = line[11:].strip()
                doc_path = (
                    path.parent.joinpath(doc_name)
                    .relative_to(sw_path.joinpath("passengersim"))
                    .with_suffix(".md")
                )
                full_doc_path = Path("API", doc_path)
            match = re.match(r"class (.*)\(.*", line)
            if match:
                classes.append(match.group(1))

    if skip:
        continue

    parts = list(module_path.parts)

    if parts[-1] == "__init__":  #
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        if title:
            print(f"# {title}\n\n", file=fd)
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)
        print("::: " + identifier)
        print("    options:", file=fd)
        print("      filters:", file=fd)
        print('        - "!^debug.*"', file=fd)
        print('        - "!^_[^_]"', file=fd)
        print('        - "!^logger$"', file=fd)
        print('        - "!^TConfig$"', file=fd)
        print('        - "!^web_opener$"', file=fd)
        print('        - "!^create_timestamp$"', file=fd)
        print('        - "!^T$"', file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

    print(">>>>>>", full_doc_path)
