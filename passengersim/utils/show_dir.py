import os


def display_directory_contents(path, indent=0):
    """Prints the contents of a directory in a visually appealing format."""
    if indent == 0 and os.path.isdir(path):
        print(f"\033[1;36m{os.path.basename(path)}/\033[0m")
        indent += 2  # Increase indentation for subdirectory contents
    for entry in os.scandir(path):
        if entry.is_dir():
            print(
                f"{' ' * indent}\033[1;34m{entry.name}/\033[0m"
            )  # Blue for directories
            display_directory_contents(
                entry.path, indent + 2
            )  # Recursive call with increased indentation
        elif entry.is_file() and not entry.name.startswith("."):
            print(f"{' ' * indent}\033[1;32m{entry.name}\033[0m")  # Green for files
