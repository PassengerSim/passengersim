class StringTracker:
    def __init__(self, case_sensitive: bool = False, start_from: int = 0):
        # Maps unique strings to their assigned numbers
        self.seen_strings = {}
        self.case_sensitive = case_sensitive
        self.start_from = start_from

    def get_number(self, input_string: str) -> int:
        """Assigns or retrieves the unique number for a string."""
        if not self.case_sensitive:
            input_string = input_string.casefold()
        if input_string not in self.seen_strings:
            next_count = len(self.seen_strings) + self.start_from
            self.seen_strings[input_string] = next_count
        return self.seen_strings[input_string]

    def get_number_if_exists(self, input_string: str) -> int | None:
        if not self.case_sensitive:
            input_string = input_string.casefold()
        if input_string in self.seen_strings:
            return self.seen_strings[input_string]
        else:
            return None

    def list_all(self) -> list[str]:
        """Returns the list of all tracked strings, in the order they were first added."""
        return list(self.seen_strings.keys())
