import time

import ipywidgets as ipw


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (e.g., IDLE)
    except NameError:
        return False  # Probably standard Python interpreter


class OutputView:
    def __init__(self, clear_on_destruct: bool = True):
        self._clear_on_destruct = clear_on_destruct
        self.output_widget = ipw.Output()
        self.start_time = time.time()
        self.is_notebook = is_notebook()
        if self.is_notebook:
            from IPython.display import display

            display(self.output_widget)

    def __enter__(self):
        return self.output_widget.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.output_widget.__exit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time // 60
        elapsed_seconds = elapsed_time % 60
        elapsed_hours = elapsed_minutes // 60
        elapsed_minutes = elapsed_minutes % 60
        if self._clear_on_destruct:
            self.output_widget.clear_output()
        with self.output_widget:
            print(f"total elapsed time: {elapsed_hours:.0f}:{elapsed_minutes:02.0f}:{elapsed_seconds:02.2f}")
