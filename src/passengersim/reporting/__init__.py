from functools import wraps

import pandas as pd

try:
    import xmle
except ImportError:
    import warnings

    warnings.warn("reporting dependency 'xmle' not installed", stacklevel=2)
    xmle = None

try:
    from altair.utils.schemapi import UndefinedType
except ImportError:
    import warnings

    warnings.warn("reporting dependencies 'altair' not installed", stacklevel=2)
    UndefinedType = None


if xmle is not None and UndefinedType is not None:
    from .report import Report  # noqa: F401


def write_trace(
    trace: pd.ExcelWriter,
    raw: pd.DataFrame,
    trace_sheetname: str | None = None,
    basesheetname: str = "Sheet",
):
    if trace_sheetname is None:
        sheet_num = 1
        while f"{basesheetname}_{sheet_num}" in trace.sheets:
            sheet_num += 1
        trace_sheetname = f"{basesheetname}_{sheet_num}"
    startrow = 0
    if "title" in raw.attrs:
        startrow += 2
    raw.to_excel(trace, sheet_name=trace_sheetname, index=False, startrow=startrow)
    print(f"TRACE: {trace_sheetname}")
    if "title" in raw.attrs:
        worksheet = trace.sheets[trace_sheetname]
        worksheet.write_string(0, 0, raw.attrs["title"])


def report_figure(func):
    """Decorator to add reporting and tracing capabilities to figure-generating functions.

    This decorator wraps a function that generates a figure (e.g., an Altair chart)
    and which can also return the raw data used to create the figure when called with
    `raw_df=True`. The decorator intercepts calls to the function and adds two optional
    keyword-only parameters `report` and `trace`.

    When `report` is provided, it should be a `Report` object, and then the generated
    figure is added to that Report object.

    When `trace` is provided, it should be a `pd.ExcelWriter` object, and then the
    raw data used to generate the figure is written to a new sheet in the Excel file.
    Optionally, `trace` can be a tuple where the first element is the `pd.ExcelWriter`
    and the second element is a string specifying the sheet name to use.
    """

    @wraps(func)
    def report_figure_wrapper(*args, report=None, trace=None, **kwargs):
        fig = func(*args, **kwargs)
        if report is not None:
            report.add_figure(fig)
        if trace is not None:
            trace_sheetname = None
            if isinstance(trace, tuple):
                if len(trace) > 1:
                    trace_sheetname = trace[1]
                if len(trace) > 0:
                    trace = trace[0]
                else:
                    raise ValueError("trace is a zero-length tuple")
            kwargs["raw_df"] = True
            raw: pd.DataFrame = func(*args, **kwargs)
            if isinstance(trace, pd.ExcelWriter):
                write_trace(trace, raw, trace_sheetname, func.__name__ or "Sheet")
            else:
                raise NotImplementedError("only tracing to excel is implemented")
        return fig

    return report_figure_wrapper
