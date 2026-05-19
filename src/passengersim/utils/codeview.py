import inspect
import re

from IPython.display import HTML, Markdown, display
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename


def show_file(filename):
    """Display a syntax-highlighted view of a source file in a Jupyter notebook.

    Uses Pygments to detect the language from the filename extension and renders
    the file contents as styled HTML via IPython's display machinery.

    Parameters
    ----------
    filename : str or path-like
        Path to the file to display.
    """
    lexer = get_lexer_for_filename(filename)
    formatter = HtmlFormatter(cssclass="pygments")
    with open(filename) as f:
        code = f.read()
    html_code = highlight(code, lexer, formatter)
    css = formatter.get_style_defs(".pygments")
    template = """<style>
    {}
    </style>
    {}"""
    html = template.format(css, html_code)
    display(HTML(html))


def display_docstring_summary(obj):
    """Display a brief summary of an object's docstring in a Jupyter notebook.

    Extracts the portion of the docstring that appears before the ``Parameters``
    section and renders it as Markdown.  If the object has no docstring, or if
    no summary can be parsed, a short fallback message is shown instead.

    Parameters
    ----------
    obj : object
        Any Python object that has a ``__name__`` and, optionally, a ``__doc__``
        attribute (e.g. a function, class, or module).
    """
    doc1 = re.compile(r"^(.*)\n\n\s*Parameters\s*-*\n(.*)$", flags=re.DOTALL)

    try:
        _match = doc1.match(obj.__doc__)
    except (AttributeError, TypeError):
        display(Markdown(f"**{obj.__name__}**: no docstring available"))
        return
    if _match:
        message = f"**{obj.__name__}**: "
        message += inspect.cleandoc(_match.groups()[0]).strip()
        display(Markdown(message))
    else:
        display(Markdown(f"**{obj.__name__}**: no summary available"))
