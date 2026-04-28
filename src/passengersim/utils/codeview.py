from IPython.display import HTML, display
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename


def show_file(filename):
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
