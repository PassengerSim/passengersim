from __future__ import annotations

import os

from IPython.display import display


class IFrame:
    """
    Generic class to embed an iframe in an IPython notebook
    """

    framestyle = (
        "border: 1px solid #aaaaaa; border-radius: 3px; "
        "padding:3px; width: {width}px; overflow-x: hidden;"
    )
    iframe = """
        <div style="{framestyle}">
        <iframe
            width="{width}"
            height="{height}"
            srcdoc="{src}"
            frameBorder="0"
        ></iframe>
        </div>
        """

    def __init__(self, src, width, height, style=None):
        if style is None:
            self.style = self.framestyle.format(width=width)
        else:
            self.style = style
        self.src = src.replace('"', "&quot;")
        self.width = width
        self.height = height

    def _repr_html_(self):
        """return the embed iframe"""
        return self.iframe.format(
            src=self.src,
            width=self.width,
            height=self.height,
            framestyle=self.style,
        )


def show_html_in_iframe(html_content, width="700", height="300"):
    """
    Displays HTML content in an iframe within a Jupyter Notebook.

    Parameters
    ----------
    html_content : str
        The HTML content to display.
    width : str, optional
        The width of the iframe. Defaults to "800".
    height : str, optional
        The height of the iframe. Defaults to "600".
    """
    iframe = IFrame(html_content, width=width, height=height)
    display(iframe)


def preview_html(filename: os.PathLike, *, width="700", height="300"):
    """Displays HTML content in a Jupyter Notebook cell.

    Parameters
    ----------
    filename : os.PathLike
        The path to the HTML file to display.
    width : str, optional
        The width of the HTML content. Defaults to "100%".
    height : str, optional
        The height of the HTML content. Defaults to "500".

    """
    with open(filename) as f:
        html_content = f.read()
    show_html_in_iframe(html_content, width=width, height=height)
