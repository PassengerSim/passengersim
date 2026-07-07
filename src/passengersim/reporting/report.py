from __future__ import annotations

import pathlib
from typing import Any

import pandas as pd
import xmle
import yaml
from altair import LayerChart
from altair.utils.schemapi import UndefinedType
from passengersim_core import __version__ as _passengersim_core_version

from passengersim import __version__ as _passengersim_version
from passengersim.utils.filenaming import filename_with_timestamp

version_tag_0 = f"PassengerSim v{_passengersim_version}"
if _passengersim_version == _passengersim_core_version:
    version_tag_1 = None
    version_tag = version_tag_0
else:
    version_tag_1 = f"Core v{_passengersim_core_version}"
    version_tag = version_tag_0 + f"\n{version_tag_1}"


# allow dumping pathlib.Path objects to YAML
def represent_path(dumper: yaml.SafeDumper, data: pathlib.PurePath) -> yaml.ScalarNode:
    """Represent a :class:`pathlib.PurePath` as a plain YAML string scalar.

    Registered as a multi-representer so that all ``PurePath`` subclasses are
    serialized by PyYAML's :class:`~yaml.SafeDumper`.

    Parameters
    ----------
    dumper : yaml.SafeDumper
        The YAML dumper instance performing the serialization.
    data : pathlib.PurePath
        The path object to serialize.

    Returns
    -------
    yaml.ScalarNode
        A YAML scalar node whose value is the string representation of the
        path.
    """
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))


yaml.add_multi_representer(pathlib.PurePath, represent_path, Dumper=yaml.SafeDumper)


class Report(xmle.Reporter):
    """An HTML reporter pre-configured with PassengerSim branding.

    Extends :class:`xmle.Reporter` with numbered figure and table captions,
    sticky section headers, and a :meth:`save` method that applies custom
    CSS, a table of contents color scheme, and optional filename timestamping.
    """

    def __init__(self, title: str | None = None, short_title: str | None = None) -> None:
        """Initialize the report.

        Parameters
        ----------
        title : str, optional
            Full title displayed at the top of the HTML report.
        short_title : str, optional
            Abbreviated title used in the table of contents and navigation.

        Returns
        -------
        None
        """
        super().__init__(title=title, short_title=short_title)
        self._numbered_figure = xmle.NumberedCaption("Figure", level=2, anchor=True)
        self._numbered_table = xmle.NumberedCaption("Table", level=2, anchor=True)

    def section(self, title: str, short_title: str | None = None) -> xmle.Elem:
        """Append a sticky section header to the report.

        The header is wrapped in a ``<div class="sticky-header">`` element so
        it remains visible while the reader scrolls through the section.

        Parameters
        ----------
        title : str
            Section title text.
        short_title : str, optional
            Abbreviated title appended to the heading with a ``|`` separator,
            for use in table-of-contents entries.

        Returns
        -------
        xmle.Elem
            The ``<div>`` element that was appended to the report.
        """
        s = xmle.Elem("div", cls="sticky-header")
        if short_title:
            s << f"# {title} | {short_title}"
        else:
            s << f"# {title}"
        self << s
        return s

    def add_section(self, title: str, level: int = 1) -> Report:
        """Append a section heading to the report.

        Clears the reporter's previously-seen-HTML state before inserting the
        heading so that the section break is always visible.  Level-1 headings
        use :meth:`section` (sticky header); higher levels use plain Markdown-
        style headings.

        Parameters
        ----------
        title : str
            Section title text.
        level : int, default 1
            Heading level (1 = top-level ``<h1>``, 2 = ``<h2>``, etc.).

        Returns
        -------
        Report
            ``self``, to allow method chaining.
        """
        self.__ilshift__(None)  # clear prior seen_html
        if level <= 1:
            self.section(title)
        else:
            tag = "#" * max(level, 1)
            self.append(f"{tag} {title}")
        return self

    def add_figure(self, title: str | Any, fig: Any = None) -> Any:
        """Append a numbered figure caption and figure to the report.

        When called with a single argument, the title is inferred from the
        figure's ``title`` attribute and the figure's title is temporarily
        cleared while it is rendered (then restored).

        Parameters
        ----------
        title : str or figure
            Either the caption string, or the figure itself when ``fig`` is
            omitted (in which case the title is read from the figure).
        fig : figure object, optional
            The Altair chart or other displayable object to append. If
            omitted, ``title`` is treated as the figure.

        Returns
        -------
        figure object
            The figure that was appended to the report.

        Raises
        ------
        ValueError
            If ``fig`` is omitted and the figure has no ``title`` attribute
            or the title is undefined.
        """
        stolen_title = False
        if fig is None:
            fig = title
            try:
                title = fig.title
            except AttributeError as err:
                raise ValueError("figure has no title attribute") from err
            if isinstance(title, UndefinedType):
                if isinstance(fig, LayerChart):
                    title = fig.layer[0].title
                    for figlayer in fig.layer:
                        figlayer.title = ""
            if isinstance(title, UndefinedType):
                raise ValueError("figure has no title defined")
            fig.title = ""
            stolen_title = True
        self << self._numbered_figure(title)
        self.__ilshift__(Elem.from_any(fig))
        if stolen_title:
            fig.title = title
        return fig

    def add_table(self, title: str, tbl: pd.DataFrame | xmle.Elem) -> pd.DataFrame | xmle.Elem:
        """Append a numbered table caption and table to the report.

        Parameters
        ----------
        title : str
            Caption text for the table.
        tbl : pd.DataFrame or xmle.Elem
            The table object to append.

        Returns
        -------
        pd.DataFrame or xmle.Elem
            The table that was appended to the report.
        """
        self << self._numbered_table(title)
        self.__ilshift__(tbl)
        return tbl

    def save(
        self,
        filename: str | pathlib.Path,
        overwrite: bool = False,
        archive_dir: str | pathlib.Path = "./archive/",
        metadata: dict | None = None,
        timestamp: bool | float = True,
        **kwargs,
    ) -> pathlib.Path:
        """Save the report to an HTML file with PassengerSim branding.

        Applies custom CSS (Roboto font, styled data tables, sticky headers,
        purple accent color) and sets table-of-contents link colors before
        delegating to :meth:`xmle.Reporter.save`.

        Parameters
        ----------
        filename : str or pathlib.Path
            Destination filename.  If ``timestamp`` is truthy the filename
            is augmented with a timestamp before writing.
        overwrite : bool, default False
            If False, raise an error when the destination file already exists.
        archive_dir : str or pathlib.Path, default "./archive/"
            Directory used to archive older versions of the file.
        metadata : dict, optional
            Arbitrary key/value pairs embedded in the HTML file's metadata.
        timestamp : bool or float, default True
            If truthy, append a timestamp to the filename. A ``float`` value
            is interpreted as a UNIX epoch for the timestamp.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`xmle.Reporter.save`.

        Returns
        -------
        pathlib.Path
            The path of the written HTML file.
        """
        _branding = kwargs.pop("branding", version_tag)

        logo_in_sig = Elem("span", {"class": "xmle_name_signature"}, text=version_tag_0)
        if version_tag_1:
            logo_in_sig << Elem("br", tail=version_tag_1)

        import xmle

        xmle.xhtml.logo_in_signature = lambda *x: logo_in_sig

        extra_css = """
        @import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,300;0,400;0,700;1,400&display=swap');

        body {
          font-family: "Roboto", Arial, Helvetica, sans-serif;
          font-weight: 400;
          font-style: normal;
        }

        div.xmle_title {
          font-size: 200%;
          font-family: "Roboto", Arial, Helvetica, sans-serif;
          font-weight: 300;
          font-style: normal;
          color: #6a3d9a;
        }

        .table_of_contents {
          font-size: 85%;
          font-family: "Roboto", Arial, Helvetica, sans-serif;
          font-weight: 400;
          font-style: normal;
        }

        table.dataframe {
          font-family: Arial, Helvetica, sans-serif;
          border-collapse: collapse;
        }

        table.dataframe td, table.dataframe th {
          border: 1px solid #ddd;
          padding: 2px;
        }

        table.dataframe tr:nth-child(even){background-color: #f2f2f2;}

        table.dataframe tr:hover {background-color: #ddd;}

        table.dataframe thead th {
          padding-top: 6px;
          padding-bottom: 6px;
          text-align: left;
          background-color: #cab2d6;
          color: white;
        }

        .sticky-header {
          position: sticky;
          top: 0;
          background: white;
          z-index: 9;
        }
        """
        if timestamp:
            filename = str(filename_with_timestamp(filename))
        toc_font = """font-family: Roboto, Arial, Helvetica, sans-serif; }
        .table_of_contents a:link { color: #6a3d9a; }
		.table_of_contents a:visited { color: #6a5681; }
		.table_of_contents a:hover { color: #6800d6; }
		.table_of_contents a:active { color: #6800d6;
        """
        return super().save(
            filename=filename,
            overwrite=overwrite,
            archive_dir=archive_dir,
            metadata=metadata,
            # branding=branding,
            toc_color="#cab2d6",
            extra_css=extra_css,
            toc_font=toc_font,
            **kwargs,
        )


class Elem(xmle.Elem):
    """An :class:`xmle.Elem` subclass with built-in Altair chart support.

    Overrides :meth:`append` and :meth:`__lshift__` so that Altair
    :class:`~altair.LayerChart` objects are automatically converted to
    embedded Vega-Embed ``<script>`` blocks before being inserted into the
    element tree.  Also provides the :meth:`from_altair` factory class method
    for explicit conversion.
    """

    @classmethod
    def from_altair(cls, fig: Any, classname: str = "altair-figure") -> Elem:
        """Convert an Altair chart to an embeddable HTML element.

        The chart specification is serialized to JSON and wrapped in a
        ``vegaEmbed`` script tag. A random token is used as the DOM ID so
        multiple charts on the same page do not conflict.

        Parameters
        ----------
        fig : altair chart
            Any Altair chart object that exposes a ``to_json`` method.
        classname : str, default "altair-figure"
            CSS class applied to the outer ``<div>`` wrapper.

        Returns
        -------
        Elem
            An :class:`Elem` containing the ``<div>`` target element and
            the ``<script>`` block that embeds the chart.
        """
        import secrets

        unique_token = secrets.token_urlsafe(6)
        spec = fig.to_json(indent=None)
        template = (
            f"""<script type="text/javascript"><![CDATA["""
            f"""vegaEmbed('#vis_{unique_token}', {spec}).catch(console.error);"""
            f"""]]></script>"""
        )
        x = cls("div", {"class": classname})
        x << cls("div", {"id": f"vis_{unique_token}"})
        x << cls.from_string(template)
        return x

    def append(self, arg: Any) -> Any:
        """Append a child element, converting Altair charts automatically.

        If ``arg`` is an Altair :class:`~altair.LayerChart` it is first
        converted via :meth:`from_altair`. The pandas display options are
        expanded so that large DataFrames are rendered in full when converted
        to HTML.

        Parameters
        ----------
        arg : any
            The child element to append. Altair ``LayerChart`` objects are
            converted to embedded HTML; all other types are passed directly
            to the parent implementation.

        Returns
        -------
        any
            The return value of the underlying :meth:`xmle.Elem.append` call.
        """
        if isinstance(arg, LayerChart):
            arg = Elem.from_altair(arg)
        with pd.option_context("display.max_rows", 10_000, "display.max_columns", 1_000):
            return super().append(arg)

    def __lshift__(self, other: Any) -> Elem:
        """Append via the ``<<`` operator, converting Altair charts automatically.

        Parameters
        ----------
        other : any
            The value to append. Altair :class:`~altair.LayerChart` objects
            are converted to embedded HTML before being passed to the parent
            implementation.

        Returns
        -------
        Elem
            ``self``, to support chained ``<<`` expressions.
        """
        if isinstance(other, LayerChart):
            other = Elem.from_altair(other)
        return super().__lshift__(other)
