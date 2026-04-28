import pathlib

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
def represent_path(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))


yaml.add_multi_representer(pathlib.PurePath, represent_path, Dumper=yaml.SafeDumper)


class Report(xmle.Reporter):
    def __init__(self, title=None, short_title=None):
        super().__init__(title=title, short_title=short_title)
        self._numbered_figure = xmle.NumberedCaption("Figure", level=2, anchor=True)
        self._numbered_table = xmle.NumberedCaption("Table", level=2, anchor=True)

    def section(self, title, short_title=None):
        s = xmle.Elem("div", cls="sticky-header")
        if short_title:
            s << f"# {title} | {short_title}"
        else:
            s << f"# {title}"
        self << s
        return s

    def add_section(self, title: str, level: int = 1):
        self.__ilshift__(None)  # clear prior seen_html
        if level <= 1:
            self.section(title)
        else:
            tag = "#" * max(level, 1)
            self.append(f"{tag} {title}")
        return self

    def add_figure(self, title, fig=None):
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

    def add_table(self, title, tbl):
        self << self._numbered_table(title)
        self.__ilshift__(tbl)
        return tbl

    def save(
        self,
        filename,
        overwrite=False,
        archive_dir="./archive/",
        metadata=None,
        timestamp: bool | float | bool = True,
        **kwargs,
    ):
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
    @classmethod
    def from_altair(cls, fig, classname="altair-figure"):
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

    def append(self, arg):
        if isinstance(arg, LayerChart):
            arg = Elem.from_altair(arg)
        with pd.option_context("display.max_rows", 10_000, "display.max_columns", 1_000):
            return super().append(arg)

    def __lshift__(self, other):
        if isinstance(other, LayerChart):
            other = Elem.from_altair(other)
        return super().__lshift__(other)
