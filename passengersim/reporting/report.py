import pathlib

import xmle
import yaml
from altair import LayerChart
from altair.utils.schemapi import UndefinedType


# allow dumping pathlib.Path objects to YAML
def represent_path(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))


yaml.add_multi_representer(pathlib.PurePath, represent_path, Dumper=yaml.SafeDumper)


class Report(xmle.Reporter):
    def __init__(self, title=None, short_title=None):
        super().__init__(title=title, short_title=short_title)
        self._numbered_figure = xmle.NumberedCaption("Figure", level=2, anchor=True)
        self._numbered_table = xmle.NumberedCaption("Table", level=2, anchor=True)

    def add_section(self, title: str, level: int = 1):
        self.__ilshift__(None)  # clear prior seen_html
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
        return super().append(arg)

    def __lshift__(self, other):
        if isinstance(other, LayerChart):
            other = Elem.from_altair(other)
        return super().__lshift__(other)
