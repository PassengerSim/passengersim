import pathlib
import re

import xmle
from altair import LayerChart
from altair.utils.schemapi import UndefinedType

from passengersim.reporting.report import Elem
from passengersim.utils.bootstrap.logo import passengersim_white_green_logo
from passengersim.utils.colors import DarkPurple, LightPurple
from passengersim.utils.filenaming import filename_with_timestamp


class BootstrapHtml:
    common_css = f"""
    @import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,300;0,400;0,700;1,400&display=swap');
    :root {{
      --bs-font-sans-serif:
            Roboto, system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial,
            "Noto Sans", "Liberation Sans", sans-serif, "Apple Color Emoji",
            "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
    }}
    body {{
        position: relative;
    }}
    .navbar {{
        background-color: {DarkPurple};
        margin: 0 0 10px;
        padding: .5rem 1.5rem;
    }}
    .nav-link, .navbar-brand {{
        color: white;
    }}
    :target::before {{
      content: "";
      display: block;
      height: 65px; /* fixed header height*/
      margin: -65px 0 0; /* negative fixed header height */
    }}
    .sticky-top-offset {{
        top: 65px;
    }}
    .nav-link {{
        border-radius: 10px;
    }}
    .nav-link.active {{
        background-color: {LightPurple};
    }}
    h1 {{
        margin-top: 1rem;
        font-size: 135%;
        font-weight: 700;
    }}
    h2 {{
        margin-top: 1rem;
        font-size: 120%;
        font-weight: 300;
    }}
    """

    def __init__(self, title: str = "PassengerSim Report", scrollspy: bool = False):
        self._numbered_figure = xmle.NumberedCaption("Figure", level=2, anchor=True)
        self._numbered_table = xmle.NumberedCaption("Table", level=2, anchor=True)
        self._scrollspy = bool(scrollspy)

        self.top = Elem("html", lang="en")
        self.head = self.top.elem("head")
        self.head.elem("meta", charset="utf-8")
        self.head.elem(
            "meta", name="viewport", content="width=device-width, initial-scale=1"
        )
        if title is None:
            title = "PassengerSim Report"
        self.title = self.head.elem("title", text=str(title))
        self.head.elem(
            "link",
            href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css",
            rel="stylesheet",
            integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH",
            crossorigin="anonymous",
        )
        self.head.elem("style", text=self.common_css)
        self.body = self.top.elem(
            "body",
            {"data-bs-spy": "scroll", "data-bs-target": "#top-nav"}
            if self._scrollspy
            else {},
        )
        self.main = self.body.elem("div", {"class": "container"})
        self.main_row = self.main.elem("div", {"class": "row"})
        self.content = self.main_row.elem("div")
        self.current_section = self.content

        self.sections = {}
        self.body.elem(
            "script",
            src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js",
        )
        self.body.elem(
            "script",
            src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js",
            integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz",
            crossorigin="anonymous",
        )

    def _add_javascript(self, vega: bool = True, floating_tablehead: bool = True):
        if vega:
            self.head.elem(
                "script",
                attrib={
                    "src": "https://cdn.jsdelivr.net/npm/vega@5",
                },
            )
            self.head.elem(
                "script",
                attrib={
                    "src": "https://cdn.jsdelivr.net/npm/vega-lite@5",
                },
            )
            self.head.elem(
                "script",
                attrib={
                    "src": "https://cdn.jsdelivr.net/npm/vega-embed@6",
                },
            )

        if floating_tablehead:
            self.floatThead = self.head.elem(
                tag="script",
                attrib={
                    "src": "https://cdnjs.cloudflare.com/ajax/libs/floatthead/1.4.0/jquery.floatThead.min.js",
                },
            )
            self.floatTheadA = self.head.elem(tag="script")
            self.floatTheadA.text = """
            $( document ).ready(function() {
                var $table = $('table.floatinghead');
                $table.floatThead({ position: 'absolute' });
                var $tabledf = $('table.dataframe');
                $tabledf.floatThead({ position: 'absolute' });
            });
            $(window).on("hashchange", function () {
                window.scrollTo(window.scrollX, window.scrollY - 50);
            });
            """

    def renumber_numbered_items(self):
        # Find all xmle_caption classes
        caption_classes = set()
        for i in self.top.findall(".//span[@xmle_caption]"):
            caption_classes.add(i.attrib["xmle_caption"])
        for caption_class in caption_classes:
            for n, i in enumerate(
                self.top.findall(f".//span[@xmle_caption='{caption_class}']")
            ):
                i.text = re.sub(
                    rf"{caption_class}(\s?[0-9]*):",
                    f"{caption_class} {n + 1}:",
                    i.text,
                    count=0,
                    flags=0,
                )

    def write(self, filename: str, make_dirs: bool = True) -> pathlib.Path:
        self._add_javascript()
        self.renumber_numbered_items()
        self.navbar()
        filename = filename_with_timestamp(
            filename, suffix=".html", make_dirs=make_dirs
        )
        with open(filename, "w") as f:
            f.write("<!doctype html>\n")
            f.write(self.top.tostring())
        return filename

    def new_section(self, title: str, level: int = 1):
        if title in self.sections:
            raise ValueError(f"Section {title} already exists")
        if level == 1:
            section = self.content.elem("div")
            ident = title.lower().replace(" ", "-")
            section.elem("h1", text=title, id=ident)
            self.sections[title] = section
        elif level == 2:
            section = self.current_section.elem("div")
            ident = title.lower().replace(" ", "-")
            section.elem("h2", text=title, id=ident)
        else:
            raise ValueError(f"Invalid level {level}")
        self.current_section = section
        return section

    def set_section(self, title: str):
        if title not in self.sections:
            return self.new_section(title)
        self.current_section = self.sections[title]
        return self.current_section

    def navbar(self):
        nav = Elem(
            "nav",
            {
                "id": "top-nav",
                "class": "navbar sticky-top navbar-expand-lg navbar-light",
            },
        )
        nav_brand = nav.elem("a", {"class": "navbar-brand", "href": "#"})
        nav_brand.append(
            passengersim_white_green_logo(
                {"width": "180px", "height": "25px", "style": "margin-top: -7px;"}
            )
        )
        nav.elem(
            "button",
            {
                "class": "navbar-toggler",
                "type": "button",
                "data-toggle": "collapse",
                "data-target": "#navbarSupportedContent",
                "aria-controls": "navbarSupportedContent",
                "aria-expanded": "false",
                "aria-label": "Toggle navigation",
            },
        ).elem("span", {"class": "navbar-toggler-icon"})
        div = nav.elem(
            "div", {"class": "collapse navbar-collapse", "id": "navbarSupportedContent"}
        )
        ul = div.elem("ul", {"class": "navbar-nav mr-auto"})

        for section in self.sections:
            ident = section.lower().replace(" ", "-")
            ul.elem("li", {"class": "nav-item"}).elem(
                "a", {"class": "nav-link", "href": f"#{ident}"}, text=section
            )

        self.nav = nav
        self.body.insert(0, nav)

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
        self.current_section.append(self._numbered_figure(title))
        self.current_section.append(Elem.from_any(fig))
        if stolen_title:
            fig.title = title
        return fig

    def add_table(self, title, tbl):
        self.current_section.append(self._numbered_table(title))
        self.current_section.append(tbl)
        return tbl
