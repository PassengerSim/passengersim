import pathlib
import re

import pandas as pd
import xmle
from altair import LayerChart
from altair.utils.schemapi import UndefinedType
from xmle.uid import uid

from passengersim.reporting.report import Elem
from passengersim.utils.bootstrap.logo import passengersim_white_green_logo
from passengersim.utils.colors import DarkGreen, DarkPurple, LightGreen, LightPurple
from passengersim.utils.filenaming import filename_with_timestamp


class NumberedCaption:
    def __init__(self, kind, level=2, anchor=None):
        self._kind = kind
        self._level = level
        self._anchor = anchor

    def __call__(self, caption, anchor=None, level=None, attrib=None, **extra):
        unique_id = uid()
        n = level if level is not None else self._level
        result = Elem(f"h{n}", {"id": unique_id})
        if anchor is None:
            anchor = self._anchor
        if anchor:
            result.put(
                "a",
                {
                    "name": unique_id,
                    "reftxt": anchor if isinstance(anchor, str) else caption,
                    "class": "toc",
                    "toclevel": f"{n}",
                },
            )
        lower_kind = self._kind.lower().replace(" ", "_")
        result.put(
            "span",
            {
                "class": f"xmle_{lower_kind}_caption xmle_caption",
                "xmle_caption": self._kind,
            },
            tail=caption,
            text=f"{self._kind}: ",
        )
        return result


class BootstrapHtml:
    common_css = f"""
    @import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,300;0,400;0,700;1,400&display=swap');
    @import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");
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
    .nav-link:hover {{
        color: {LightGreen};
    }}
    .nav-link-side {{
        color: black;
        border-radius: 3px;
    }}
    .nav-link-side:hover {{
        color: {DarkGreen};
    }}
    .nav-link-side.active {{
        color: {DarkPurple};
        background-color: #f8ebff;
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
        color: {LightGreen};
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
    .scrolly-sidebar {{
        position: -webkit-sticky;
        position: sticky;
        height: calc(100vh - 65px);
        overflow-y: auto;
        font-size: 85%;
    }}
    """

    def __init__(self, title: str = "PassengerSim Report", scrollspy: bool = True):
        self._numbered_figure = NumberedCaption("Figure", level=2, anchor=True)
        self._numbered_table = NumberedCaption("Table", level=2, anchor=True)
        self._scrollspy = bool(scrollspy)
        self._toc_place = "sidebar"

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
            {
                "data-bs-spy": "scroll",
                "data-bs-target": "#top-nav" if self._toc_place == "top" else "#toc",
            },
            # if self._scrollspy
            # else {},
        )
        self.main = self.body.elem("div", {"class": "container"})
        self.main_row = self.main.elem("div", {"class": "row"})
        if self._toc_place == "sidebar":
            self.sidebar = self.main_row.elem(
                "div",
                {
                    "class": "col-lg-3 scrolly-sidebar sticky-top "
                    "sticky-top-offset d-none d-lg-inline-flex"
                },
            )
            self.content = self.main_row.elem("div", {"class": "col-12 col-lg-9"})
        else:
            self.content = self.main_row.elem("div")
        self.current_section = self.content
        self.frontmatter = self.current_section.elem("div")

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

    def _add_javascript(self, vega: bool = True, floating_tablehead: bool = False):
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

    def write(
        self, filename: str, *, make_dirs: bool = True, timestamp=None
    ) -> pathlib.Path:
        self._add_javascript()
        self.renumber_numbered_items()
        self.navbar()
        filename = filename_with_timestamp(
            filename, suffix=".html", make_dirs=make_dirs, timestamp=timestamp
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
            section.elem("h1", text=title, id=ident).anchor(
                ident, reftxt=title, cls={}, toclevel=str(level)
            )
            self.sections[title] = section
        elif level == 2:
            section = self.current_section.elem("div")
            ident = title.lower().replace(" ", "-")
            section.elem("h2", text=title, id=ident).anchor(
                ident, reftxt=title, cls={}, toclevel=str(level)
            )
        else:
            raise ValueError(f"Invalid level {level}")
        self.current_section = section
        return section

    add_section = new_section

    def set_section(self, title: str):
        if title not in self.sections:
            return self.new_section(title)
        self.current_section = self.sections[title]
        return self.current_section

    def append(self, *args):
        self.current_section.append(*args)

    def navbar(self):
        nav_top = Elem(
            "nav",
            {
                "id": "top-nav",
                "class": "navbar sticky-top navbar-expand-lg navbar-light",
            },
        )
        nav = nav_top
        nav_brand = nav.elem(
            "a", {"class": "navbar-brand", "href": "https://www.passengersim.com"}
        )
        nav_brand.append(
            passengersim_white_green_logo(
                {"width": "180px", "height": "25px", "style": "margin-top: -7px;"}
            )
        )
        if self._toc_place == "top":
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
                "div",
                {"class": "collapse navbar-collapse", "id": "navbarSupportedContent"},
            )
            ul = div.elem("ul", {"class": "navbar-nav mr-auto"})
            for section in self.sections:
                ident = section.lower().replace(" ", "-")
                ul.elem("li", {"class": "nav-item"}).elem(
                    "a", {"class": "nav-link", "href": f"#{ident}"}, text=section
                )
        else:
            ul = nav.elem("ul", {"class": "navbar-nav mr-auto"})
            ul.elem("li", {"class": "nav-item"}).elem(
                "a", {"class": "nav-link pt-1", "href": "#"}, text=self.title.text
            )

        self.nav = nav_top
        self.body.insert(0, nav_top)

        if self._toc_place == "sidebar":
            toc = Elem("nav", {"id": "toc"})
            toc.elem(
                "div",
                text="Table of Contents",
                attrib={"class": "fw-bold mt-3 ms-1 ps-1"},
            )
            for i in self._rebuild_toc():
                toc.append(i)
            self.sidebar.append(toc)

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

    def add_table(self, title, tbl, collapsible: bool | None = None):
        self.current_section.append(self._numbered_table(title))
        unique_id = uid()
        if collapsible is None:
            collapsible = isinstance(tbl, pd.DataFrame) and len(tbl) > 9
        if collapsible:
            tbl_eye = Elem(
                "button",
                {
                    "class": "btn btn-default btn-xs",
                    "type": "button",
                    "data-bs-toggle": "collapse",
                    "data-bs-target": f"#{unique_id}",
                    "aria-expanded": "false",
                    "aria-controls": f"{unique_id}",
                },
            )
            tbl_eye.elem(
                "i",
                {"class": "bi bi-eye mr-2", "aria-hidden": "true"},
                tail="View Table",
            )
            tbl_collapser = Elem("div", {"class": "collapse", "id": unique_id})
            tbl_collapser.append(tbl)
            self.current_section.append(tbl_eye)
            self.current_section.append(tbl_collapser)
        else:
            self.current_section.append(tbl)
        return tbl

    def add_extra(self, obj, *args):
        """Add extra content to the report.

        Each extra content item can one of a number of things:
        - A string starting with "# " will create a new section.
        - A string starting with "## " will create a new subsection.
        - A tuple or list of length 2, where the first element is a string
          and the second element is a function that returns a figure or table.
        - A function that returns a figure or table.

        Each "function that returns a figure or table" should take a single
        argument, which is the `obj` that is passed to this method.  To give
        a function that takes other specific arguments, use functools.partial
        to create a new function that takes only the `obj` argument.

        Parameters
        ----------
        obj : object
            The object to pass to the functions that generate figures and tables.
        """
        for item in args:
            if isinstance(item, str) and item.startswith("# "):
                self.new_section(item[2:])
                continue
            if isinstance(item, str) and item.startswith("## "):
                self.new_section(item[3:], level=2)
                continue
            if (
                isinstance(item, tuple | list)
                and len(item) == 2
                and isinstance(item[0], str)
            ):
                title, item = item
                fig = item(obj)
                if isinstance(fig, pd.DataFrame):
                    self.add_table(title, fig)
                else:
                    self.add_figure(title, fig)
            else:
                fig = item(obj)
                self.add_figure(fig)

    def add_frontmatter(self, *content):
        for c in content:
            self.frontmatter.append(Elem.from_any(c))

    def _rebuild_toc(self):
        current_toc = Elem("div")

        xtoc_tree = [
            current_toc.put("nav", {"class": "nav nav-pills flex-column pb-3"})
        ]

        min_anchor_lvl = 5
        for anchor in self.content.findall(".//a[@toclevel]"):
            anchor_lvl = int(anchor.get("toclevel"))
            if anchor_lvl < min_anchor_lvl:
                min_anchor_lvl = anchor_lvl

        for anchor in self.content.findall(".//a[@toclevel]"):
            anchor_ref = anchor.get("name")
            anchor_text = anchor.get("reftxt")
            anchor_lvl = int(anchor.get("toclevel")) - min_anchor_lvl + 1
            while anchor_lvl > len(xtoc_tree):
                xtoc_tree.append(
                    xtoc_tree[-1].put(
                        "nav", {"class": "nav nav-pills flex-column ms-1 ps-1"}
                    )
                )
            while anchor_lvl < len(xtoc_tree):
                xtoc_tree = xtoc_tree[:-1]
            xtoc_tree[-1].append(
                Elem(
                    "a",
                    text=anchor_text,
                    attrib={
                        "class": "nav-link-side ms-1 ps-1",
                        "href": f"#{anchor_ref}",
                    },
                )
            )

        return list(current_toc)
