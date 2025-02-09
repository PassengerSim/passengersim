from xmle import Elem

from passengersim.utils.colors import DarkPurple


class BootstrapHtml:
    common_css = f"""
    @import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,300;0,400;0,700;1,400&display=swap');
    :root {{
      --bs-font-sans-serif:
            Roboto, system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial,
            "Noto Sans", "Liberation Sans", sans-serif, "Apple Color Emoji",
            "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
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

    """

    def __init__(self, title: str = "Bootstrap Report"):
        self.top = Elem("html", lang="en")
        self.head = self.top.elem("head")
        self.head.elem("meta", charset="utf-8")
        self.head.elem(
            "meta", name="viewport", content="width=device-width, initial-scale=1"
        )
        self.title = self.head.elem("title", text=str(title))
        self.head.elem(
            "link",
            href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css",
            rel="stylesheet",
            integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH",
            crossorigin="anonymous",
        )
        self.head.elem("style", text=self.common_css)
        self.body = self.top.elem("body")
        self.main = self.body.elem("div", {"class": "container"})
        self.main_row = self.main.elem("div", {"class": "row"})
        self.sidebar = self.main_row.elem("div", {"class": "col-sm-2"})
        self.content = self.main_row.elem("div", {"class": "col-sm-10"})

        self.sections = {}
        # self.body.elem("h1", text="Hello, world!")
        self.head.elem(
            "script",
            src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js",
        )
        self.head.elem(
            "script",
            src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js",
            integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz",
            crossorigin="anonymous",
        )

    def write(self, filename: str):
        self.navbar()
        self.toc()
        with open(filename, "w") as f:
            f.write("<!doctype html>\n")
            f.write(self.top.tostring())

    def new_section(self, title: str):
        if title in self.sections:
            raise ValueError(f"Section {title} already exists")
        section = self.content.elem("div")
        ident = title.lower().replace(" ", "-")
        section.elem("h2", text=title, id=ident)
        self.sections[title] = section
        return section

    def toc(self):
        """
                <!-- add after bootstrap.min.css -->
        <link
          rel="stylesheet"
          href="https://cdn.rawgit.com/afeld/bootstrap-toc/v1.0.1/dist/bootstrap-toc.min.css"
        />
        <!-- add after bootstrap.min.js or bootstrap.bundle.min.js -->
        <script src="https://cdn.rawgit.com/afeld/bootstrap-toc/v1.0.1/dist/bootstrap-toc.min.js"></script>

        """
        self.head.elem(
            "link",
            rel="stylesheet",
            href="https://cdn.rawgit.com/afeld/bootstrap-toc/v1.0.1/dist/bootstrap-toc.min.css",
        )
        self.head.elem(
            "script",
            src="https://cdn.rawgit.com/afeld/bootstrap-toc/v1.0.1/dist/bootstrap-toc.min.js",
        )
        # self.sidebar.insert(0, Elem("b", text="Table of Contents"))
        self.sidebar.insert(
            0,
            Elem(
                "nav",
                {
                    "id": "toc",
                    "data-toggle": "toc",
                    "class": "sticky-top sticky-top-offset",
                },
            ),
        )
        self.body.attrib["data-bs-spy"] = "scroll"
        self.body.attrib["data-bs-target"] = "#toc"

    def navbar(self):
        nav = Elem("nav", {"class": "navbar sticky-top navbar-expand-lg navbar-light"})
        nav.elem("a", {"class": "navbar-brand", "href": "#"}, text="PassengerSim")
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
        ul.elem("li", {"class": "nav-item active"}).elem(
            "a", {"class": "nav-link", "href": "#"}, text="Home"
        ).elem("span", {"class": "sr-only"}, text="(current)")

        for section in self.sections:
            ident = section.lower().replace(" ", "-")
            ul.elem("li", {"class": "nav-item"}).elem(
                "a", {"class": "nav-link", "href": f"#{ident}"}, text=section
            )

        self.nav = nav
        self.body.insert(0, nav)
