# The default viewer for content published to GitHub does not execute any
# embedded Javascript code. Altair charts are rendered via javascript, so they
# do not appear in Github notebook views.
# https://stackoverflow.com/questions/71346406/why-are-my-altair-data-visualizations-not-showing-up-in-github#
#
# When publishing a notebook to GitHub, the following code is used to render
# Altair plots in a notebook...
# ref: https://altair-viz.github.io/user_guide/display_frontends.html#displaying-in-jupyterlab

import altair as alt

alt.renderers.enable("mimetype")
