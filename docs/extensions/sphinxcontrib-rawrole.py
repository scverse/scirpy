"""Adds a :raw:html inline role to sphinx.

https://doughellmann.com/posts/defining-custom-roles-in-sphinx/
"""

from docutils import nodes


def html_raw_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
    return [nodes.raw("", text, format="html")], []


def setup(app):
    app.add_role("rawhtml", html_raw_role)
