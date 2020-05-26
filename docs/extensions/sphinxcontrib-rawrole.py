"""Adds a :raw:html role to sphinx. 

From https://gist.github.com/tk0miya/88901de09906f5f8a9fb8b729a4ff3ab
"""

from docutils import nodes


def generate_rawrole(format):
    def role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
        return [nodes.raw("", text, format=format)], []

    return role


def on_builder_inited(app):
    for format in app.config.rawrole_formats:
        name = "raw:%s" % format
        app.add_role(name, generate_rawrole(format))


def setup(app):
    app.add_config_value("rawrole_formats", ["html", "latex"], "env")
    app.connect("builder-inited", on_builder_inited)
