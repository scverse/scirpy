"""Old code that might be removed in future versions. """
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor


def _convert_to_html_nbconvert(nb_path, out_file):
    """convert executed ipynb file to html document using nbconvert. """
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    html_exporter = HTMLExporter()
    tag_remove_preprocessor = TagRemovePreprocessor(
        remove_cell_tags=["remove_cell"],
        remove_all_outputs_tags=["hide_output"],
        remove_input_tags=["hide_input"],
    )
    html_exporter.template_file = "full"
    html_exporter.register_preprocessor(tag_remove_preprocessor, enabled=True)

    html, resources = html_exporter.from_notebook_node(nb)

    with open(out_file, "w") as f:
        f.write(html)
