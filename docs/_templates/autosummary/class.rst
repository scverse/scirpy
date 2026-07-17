{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. add toctree option to make autodoc generate the pages

.. autoclass:: {{ objname }}

{% block attributes %}
{% if attributes %}
Attributes table
~~~~~~~~~~~~~~~~

.. autosummary::
{% for item in attributes %}
    ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if methods %}
Methods table
~~~~~~~~~~~~~

.. autosummary::
{% for item in methods %}
    {%- if item != '__init__' %}
    ~{{ name }}.{{ item }}
    {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes_documentation %}
{% if attributes %}
Attributes
~~~~~~~~~~~
{# TODO WORKAROUND:
    Due to reasons obscure to me, automatically listing the attributes
    for DataHandler leads to an infinite loop. Also `TYPE` MUST come
    last in the list. I somehow suspect that this is due to the fact that
    the actual docstring of `TYPE` is not displayed in the docs, it is rather replaced
    with "alias of Union[...]" by sphinx.
    I already wasted an evening on it, so let's just go with this explicit list that seems to work #}
{% if objname == "DataHandler" %}
.. autoattribute:: DataHandler.data
.. autoattribute:: DataHandler.adata
.. autoattribute:: DataHandler.mdata
.. autoattribute:: DataHandler.airr
.. autoattribute:: DataHandler.chain_indices
.. autoattribute:: DataHandler.TYPE
{% else %}
{% for item in attributes %}
.. autoattribute:: {{ [objname, item] | join(".") }}
{%- endfor %}
{% endif %}

{% endif %}
{% endblock %}

{% block methods_documentation %}
{% if methods %}
Methods
~~~~~~~

{% for item in methods %}
{%- if item != '__init__' %}

.. automethod:: {{ [objname, item] | join(".") }}
{%- endif -%}
{%- endfor %}

{% endif %}
{% endblock %}
