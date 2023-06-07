Other versions
==============

.. note::
    **Versions of the documentation ≥v0.13 are hosted on readthedocs and can be accessed via the
    sidebar menu at the bottom left.**

Older versions of the documentation are hosted on GitHub pages for historical reasons and can be accessed via
the following list:

.. raw:: html

   <ul id="versions_container">
   </ul>
   <script type="text/javascript">
   	fetch("https://scverse.org/scirpy/versions/versions.json")
	   .then(response => response.json())
	   .then(versions => versions.forEach((x, i) => {
	   	document.getElementById("versions_container").innerHTML += '<li><a href="https://scverse.org/scirpy/tags/' + x + '/">' + x + '</a></li>\n'
           })).catch((error) => {
	       document.getElementById("versions_container").innerHTML = "Could not download version information..."
	   })
   </script>
