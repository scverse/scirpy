Other versions
==============

There are other versions of the documentation available:


.. raw:: html

   <ul id="versions_container">
	<li><a href="https://scverse.org/scirpy">latest release</a></li>
	<li><a href="https://scverse.org/scirpy/develop">latest development version</a></li>
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
