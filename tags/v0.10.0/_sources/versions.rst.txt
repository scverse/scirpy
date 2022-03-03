Other versions
==============

There are other versions of the documentation available: 


.. raw:: html

   <ul id="versions_container">
	<li><a href="https://icbi-lab.github.io/scirpy">latest release</a></li>
	<li><a href="https://icbi-lab.github.io/scirpy/develop">latest development version</a></li>
   </ul>
   <script type="text/javascript">
   	fetch("https://icbi-lab.github.io/scirpy/versions/versions.json")
	   .then(response => response.json())
	   .then(versions => versions.forEach((x, i) => {
	   	document.getElementById("versions_container").innerHTML += '<li><a href="https://icbi-lab.github.io/scirpy/tags/' + x + '/">' + x + '</a></li>\n'
           })).catch((error) => {
	       document.getElementById("versions_container").innerHTML = "Could not download version information..."
	   })
   </script>
