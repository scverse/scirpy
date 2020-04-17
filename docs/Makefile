# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

livehtml: | clean
	sleep 1 && touch *.rst
	sphinx-autobuild \
	-p 0 \
        --open-browser \
        --delay 5 \
        --ignore "*.swp" \
        --ignore "*.pdf" \
        --ignore "*.log" \
        --ignore "*.out" \
        --ignore "*.toc" \
        --ignore "*.aux" \
        --ignore "*.idx" \
        --ignore "*.ind" \
        --ignore "*.ilg" \
        --ignore "*.tex" \
	-b html $(SOURCEDIR) $(BUILDDIR)/html

clean: 
	rm -v scirpy.*.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
