TARGETS=dtw softdtw
.PHONY: all clean

all: $(TARGETS)
	python make_index.py
	pandoc -s -o html/index.html \
		--highlight-style pygments \
		--data-dir . \
		--columns 1000 --ascii \
		--template=templates/default_index.html index.md

local: $(TARGETS)
	python make_index.py --include-draft
	pandoc -s -o html/index.html \
		--highlight-style pygments \
		--data-dir . \
		--columns 1000 --ascii \
		--template=templates/default_index.html index.md

$(TARGETS): %: html/%.html

html/%.html: %.md
	pandoc -s -o $@ \
		--highlight-style pygments \
		--filter pandoc-eqnos -M eqnos-eqref=True -M eqnos-cleveref=True -M eqnos-plus-name=Equation -M eqnos-warning-level=1 \
		--citeproc -M link-citations=true -M reference-section-title=References \
		--data-dir . \
		--columns 1000 --ascii --mathjax \
		--toc --toc-depth=2 \
		--csl="templates/custom.csl" \
		--template=templates/default.html $^

clean:
	rm -f html/*.html
