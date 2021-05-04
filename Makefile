TARGETS=dtw softdtw
.PHONY: all clean

all: post-build

post-build: $(TARGETS)
	python make_index.py
	pandoc -s -o html/index.html --highlight-style pygments --data-dir . --columns 1000 --ascii --template=templates/default_index.html index.md

$(TARGETS): %: html/%.html

html/%.html: %.md
	pandoc -s -o $@ -C --highlight-style pygments --data-dir . --columns 1000 --ascii --mathjax --toc --toc-depth=2 --csl="templates/din-1505-2-alphanumeric.csl.xml" --template=templates/default.html $^

clean:
	rm -f html/*.html
