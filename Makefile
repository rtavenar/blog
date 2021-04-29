TARGETS=dtw softdtw
.PHONY: all clean

all: post-build

post-build: $(TARGETS)
	python make_index.py
	pandoc -s --self-contained -o html/index.html --highlight-style pygments --data-dir . --columns 1000 --ascii --template=templates/default_index.html index.md

$(TARGETS): %: html/%.html

html/%.html: %.md
	pandoc -s --self-contained -o $@ -C --highlight-style pygments --data-dir . --columns 1000 --ascii --mathjax --toc --template=templates/default.html $^

clean:
	rm -f html/*.html
