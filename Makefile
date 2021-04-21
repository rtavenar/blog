TARGETS=dtw
all: post-build

post-build: $(TARGETS)
	python make_index.py
	pandoc -s -o html/index.html --highlight-style pygments --data-dir . --columns 1000 --ascii --template=templates/default.html index.md

$(TARGETS): %: html/%.html

html/%.html: %.md
	pandoc -s -o $@ -C --highlight-style pygments --data-dir . --columns 1000 --ascii --number-sections --template=templates/default.html $^

clean:
	rm -f html/*.html
