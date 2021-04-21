TARGETS=index dtw
all: $(TARGETS)

$(TARGETS): %: html/%.html

html/index.html: index.md
	pandoc -s -o $@ --highlight-style pygments --data-dir . --columns 1000 --ascii --template=templates/default.html $^

html/%.html: %.md
	pandoc -s -o $@ -C --highlight-style pygments --data-dir . --columns 1000 --ascii --number-sections --template=templates/default.html $^

clean:
	rm -f html/*.html
