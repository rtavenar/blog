import os
import frontmatter
import datetime as dt

def cast_date(s_date):
    date = dt.datetime.strptime(s_date, "%Y/%m/%d")
    return date.strftime("%b. %d, %Y")


def valid_file(fname, aggreg=False):
    if not fname.endswith(".md"):
        return False
    post = frontmatter.load(fname)
    return (fname != "index.md" 
            and (("aggreg" in post.keys()) == aggreg)
            and "title" in post.keys() 
            and not post.get("draft", False))

l_base_md = [fname for fname in os.listdir(".") if valid_file(fname, aggreg=False)]
l_aggreg_md = [fname for fname in os.listdir("aggreg/") if valid_file(fname, aggreg=True)]

header = """---
title : A blog by [Romain Tavenard](https://rtavenar.github.io)
language: en
rights: Creative Commons CC BY-NC-SA
---

"""

for l_md, title_section in zip([l_aggreg_md, l_base_md], ["Aggregated posts", "Latest posts"]):
    if len(l_md) == 0:
        continue
    
    header += f"# {title_section}\n\n"

    l_metadata = []

    for fname in l_md:
        post = frontmatter.load(fname)
        if "title" in post.keys() and not post.get("draft", False):
            title = post["title"]
            date_post = post.get("date", "")
            l_metadata.append({"fname": fname, "title": title, "date_post": date_post})

    for d in sorted(l_metadata, key=lambda d: d["date_post"], reverse=True):
        if d["date_post"] != "":
            str_date = f", _posted on {cast_date(d['date_post'])}_"
        else:
            str_date = ""
        header += f"* [{d['title']}]({d['fname'].replace('.md', '.html')}){str_date}.\n"
    header += "\n"

open("index.md", "w").write(header)
