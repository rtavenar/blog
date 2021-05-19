import os
import frontmatter
import datetime as dt
import argparse

def cast_date(s_date):
    date = dt.datetime.strptime(s_date, "%Y/%m/%d")
    return date.strftime("%b. %d, %Y")


def valid_file(fname, aggreg=False, include_draft=False):
    if not fname.endswith(".md"):
        return False
    post = frontmatter.load(fname)
    return (
        fname != "index.md" 
        and (("aggreg" in post.keys()) == aggreg)
        and "title" in post.keys() 
        and (include_draft or (not post.get("draft", False)))
    )
    
parser = argparse.ArgumentParser(description='Prepare index markdown')
parser.add_argument('--include-draft', action='store_true', help='Include draft posts in the index')
args = parser.parse_args()

l_base_md = [fname for fname in os.listdir(".") if valid_file(fname, aggreg=False, include_draft=args.include_draft)]
l_aggreg_md = []  # [fname for fname in os.listdir("aggreg/") if valid_file(fname, aggreg=True)]

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
