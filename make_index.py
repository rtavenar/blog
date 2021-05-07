import os
import frontmatter
import datetime as dt

def cast_date(s_date):
    date = dt.datetime.strptime(s_date, "%Y/%m/%d")
    return date.strftime("%b. %d, %Y")

l_md = [
    fname for fname in os.listdir(".") 
    if fname.endswith(".md") and fname != "index.md"
]

header = """---
title : A blog by Romain Tavenard
language: en
rights: Creative Commons CC BY-NC-SA
---

# Latest posts

"""

l_metadata = []

for fname in l_md:
    post = frontmatter.load(fname)
    if "title" in post.keys() and not post.get("draft", False):
        title = post["title"]
        date_post = post.get("date", "")
        l_metadata.append({"fname": fname, "title": title, "date_post": date_post})

for d in sorted(l_metadata, key=lambda d: d["date_post"], reverse=True):
    header += f"* [{d['title']}]({d['fname'].replace('.md', '.html')}), _posted on {cast_date(d['date_post'])}_.\n"

open("index.md", "w").write(header)
