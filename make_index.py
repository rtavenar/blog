import os
import frontmatter

l_md = [
    fname for fname in os.listdir(".") 
    if fname.endswith(".md") and fname != "index.md"
]

header = """---
title : Blog
language: en
rights: Creative Commons CC BY-NC-SA
---

# Latest posts

"""

l_metadata = []

for fname in l_md:
    post = frontmatter.load(fname)
    if "title" in post.keys():
        title = post["title"]
        date_post = post.get("date", "")
        l_metadata.append({"fname": fname, "title": title, "date_post": date_post})

for d in sorted(l_metadata, key=lambda d: d["date_post"], reverse=True):
    header += f"* {d['date_post']} [{d['title']}]({d['fname']})\n"

print(header)
