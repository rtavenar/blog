import os
import frontmatter
import datetime as dt

def cast_date(s_date):
    date = dt.datetime.strptime(s_date, "%Y/%m/%d")
    return date.strftime("%b. %d, %Y")

l_md = [
    fname for fname in os.listdir(".") 
    if fname.endswith(".md") and "aggreg" in frontmatter.load(fname).keys()
]

for fname in l_md:
    post = frontmatter.load(fname)
    full_content = "---\n"
    for k in post.keys():
        full_content += f"{k}: {post[k]}\n"
    full_content += "---\n\n"
    for basename in post["aggreg"].split(","):
        fname_inc = (basename + ".md").strip()
        metadata, content = frontmatter.parse(open(fname_inc, "r").read())
        full_content += content
    open(f"aggreg/{fname}", "w").write(full_content)

# Yet to do: add section titles equal to current file + decrement all other title levels by 1, then include make_aggreg in makefile