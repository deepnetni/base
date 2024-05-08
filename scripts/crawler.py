import os
import requests
from urllib import request
from bs4 import BeautifulSoup as bfs

url = f"https://docbox.etsi.org/stq/Open/TS%20103%20224%20Background%20Noise%20Database"


def download(outdir: str = ""):
    os.makedirs(outdir) if not os.path.exists(outdir) else None
    resp = requests.get(url)
    # print(resp.text)
    soup = bfs(resp.text, "html.parser")
    # extract all links item surrounded by <a></a>
    # res = soup.find_all("a")
    res = soup.find_all("a")
    for link in res:
        fname = link.string
        if not fname.endswith("wav"):
            continue
        fname = os.path.join(outdir, fname)

        durl = link["href"]
        with request.urlopen(durl) as check:
            print(check.status, check.reason, durl)
        data = requests.get(durl)
        with open(fname, "wb") as f:
            f.write(data.content)
        print(f"{fname} success")

        # print(link.get("href"))


if __name__ == "__main__":
    download("tmp")
