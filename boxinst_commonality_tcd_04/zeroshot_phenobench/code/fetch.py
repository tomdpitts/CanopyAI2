"""PhenoBench v1.1.0 val split -> data/val/{images,plant_instances,semantics}/ (downloads are untrusted data: names
checked against a strict pattern, files parsed as PNG only). Pulled member by member from the 7.6 GB zip via HTTP range
requests (server ~150 KB/s per connection -> run several shards in parallel). Licence: CC BY-NC-SA 4.0 per the zip
README (data/README.MD); the website says CC BY-SA 4.0.

  python fetch.py select              -> results/val/selection.json (all val images, sorted; refuses to overwrite)
  python fetch.py files k n           -> shard k of n
  python fetch.py sha1                -> results/val/files_sha1.json (+ check the 60 sample images against the screen record)

  cd zeroshot_phenobench/code && ../../../.venv/bin/python -I -u fetch.py ...
"""
import hashlib, io, json, os, re, sys, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
URL = "https://www.phenobench.org/data/PhenoBench-v110.zip"
D = os.path.join(ROOT, "data", "val"); SUBS = ("images", "plant_instances", "semantics")
SEL = os.path.join(ROOT, "results", "val", "selection.json")
PAT = r"[0-9A-Za-z_\-]+\.png"


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (enough for zipfile)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            b = r.read()
        self.pos += len(b)
        return b

    def readinto(self, buf):
        b = self.read(len(buf)); buf[:len(b)] = b
        return len(b)


def zf():
    return zipfile.ZipFile(io.BufferedReader(RangeFile(URL), buffer_size=1 << 20))


def select():
    assert not os.path.exists(SEL), "selection frozen"
    z = zf(); names = z.namelist()
    files = sorted(os.path.basename(n) for n in names if re.fullmatch(r"PhenoBench/val/images/" + PAT, n))
    for s in SUBS[1:]:
        assert all(f"PhenoBench/val/{s}/{f}" in names for f in files), s
    os.makedirs(os.path.dirname(SEL), exist_ok=True)
    json.dump({"split": "val", "n": len(files), "rule": "all val images, sorted by name", "source": URL, "images": files}, open(SEL, "w"), indent=1)
    print(f"  {len(files)} val images", flush=True)


def files(k, n):
    z = zf(); sel = json.load(open(SEL))["images"]
    for f in sel[k::n]:
        assert re.fullmatch(PAT, f), f
        for s in SUBS:
            os.makedirs(os.path.join(D, s), exist_ok=True); dst = os.path.join(D, s, f)
            if not os.path.exists(dst):
                b = z.read(f"PhenoBench/val/{s}/{f}"); assert b[:8] == b"\x89PNG\r\n\x1a\n", (s, f)
                open(dst + f".tmp{k}", "wb").write(b); os.replace(dst + f".tmp{k}", dst)
        print(f"  {f}", flush=True)
    print(f"  shard {k}/{n} done", flush=True)


def sha1():
    sel = json.load(open(SEL))["images"]
    rec = {f"{s}/{f}": hashlib.sha1(open(os.path.join(D, s, f), "rb").read()).hexdigest() for f in sel for s in SUBS}
    json.dump({"source": URL, "sha1": rec}, open(os.path.join(ROOT, "results", "val", "files_sha1.json"), "w"), indent=1)
    scr = os.path.join(ROOT, "results", "sample60", "files_sha1_screen.json")
    if os.path.exists(scr):
        old = json.load(open(scr))["sha1"]; bad = [k for k, v in old.items() if rec.get(k) != v]
        print(f"  sample60 vs screen record: {len(old) - len(bad)}/{len(old)} identical", flush=True); assert not bad, bad[:3]
    print(f"  {len(rec)} files hashed", flush=True)


if __name__ == "__main__":
    {"select": select, "files": lambda: files(int(sys.argv[2]), int(sys.argv[3])), "sha1": sha1}[sys.argv[1]]()
