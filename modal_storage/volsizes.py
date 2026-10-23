import modal, json, sys, collections, concurrent.futures as cf
names = [v for v in sys.argv[1:]]
def size(n):
    v = modal.Volume.from_name(n)
    try:
        ents = v.listdir("/", recursive=True)
    except Exception as e:
        return n, None, str(e), {}
    tot = 0; top = collections.Counter(); nf=0
    for e in ents:
        if e.type == modal.volume.FileEntryType.FILE:
            tot += e.size; nf+=1
            top[e.path.lstrip("/").split("/")[0]] += e.size
    return n, tot, nf, dict(top.most_common(8))
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(size, names))
res.sort(key=lambda r: -(r[1] or 0))
out = []
for n, tot, nf, top in res:
    print(f"{n:35s} {(tot or 0)/1e9:8.2f} GB  files={nf}")
    for k, s in top.items(): print(f"      {k:40s} {s/1e9:7.2f} GB")
    out.append(dict(name=n, bytes=tot, files=nf, top=top))
json.dump(out, open(sys.argv[0].replace('.py','.json'),'w'), indent=1)
