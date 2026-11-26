"""Biome composition of the 439-tile OAM-TCD holdout test set -> table_test_biomes.tex.

Source: tile_index.json (split == "test"), same index and same crown/canopy fractions as
table_sparse_biomes.tex. Groups follow tab:gap (Tr / Te / Sv). Paper: tab:test-biomes.

Usage:
    python3 -m boxinst_commonality_tcd_04.modal_sparse_tcd_multiseed.table_test_biomes
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
IDX = os.path.join(HERE, "tile_index.json")
OUT = os.path.join(HERE, "table_test_biomes.tex")

NAMES = {
    1: r"Trop./Subtrop. Moist Broadleaf Forest",
    2: r"Trop./Subtrop. Dry Broadleaf Forest",
    3: r"Trop./Subtrop. Coniferous Forest",
    4: r"Temperate Broadleaf \& Mixed Forest",
    5: r"Temperate Conifer Forest",
    6: r"Boreal Forest/Taiga",
    7: r"Trop./Subtrop. Grassland, Savanna \& Shrubland",
    8: r"Temperate Grassland, Savanna \& Shrubland",
    9: r"Flooded Grassland \& Savanna",
    10: r"Montane Grassland \& Shrubland",
    11: r"Tundra",
    12: r"Mediterranean Forest, Woodland \& Scrub",
    13: r"Desert \& Xeric Shrubland",
    14: r"Mangroves",
}
GROUPS = [("Tropical forest (Tr)", (1, 2, 3, 14)),
          ("Temperate and boreal forest (Te)", (4, 5, 6, 11)),
          ("Savannah/dryland (Sv)", (7, 8, 9, 10, 12, 13)),
          ("No WWF biome match", (-1,))]


def fmt_n(n):
    return f"${n:,}$".replace(",", "{,}")


def main():
    tiles = [r for r in json.load(open(IDX))["tiles"] if r["split"] == "test"]
    assert len(tiles) == 439, len(tiles)
    assert sum(r["n_crowns"] for r in tiles) == 25705

    def row(code, rs, label):
        n = len(rs)
        cr = 100 * sum(r["crown_frac"] for r in rs) / n
        ca = 100 * sum(r["canopy_frac"] for r in rs) / n
        return (f"{code} & {label} & ${n}$ & {fmt_n(sum(r['n_crowns'] for r in rs))} & "
                f"${cr:.1f}\\%$ & ${ca:.1f}\\%$ \\\\")

    L = []
    for gname, codes in GROUPS:
        g = [r for r in tiles if r["biome"] in codes]
        L.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{gname}}}}} \\")
        if codes == (-1,):
            L.append(row("--", g, "Unmatched"))
        else:
            for c in sorted(codes, key=lambda c: -sum(r["biome"] == c for r in tiles)):
                rs = [r for r in tiles if r["biome"] == c]
                if rs:
                    L.append(row(f"${c}$", rs, NAMES[c]))
            if sum(1 for c in codes if any(r["biome"] == c for r in tiles)) > 1:
                L.append(row("", g, r"\emph{Group}"))
        L.append(r"\midrule")
    L[-1] = r"\cmidrule(lr){1-6}"
    n_cr = f"{sum(r['n_crowns'] for r in tiles):,}".replace(",", "{,}")
    L.append(rf" & \textbf{{Total}} & $\mathbf{{{len(tiles)}}}$ & "
             rf"$\mathbf{{{n_cr}}}$ & & \\")

    body = "\n".join(L)
    tex = rf"""\begin{{table}}[H]
\centering
\caption{{Biome composition of the $439$-tile OAM-TCD holdout test set by WWF biome label, grouped as in Table~\ref{{tab:gap}}. Crowns counts individual crown annotations. Crown px and canopy px are the mean per-tile fraction of labelled crown and labelled canopy pixels.}}
\label{{tab:test-biomes}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}clrrrr@{{}}}}
\toprule
Code & Biome (WWF) & Tiles & Crowns & Crown px & Canopy px \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    open(OUT, "w").write(tex)
    print(tex)


if __name__ == "__main__":
    main()
