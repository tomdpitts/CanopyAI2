"""Labelled tree cover (canopy_frac + crown_frac, capped at 1) of the sparse-canopy slice vs
forest-biome tiles, against FAO canopy-cover thresholds (10% forest; 40% closed forest, FRA 2000).
Tiles: modal_sparse_tcd_multiseed/tile_index.json, seen==False (excludes the 900 training tiles).
Slice = WWF biomes {7,8,9,10,12,13} (237 here; paper drops one unreadable -> 236). Forest = {1..6}."""
import json, statistics as st
d=json.load(open('modal_sparse_tcd_multiseed/tile_index.json'))['tiles']
sl={7,8,9,10,12,13}; forest={1,2,3,4,5,6}
def stats(R):
    cov=[min(1.0,r['canopy_frac']+r['crown_frac']) for r in R]
    return dict(
        n=len(R),median_cover=round(st.median(cov),4),
        frac_below_10pct=round(sum(c<0.10 for c in cov)/len(cov),4),
        frac_below_40pct=round(sum(c<0.40 for c in cov)/len(cov),4))
out={'definition':'labelled cover = canopy_frac + crown_frac per tile, capped at 1; unseen tiles only',
     'sparse_slice_biomes_7_8_9_10_12_13':stats([r for r in d if r.get('biome') in sl and not r.get('seen')]),
     'forest_biomes_1_to_6':stats([r for r in d if r.get('biome') in forest and not r.get('seen')])}
print(json.dumps(out,indent=1)); json.dump(out,open('ablation/results/slice_fao_cover.json','w'),indent=1)
