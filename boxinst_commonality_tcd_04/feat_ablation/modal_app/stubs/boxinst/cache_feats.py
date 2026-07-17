"""Modal stub — the real module extracts DINOv3 features; here only the LAYERS
constant is imported (by cache_test/cache_train_tiles, themselves imported only
for path constants). Value matches boxinst/cache_feats.py in the repo."""
LAYERS = (21, 22, 23, 24)          # last 4 blocks of ViT-L/16
