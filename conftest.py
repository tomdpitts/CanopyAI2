# Empty root conftest: its presence makes pytest add the repository root to
# sys.path (prepend import mode), so `import shadow_prior` resolves when running
# `pytest` from anywhere in the tree.
