# Legacy C++ sources

These files are the original AVX2-based C++ implementation that powered
`creversi` via Cython before the pure-Python migration. They are kept
in-tree as a reference / fall-back, but they are **not built by
`setup.py`** any more.

If you need the AVX2 path again, the build instructions are preserved
in earlier git history (commit prior to the pure-Python migration).
