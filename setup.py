"""Pure-Python setup for ``creversi``.

This package was originally a Cython/C++ extension targeting Windows +
AVX2. It has been migrated to a pure-Python implementation that:

* requires no C/C++ compiler at install time
* works on all Python 3.8+ platforms
* uses NumPy for batched bitboard ops
* optionally uses Numba (``pip install creversi[fast]``) for an
  AVX2-class JIT speedup on the batched path

The legacy C++ sources still live in ``legacy_cpp/`` for reference.
"""

from setuptools import setup, find_packages

setup(
    name="creversi",
    version="0.1.0",
    description="Fast pure-Python Reversi/Othello library",
    long_description=open("README.rst", encoding="utf-8").read(),
    long_description_content_type="text/x-rst",
    license="GPL-3.0-only",
    packages=find_packages(exclude=("tests", "legacy_cpp", "legacy_test_cpp")),
    python_requires=">=3.8",
    install_requires=["numpy>=1.20"],
    extras_require={
        "fast": ["numba>=0.55"],
        "gym": ["gym"],
        "test": ["pytest"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Games/Entertainment :: Board Games",
    ],
)
