creversi: a fast pure-Python Reversi library
============================================
.. image:: https://img.shields.io/pypi/v/creversi-python.svg
    :target: https://pypi.python.org/pypi/creversi-python
    :alt: PyPI package
.. image:: https://img.shields.io/pypi/pyversions/creversi-python.svg
    :target: https://pypi.python.org/pypi/creversi-python
    :alt: Supported Python versions
.. image:: https://img.shields.io/pypi/l/creversi-python.svg
    :target: https://pypi.python.org/pypi/creversi-python
    :alt: License

Overview
--------

creversi is a fast Python library for Reversi/Othello with board management,
legal-move generation, and machine-learning friendly utilities.

**Starting with v0.1.0, this library has been migrated from a Cython + AVX2
C++ implementation to a pure-Python implementation.** This brings several
benefits:

* No C/C++ compiler required (installs directly via ``pip install``).
* Works on Linux, macOS, and Windows.
* Optionally install ``numba`` to speed batched operations up to AVX2-class
  performance.
* Fully API-compatible with the previous release — existing code keeps
  working.

The original C++ sources are preserved under ``legacy_cpp/`` for reference
but are no longer built.

The example below creates a board, lists the legal moves at the opening
position, and plays one move:

.. code:: python

    >>> import creversi

    >>> board = creversi.Board()

    >>> for move in board.legal_moves:
    ...     print(creversi.move_to_str(move))

::

    d3
    c4
    f5
    e6

.. code:: python

    >>> board.move_from_str('d3')

Features
--------

* Supports Python 3.8+ (no Cython required).

* IPython / Jupyter Notebook integration.

  .. code:: python

      >>> board

  Highlight the most recent move:

  .. code:: python

      >>> move = creversi.move_from_str('c3')
      >>> board.to_svg(move)

* Render the board as text.

  .. code:: python

      >>> board = creversi.Board('------------------OOO------OXX----OOXX----OX--------------------', creversi.BLACK_TURN)
      >>> print(board)

  ::

         |abcdefgh
        -+--------
        1|........
        2|........
        3|..ooo...
        4|...oxx..
        5|..ooxx..
        6|..ox....
        7|........
        8|........

* Move encoding.

  Moves are integers from 0 to 64. Square ``a1`` is 0, ``b1`` is 1, …,
  ``h8`` is 63, and 64 means "pass". Helper functions convert to and from
  the string form.

  .. code:: python

      >>> move = list(board.legal_moves)[0]
      >>> creversi.move_to_str(move)
      'b2'

  String form back to integer; ``'pass'`` represents a pass.

  .. code:: python

      >>> creversi.move_from_str('b2')
      9

* Play a move.

  .. code:: python

      >>> move = creversi.move_from_str('d3')
      >>> board.move(move)
      >>> board.move_from_str('d3')

* Legal-move generation.

  .. code:: python

      >>> for move in board.legal_moves:
      ...     print(creversi.move_to_str(move))

* Legality check.

  .. code:: python

      >>> board.is_legal(move)
      False

* Side to move (Black = ``True``, White = ``False``).

  .. code:: python

      >>> board.turn
      True
      >>> board.turn == creversi.WHITE_TURN
      False

* Game-over detection.

  .. code:: python

      >>> board.is_game_over()
      False

* Position string format.

  .. code:: python

      >>> line = board.to_line()
      >>> board.set_line('------------------OOO------OXX----OOXX----OX--------------------', creversi.BLACK_TURN)

* Piece counts.

  .. code:: python

      >>> board.piece_sum()
      >>> board.piece_num()           # side to move
      >>> board.opponent_piece_num()  # opponent
      >>> board.diff_num()            # difference from the side-to-move's perspective
      >>> board.puttable_num()
      >>> board.opponent_puttable_num()

* Bitboard representation (16 bytes = ``uint64`` x 2).

  .. code:: python

      >>> import numpy as np
      >>> bitboard = np.empty(1, creversi.dtypeBitboard)
      >>> board.to_bitboard(bitboard)
      >>> board.set_bitboard(bitboard, creversi.BLACK_TURN)

* 2-D plane representation.

  Returns an ``ndarray`` in NCHW format with two channels (side to move
  and opponent), where occupied squares are 1 and the rest are 0.

  .. code:: python

      >>> import numpy as np
      >>> planes = np.empty((1, 2, 8, 8), dtype=np.float32)
      >>> board.piece_planes(planes[0])
      >>> board.piece_planes_rotate90(planes[0])
      >>> board.piece_planes_rotate180(planes[0])
      >>> board.piece_planes_rotate270(planes[0])

* Training-data layout for machine learning.

  .. code:: python

      >>> data = np.empty(1, creversi.TrainingData)
      >>> board.to_bitboard(data['bitboard'])
      >>> data['turn'] = board.turn
      >>> data['move'] = list(board.legal_moves)[0]
      >>> data['reward'] = 1
      >>> data['done'] = False

* Gym environment.

  .. code:: python

      >>> import gym
      >>> import creversi.gym_reversi
      >>> env = gym.make('Reversi-v0').unwrapped
      >>> env.reset()
      >>> next_board, reward, done, _ = env.step(move)

  Vectorized version (NumPy-based, JIT-compiled if Numba is installed):

  .. code:: python

      >>> from creversi.gym_reversi.envs import ReversiVecEnvFast
      >>> vecenv = ReversiVecEnvFast(num_envs=1024)
      >>> rewards, dones = vecenv.step(moves)

Installation
------------

Install from PyPI:

::

    pip install creversi-python

Accelerate the NumPy batched path with the Numba JIT (recommended):

::

    pip install "creversi-python[fast]"

With Gym integration:

::

    pip install "creversi-python[gym]"

Install from GitHub:

::

    pip install git+https://github.com/jhchundev/creversi

Performance
-----------

The pure-Python implementation is roughly 20–100x slower than the original
AVX2 C++ implementation for single-call usage, but the NumPy + Numba batched
path matches AVX2 for workloads such as self-play and RL data generation.

Indicative measurements (64-bit Linux)::

    Pure-Python mobility    : ~220K ops/sec   (4.6 us/op)
    Pure-Python Board.move  : ~225K ops/sec   (4.4 us/op)
    NumPy batched mobility  : ~6M  ops/sec
    Numba batched mobility  : ~170M ops/sec   (5.7 ns/board, B=1024)

Acknowledgements
----------------

The bitboard fast-move algorithm is based on
`issen <https://github.com/primenumber/issen>`_.

License
-------

creversi is licensed under GPL-3.0. See ``LICENSE`` for details.
