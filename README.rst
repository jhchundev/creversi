creversi: 高速な Pure-Python のリバーシライブラリ
==================================================
.. image:: https://img.shields.io/pypi/v/creversi.svg
    :target: https://pypi.python.org/pypi/creversi
    :alt: PyPI package

概要
----

creversi は、盤面管理、合法手生成、機械学習向けのサポートを備えた高速な
リバーシ／オセロ向け Python ライブラリです。

**v0.1.0 から、本ライブラリは Cython + AVX2 C++ 実装から純粋な Python 実装に
移行しました。** これにより以下のメリットがあります。

* C/C++ コンパイラ不要（pip install で即インストール）
* Linux / macOS / Windows ですべて動作
* オプションで ``numba`` を入れると、バッチ処理が AVX2 実装に匹敵する速度に
* 元の API は完全互換 — 既存コードはそのまま動きます

旧 C++ ソースは互換のため ``legacy_cpp/`` 以下に保存されていますが、ビルドは
されません。

以下は、盤を作成して、開始局面で合法手を生成して表示し、1 手打つ処理の例です。

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

機能
----

* Python 3.8 以上をサポート（Cython は不要）

* IPython/Jupyter Notebook と統合

  .. code:: python

      >>> board

  直前の手をハイライトして表示する場合

  .. code:: python

      >>> move = creversi.move_from_str('c3')
      >>> board.to_svg(move)

* テキスト形式で盤面を表示

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

* 打ち手の表現

  打ち手は 0 から 64 の数値で扱う。座標 a1 が 0、b1 が 1、…、h8 が 63 になり、
  パスが 64 になる。ヘルパー関数で文字列形式に変換できる。

  .. code:: python

      >>> move = list(board.legal_moves)[0]
      >>> creversi.move_to_str(move)
      'b2'

  文字列形式から数値の打ち手に変換できる。パスを表す文字列は ``pass`` となる。

  .. code:: python

      >>> creversi.move_from_str('b2')
      9

* 打つ

  .. code:: python

      >>> move = creversi.move_from_str('d3')
      >>> board.move(move)
      >>> board.move_from_str('d3')

* 合法手生成

  .. code:: python

      >>> for move in board.legal_moves:
      ...     print(creversi.move_to_str(move))

* 合法手チェック

  .. code:: python

      >>> board.is_legal(move)
      False

* 手番の表現 (黒番=True / 白番=False)

  .. code:: python

      >>> board.turn
      True
      >>> board.turn == creversi.WHITE_TURN
      False

* 終局判定

  .. code:: python

      >>> board.is_game_over()
      False

* 局面の文字列形式

  .. code:: python

      >>> line = board.to_line()
      >>> board.set_line('------------------OOO------OXX----OOXX----OX--------------------', creversi.BLACK_TURN)

* 石の数の取得

  .. code:: python

      >>> board.piece_sum()
      >>> board.piece_num()           # 手番側
      >>> board.opponent_piece_num()  # 相手番側
      >>> board.diff_num()            # 手番側から見た差
      >>> board.puttable_num()
      >>> board.opponent_puttable_num()

* 局面のビットボード形式 (16 byte = uint64 x 2)

  .. code:: python

      >>> import numpy as np
      >>> bitboard = np.empty(1, creversi.dtypeBitboard)
      >>> board.to_bitboard(bitboard)
      >>> board.set_bitboard(bitboard, creversi.BLACK_TURN)

* 局面の 2 次元ベクトル表現

  石のある位置を 1、それ以外を 0 とした 2 次元ベクトルを、手番側／相手番の 2
  チャンネルで NCHW 形式の ndarray として取得できる。

  .. code:: python

      >>> import numpy as np
      >>> planes = np.empty((1, 2, 8, 8), dtype=np.float32)
      >>> board.piece_planes(planes[0])
      >>> board.piece_planes_rotate90(planes[0])
      >>> board.piece_planes_rotate180(planes[0])
      >>> board.piece_planes_rotate270(planes[0])

* 機械学習向け訓練データ形式

  .. code:: python

      >>> data = np.empty(1, creversi.TrainingData)
      >>> board.to_bitboard(data['bitboard'])
      >>> data['turn'] = board.turn
      >>> data['move'] = list(board.legal_moves)[0]
      >>> data['reward'] = 1
      >>> data['done'] = False

* Gym 環境

  .. code:: python

      >>> import gym
      >>> import creversi.gym_reversi
      >>> env = gym.make('Reversi-v0').unwrapped
      >>> env.reset()
      >>> next_board, reward, done, _ = env.step(move)

  並列実行バージョン (NumPy ベース、Numba があれば JIT)

  .. code:: python

      >>> from creversi.gym_reversi.envs import ReversiVecEnvFast
      >>> vecenv = ReversiVecEnvFast(num_envs=1024)
      >>> rewards, dones = vecenv.step(moves)

インストール
------------

PyPI からインストール

::

    pip install creversi

NumPy バッチ処理を Numba JIT で高速化する場合 (推奨)

::

    pip install "creversi[fast]"

Gym 統合を使う場合

::

    pip install "creversi[gym]"

GitHub からのインストール

::

    pip install git+https://github.com/TadaoYamaoka/creversi

パフォーマンスについて
----------------------

純 Python 実装は、AVX2 を使っていた C++ 実装に比べて単発呼び出しでは
20〜100 倍程度遅くなりますが、自己対局／RL データ生成のような
バッチ処理では、NumPy + Numba の構成で AVX2 とほぼ同等の速度が出ます。

実測例（参考値、64-bit Linux）::

    Pure-Python mobility    : ~220K ops/sec   (4.6 us/op)
    Pure-Python Board.move  : ~225K ops/sec   (4.4 us/op)
    NumPy batched mobility  : ~6M  ops/sec
    Numba batched mobility  : ~170M ops/sec   (5.7 ns/board, B=1024)

謝辞
----

ビット盤の高速化アルゴリズムは
`issen <https://github.com/primenumber/issen>`_ を参考にしています。

ライセンス
----------

creversi は GPL3 の元にライセンスされています。詳細は LICENSE を確認してください。
