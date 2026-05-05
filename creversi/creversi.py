"""Pure-Python ``creversi`` public API.

This module replaces the original Cython extension ``creversi.creversi``.
It re-exports the constants, the :class:`Board` and :class:`LegalMoveList`
classes, the move-string helpers and the GGF parser. The public API
matches the original C++/Cython version one-to-one so existing code
keeps working.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from . import _bitboard as _bb
from ._bitboard import (
    INITIAL_OPPONENT,
    INITIAL_PLAYER,
    MASK64,
    apply_move,
    bitboards_to_dump,
    bitboards_to_ffo,
    flip,
    is_game_over as _is_game_over,
    is_legal_move as _is_legal_move,
    iter_set_bits,
    line_to_bitboards,
    mobility,
    move_rotate90 as _move_rotate90,
    move_rotate180 as _move_rotate180,
    move_rotate270 as _move_rotate270,
    popcount,
    stone_sum,
)
from ._ggf import GgfParser as _GgfParser
from ._ggf import MUTUAL_SCORE, NONE, RESIGNED, TIME_OUT
from ._hand import PASS, to_hand, to_S, to_s

# ---- numpy dtypes (parity with the Cython module) ----

dtypeBitboard = np.dtype((np.uint8, 16))
dtypeTurn = np.dtype(bool)
dtypeMove = np.dtype(np.int8)
dtypeReward = np.dtype(np.int8)
dtypeDone = np.dtype(bool)

TrainingData = np.dtype(
    [
        ("bitboard", dtypeBitboard),
        ("turn", dtypeTurn),
        ("move", dtypeMove),
        ("reward", dtypeReward),
        ("done", dtypeDone),
    ]
)

# ---- square-name constants ----

SQUARES = list(range(64))
(
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
) = SQUARES

BLACK_TURN = True
WHITE_TURN = False
EMPTY = 0
BLACK = 1
WHITE = 2
PIECES = [EMPTY, BLACK, WHITE]
RESULT_FLAGS = [NONE, RESIGNED, TIME_OUT, MUTUAL_SCORE]

# ---- SVG snippets (verbatim from the Cython module) ----

SVG_PIECE_DEFS = [
    '<g id="black"><circle cx="10" cy="10" r="8.5" fill="black"/></g>',
    '<g id="white"><circle cx="10" cy="10" r="8.5" fill="white"/></g>',
]
SVG_PIECE_DEF_IDS = [None, "black", "white"]
SVG_BOARD = '<rect fill="green" height="161" width="161" x="10" y="10" />'
SVG_SQUARES = '<g stroke="black"><rect width="161" height="161" stroke-width="1.5" fill="none" x="10" y="10" /><line stroke-width="1.0" x1="10.5" x2="170.5" y1="30.5" y2="30.5" /><line stroke-width="1.0" x1="10.5" x2="170.5" y1="50.5" y2="50.5" /><line stroke-width="1.0" x1="10.5" x2="170.5" y1="70.5" y2="70.5" /><line stroke-width="1.0" x1="10.5" x2="170.5" y1="90.5" y2="90.5" /><line stroke-width="1.0" x1="10.5" x2="170.5" y1="110.5" y2="110.5" /><line stroke-width="1.0" x1="10.5" x2="170.5" y1="130.5" y2="130.5" /><line stroke-width="1.0" x1="10.5" x2="170.5" y1="150.5" y2="150.5" /><line stroke-width="1.0" x1="30.5" x2="30.5" y1="10.5" y2="170.5" /><line stroke-width="1.0" x1="50.5" x2="50.5" y1="10.5" y2="170.5" /><line stroke-width="1.0" x1="70.5" x2="70.5" y1="10.5" y2="170.5" /><line stroke-width="1.0" x1="90.5" x2="90.5" y1="10.5" y2="170.5" /><line stroke-width="1.0" x1="110.5" x2="110.5" y1="10.5" y2="170.5" /><line stroke-width="1.0" x1="130.5" x2="130.5" y1="10.5" y2="170.5" /><line stroke-width="1.0" x1="150.5" x2="150.5" y1="10.5" y2="170.5" /></g>'
SVG_COORDINATES = '<g><text font-family="serif" font-size="9.5" text-anchor="middle" x="20.5" y="8">a</text><text font-family="serif" font-size="9.5" text-anchor="middle" x="40.5" y="8">b</text><text font-family="serif" font-size="9.5" text-anchor="middle" x="60.5" y="8">c</text><text font-family="serif" font-size="9.5" text-anchor="middle" x="80.5" y="8">d</text><text font-family="serif" font-size="9.5" text-anchor="middle" x="100.5" y="8">e</text><text font-family="serif" font-size="9.5" text-anchor="middle" x="120.5" y="8">f</text><text font-family="serif" font-size="9.5" text-anchor="middle" x="140.5" y="8">g</text><text font-family="serif" font-size="9.5" text-anchor="middle" x="160.5" y="8">h</text><text font-family="serif" font-size="9.5" x="2.5" y="23.5">1</text><text font-family="serif" font-size="9.5" x="2.5" y="43.5">2</text><text font-family="serif" font-size="9.5" x="2.5" y="63.5">3</text><text font-family="serif" font-size="9.5" x="2.5" y="83.5">4</text><text font-family="serif" font-size="9.5" x="2.5" y="103.5">5</text><text font-family="serif" font-size="9.5" x="2.5" y="123.5">6</text><text font-family="serif" font-size="9.5" x="2.5" y="143.5">7</text><text font-family="serif" font-size="9.5" x="2.5" y="163.5">8</text></g>'


class SvgWrapper(str):
    def _repr_svg_(self):
        return self


# ---- Board ----


class Board:
    """A Reversi/Othello position.

    Attributes are not exposed directly; the public surface matches the
    original C++/Cython class.
    """

    __slots__ = ("_player", "_opponent", "_is_black_turn")

    def __init__(self, line=None, is_black_turn=True, board=None):
        if line is not None:
            self._player, self._opponent = line_to_bitboards(line, is_black_turn)
            self._is_black_turn = is_black_turn
        elif board is not None:
            self._player = board._player
            self._opponent = board._opponent
            self._is_black_turn = board._is_black_turn
        else:
            self._player = INITIAL_PLAYER
            self._opponent = INITIAL_OPPONENT
            self._is_black_turn = True

    def __copy__(self):
        return Board(board=self)

    def copy(self):
        return Board(board=self)

    def reset(self):
        self._player = INITIAL_PLAYER
        self._opponent = INITIAL_OPPONENT
        self._is_black_turn = True

    def set_line(self, line, is_black_turn=True):
        self._player, self._opponent = line_to_bitboards(line, is_black_turn)
        self._is_black_turn = is_black_turn

    def set_bitboard(self, bitboard, is_black_turn):
        """Set the position from a 16-byte bitboard ndarray."""
        arr = np.asarray(bitboard, dtype=np.uint8).reshape(-1)
        if arr.size != 16:
            raise ValueError("bitboard must be 16 bytes")
        self._player = int(np.frombuffer(arr[:8].tobytes(), dtype="<u8")[0])
        self._opponent = int(np.frombuffer(arr[8:16].tobytes(), dtype="<u8")[0])
        self._is_black_turn = bool(is_black_turn)

    def to_bitboard(self, bitboard):
        """Write the position into a 16-byte ndarray (modifies in place)."""
        arr = np.asarray(bitboard, dtype=np.uint8).reshape(-1)
        if arr.size != 16:
            raise ValueError("bitboard must be 16 bytes")
        arr[:8] = np.frombuffer(
            np.uint64(self._player).tobytes(), dtype=np.uint8
        )
        arr[8:16] = np.frombuffer(
            np.uint64(self._opponent).tobytes(), dtype=np.uint8
        )

    # ---- play / iterate ----

    def move(self, move):
        if move == PASS:
            self._player, self._opponent = self._opponent, self._player
        else:
            self._player, self._opponent = apply_move(
                self._player, self._opponent, move
            )
        self._is_black_turn = not self._is_black_turn

    def move_from_str(self, s):
        h = to_hand(s)
        self.move(h)
        return h

    def move_pass(self):
        self._player, self._opponent = self._opponent, self._player
        self._is_black_turn = not self._is_black_turn

    def is_legal(self, move):
        return _is_legal_move(self._player, self._opponent, move)

    @property
    def legal_moves(self):
        return LegalMoveList(self)

    def is_game_over(self):
        return _is_game_over(self._player, self._opponent)

    @property
    def turn(self):
        return self._is_black_turn

    # ---- counts ----

    def piece_num(self):
        return popcount(self._player)

    def opponent_piece_num(self):
        return popcount(self._opponent)

    def piece_sum(self):
        return stone_sum(self._player, self._opponent)

    def puttable_num(self):
        return popcount(mobility(self._player, self._opponent))

    def opponent_puttable_num(self):
        return popcount(mobility(self._opponent, self._player))

    def diff_num(self):
        return popcount(self._player) - popcount(self._opponent)

    def puttable_diff(self):
        return self.puttable_num() - self.opponent_puttable_num()

    # ---- string forms ----

    def to_line(self):
        return bitboards_to_ffo(self._player, self._opponent, self._is_black_turn)

    def __repr__(self):
        return bitboards_to_dump(self._player, self._opponent, self._is_black_turn)

    def piece(self, sq):
        bit = 1 << sq
        if self._player & bit:
            return BLACK if self._is_black_turn else WHITE
        if self._opponent & bit:
            return WHITE if self._is_black_turn else BLACK
        return EMPTY

    # ---- planes ----

    def piece_planes(self, features):
        """Write a (2, 8, 8) float32 view of the position."""
        flat = np.asarray(features, dtype=np.float32).reshape(2, 64)
        flat[:] = 0.0
        for sq in iter_set_bits(self._player):
            flat[0, sq] = 1.0
        for sq in iter_set_bits(self._opponent):
            flat[1, sq] = 1.0

    def piece_planes_rotate90(self, features):
        flat = np.asarray(features, dtype=np.float32).reshape(2, 64)
        flat[:] = 0.0
        for sq in iter_set_bits(self._player):
            flat[0, _move_rotate90(sq)] = 1.0
        for sq in iter_set_bits(self._opponent):
            flat[1, _move_rotate90(sq)] = 1.0

    def piece_planes_rotate180(self, features):
        flat = np.asarray(features, dtype=np.float32).reshape(2, 64)
        flat[:] = 0.0
        for sq in iter_set_bits(self._player):
            flat[0, _move_rotate180(sq)] = 1.0
        for sq in iter_set_bits(self._opponent):
            flat[1, _move_rotate180(sq)] = 1.0

    def piece_planes_rotate270(self, features):
        flat = np.asarray(features, dtype=np.float32).reshape(2, 64)
        flat[:] = 0.0
        for sq in iter_set_bits(self._player):
            flat[0, _move_rotate270(sq)] = 1.0
        for sq in iter_set_bits(self._opponent):
            flat[1, _move_rotate270(sq)] = 1.0

    # ---- SVG ----

    def to_svg(self, lastmove=None, scale=1.0):
        svg = ET.Element(
            "svg",
            {
                "xmlns": "http://www.w3.org/2000/svg",
                "version": "1.1",
                "xmlns:xlink": "http://www.w3.org/1999/xlink",
                "width": str(215 * scale),
                "height": str(215 * scale),
                "viewBox": "0 0 172 172",
            },
        )
        defs = ET.SubElement(svg, "defs")
        for piece_def in SVG_PIECE_DEFS:
            defs.append(ET.fromstring(piece_def))
        svg.append(ET.fromstring(SVG_BOARD))

        if lastmove is not None and lastmove != PASS:
            i, j = divmod(lastmove, 8)
            ET.SubElement(
                svg,
                "rect",
                {
                    "x": str(10.5 + j * 20),
                    "y": str(10.5 + i * 20),
                    "width": str(20),
                    "height": str(20),
                    "fill": "#8bbf83",
                },
            )

        svg.append(ET.fromstring(SVG_SQUARES))
        svg.append(ET.fromstring(SVG_COORDINATES))

        for sq in SQUARES:
            pc = self.piece(sq)
            if pc != EMPTY:
                i, j = divmod(sq, 8)
                ET.SubElement(
                    svg,
                    "use",
                    {
                        "xlink:href": "#" + SVG_PIECE_DEF_IDS[pc],
                        "x": str(10.5 + j * 20),
                        "y": str(10.5 + i * 20),
                    },
                )

        return SvgWrapper(ET.tostring(svg).decode("utf-8"))

    def _repr_svg_(self):
        return self.to_svg()

    # internal accessors used by gym envs
    @property
    def _bitboards(self):
        return self._player, self._opponent


class LegalMoveList:
    """Iterable / sized view of legal moves for a board snapshot.

    Iteration order matches the C++ implementation: low-bit-first square
    indices, with a single ``PASS`` (64) emitted when no move is legal.
    """

    __slots__ = ("_bits", "_pass", "_size")

    def __init__(self, board):
        bits = mobility(board._player, board._opponent)
        self._bits = bits
        if bits == 0:
            self._pass = True
            self._size = 1
        else:
            self._pass = False
            self._size = popcount(bits)

    def __iter__(self):
        if self._pass:
            yield PASS
            return
        bits = self._bits
        while bits:
            lsb = bits & -bits
            yield lsb.bit_length() - 1
            bits ^= lsb

    def __len__(self):
        return self._size


# ---- module-level helpers ----


def move_to_str(move):
    return to_s(int(move))


def move_to_STR(move):
    return to_S(int(move))


def move_from_str(s):
    return to_hand(s)


def move_rotate90(move):
    return _move_rotate90(move)


def move_rotate180(move):
    return _move_rotate180(move)


def move_rotate270(move):
    return _move_rotate270(move)


# ---- GGF parser ----


class GgfParser:
    """Wrapper preserving the C++/Cython interface for GgfParser."""

    def __init__(self):
        self._inner = _GgfParser()

    def parse_file(self, path):
        self._inner.parse_file(path)

    def parse_str(self, s):
        self._inner.parse_str(s)

    @property
    def names(self):
        return self._inner.names

    @property
    def ratings(self):
        return self._inner.ratings

    @property
    def results(self):
        return self._inner.results

    @property
    def result_flags(self):
        return self._inner.result_flags

    @property
    def moves(self):
        return self._inner.moves

    @property
    def evaluations(self):
        return self._inner.evaluations

    def game_num(self):
        return self._inner.game_num()


__all__ = [
    # constants
    "BLACK", "WHITE", "EMPTY", "PIECES",
    "BLACK_TURN", "WHITE_TURN",
    "PASS",
    "NONE", "RESIGNED", "TIME_OUT", "MUTUAL_SCORE", "RESULT_FLAGS",
    "SQUARES",
    "A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1",
    "A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2",
    "A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3",
    "A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4",
    "A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5",
    "A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6",
    "A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7",
    "A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8",
    # dtypes
    "dtypeBitboard", "dtypeTurn", "dtypeMove", "dtypeReward", "dtypeDone",
    "TrainingData",
    # SVG bits
    "SVG_PIECE_DEFS", "SVG_PIECE_DEF_IDS", "SVG_BOARD", "SVG_SQUARES",
    "SVG_COORDINATES", "SvgWrapper",
    # core
    "Board", "LegalMoveList",
    # helpers
    "move_to_str", "move_to_STR", "move_from_str",
    "move_rotate90", "move_rotate180", "move_rotate270",
    # ggf
    "GgfParser",
]
