"""Low-level bitboard primitives for Reversi/Othello.

Bit layout:
  bit (i*8 + j) is the square at row i (0=top), column j (0='a').
  So square 'a1' = bit 0, 'h8' = bit 63.

Board state is two uint64s: (player, opponent), where ``player`` is the
side to move. After ``apply_move``, the roles swap (the other side is
now to move) — this matches the C++ creversi convention.

These functions use Python's arbitrary-precision int with explicit
masking. For the batched / JIT-accelerated variant see
``_bitboard_np``.
"""

from __future__ import annotations

MASK64 = 0xFFFFFFFFFFFFFFFF
NOT_FILE_A = 0xFEFEFEFEFEFEFEFE
NOT_FILE_H = 0x7F7F7F7F7F7F7F7F

INITIAL_PLAYER = 0x0000000810000000
INITIAL_OPPONENT = 0x0000001008000000

PASS = 64


def mobility(player, opponent):
    """Return a bitboard of all legal move squares for ``player``.

    Classic 8-direction "dumb-7-fill" sweep. Six iterations are
    sufficient because the longest run of opponent stones along any
    direction is 6 (between two anchors).
    """
    empty = ~(player | opponent) & MASK64

    # East (shift left 1, must not wrap from h-file to a-file)
    masked = opponent & NOT_FILE_A
    f = masked & ((player & NOT_FILE_H) << 1)
    f |= masked & ((f & NOT_FILE_H) << 1)
    f |= masked & ((f & NOT_FILE_H) << 1)
    f |= masked & ((f & NOT_FILE_H) << 1)
    f |= masked & ((f & NOT_FILE_H) << 1)
    f |= masked & ((f & NOT_FILE_H) << 1)
    moves = ((f & NOT_FILE_H) << 1) & empty

    # West
    masked = opponent & NOT_FILE_H
    f = masked & ((player & NOT_FILE_A) >> 1)
    f |= masked & ((f & NOT_FILE_A) >> 1)
    f |= masked & ((f & NOT_FILE_A) >> 1)
    f |= masked & ((f & NOT_FILE_A) >> 1)
    f |= masked & ((f & NOT_FILE_A) >> 1)
    f |= masked & ((f & NOT_FILE_A) >> 1)
    moves |= ((f & NOT_FILE_A) >> 1) & empty

    # South (i increases, shift left 8)
    masked = opponent
    f = masked & ((player << 8) & MASK64)
    f |= masked & ((f << 8) & MASK64)
    f |= masked & ((f << 8) & MASK64)
    f |= masked & ((f << 8) & MASK64)
    f |= masked & ((f << 8) & MASK64)
    f |= masked & ((f << 8) & MASK64)
    moves |= ((f << 8) & MASK64) & empty

    # North
    f = masked & (player >> 8)
    f |= masked & (f >> 8)
    f |= masked & (f >> 8)
    f |= masked & (f >> 8)
    f |= masked & (f >> 8)
    f |= masked & (f >> 8)
    moves |= (f >> 8) & empty

    # SE (di=+1, dj=+1) shift left 9, mask not column h
    masked = opponent & NOT_FILE_A
    f = masked & (((player & NOT_FILE_H) << 9) & MASK64)
    f |= masked & (((f & NOT_FILE_H) << 9) & MASK64)
    f |= masked & (((f & NOT_FILE_H) << 9) & MASK64)
    f |= masked & (((f & NOT_FILE_H) << 9) & MASK64)
    f |= masked & (((f & NOT_FILE_H) << 9) & MASK64)
    f |= masked & (((f & NOT_FILE_H) << 9) & MASK64)
    moves |= (((f & NOT_FILE_H) << 9) & MASK64) & empty

    # SW (di=+1, dj=-1) shift left 7, mask not column a
    masked = opponent & NOT_FILE_H
    f = masked & (((player & NOT_FILE_A) << 7) & MASK64)
    f |= masked & (((f & NOT_FILE_A) << 7) & MASK64)
    f |= masked & (((f & NOT_FILE_A) << 7) & MASK64)
    f |= masked & (((f & NOT_FILE_A) << 7) & MASK64)
    f |= masked & (((f & NOT_FILE_A) << 7) & MASK64)
    f |= masked & (((f & NOT_FILE_A) << 7) & MASK64)
    moves |= (((f & NOT_FILE_A) << 7) & MASK64) & empty

    # NE (di=-1, dj=+1) shift right 7
    masked = opponent & NOT_FILE_A
    f = masked & ((player & NOT_FILE_H) >> 7)
    f |= masked & ((f & NOT_FILE_H) >> 7)
    f |= masked & ((f & NOT_FILE_H) >> 7)
    f |= masked & ((f & NOT_FILE_H) >> 7)
    f |= masked & ((f & NOT_FILE_H) >> 7)
    f |= masked & ((f & NOT_FILE_H) >> 7)
    moves |= ((f & NOT_FILE_H) >> 7) & empty

    # NW (di=-1, dj=-1) shift right 9
    masked = opponent & NOT_FILE_H
    f = masked & ((player & NOT_FILE_A) >> 9)
    f |= masked & ((f & NOT_FILE_A) >> 9)
    f |= masked & ((f & NOT_FILE_A) >> 9)
    f |= masked & ((f & NOT_FILE_A) >> 9)
    f |= masked & ((f & NOT_FILE_A) >> 9)
    f |= masked & ((f & NOT_FILE_A) >> 9)
    moves |= ((f & NOT_FILE_A) >> 9) & empty

    return moves & MASK64


def flip(player, opponent, pos):
    """Return the bitboard of opponent stones flipped by playing at ``pos``."""
    pos_bb = (1 << pos) & MASK64
    flipped = 0

    # East
    ray = ((pos_bb & NOT_FILE_H) << 1) & opponent
    ray |= ((ray & NOT_FILE_H) << 1) & opponent
    ray |= ((ray & NOT_FILE_H) << 1) & opponent
    ray |= ((ray & NOT_FILE_H) << 1) & opponent
    ray |= ((ray & NOT_FILE_H) << 1) & opponent
    ray |= ((ray & NOT_FILE_H) << 1) & opponent
    if ((ray & NOT_FILE_H) << 1) & player:
        flipped |= ray

    # West
    ray = ((pos_bb & NOT_FILE_A) >> 1) & opponent
    ray |= ((ray & NOT_FILE_A) >> 1) & opponent
    ray |= ((ray & NOT_FILE_A) >> 1) & opponent
    ray |= ((ray & NOT_FILE_A) >> 1) & opponent
    ray |= ((ray & NOT_FILE_A) >> 1) & opponent
    ray |= ((ray & NOT_FILE_A) >> 1) & opponent
    if ((ray & NOT_FILE_A) >> 1) & player:
        flipped |= ray

    # South
    ray = ((pos_bb << 8) & MASK64) & opponent
    ray |= ((ray << 8) & MASK64) & opponent
    ray |= ((ray << 8) & MASK64) & opponent
    ray |= ((ray << 8) & MASK64) & opponent
    ray |= ((ray << 8) & MASK64) & opponent
    ray |= ((ray << 8) & MASK64) & opponent
    if ((ray << 8) & MASK64) & player:
        flipped |= ray

    # North
    ray = (pos_bb >> 8) & opponent
    ray |= (ray >> 8) & opponent
    ray |= (ray >> 8) & opponent
    ray |= (ray >> 8) & opponent
    ray |= (ray >> 8) & opponent
    ray |= (ray >> 8) & opponent
    if (ray >> 8) & player:
        flipped |= ray

    # SE
    ray = (((pos_bb & NOT_FILE_H) << 9) & MASK64) & opponent
    ray |= (((ray & NOT_FILE_H) << 9) & MASK64) & opponent
    ray |= (((ray & NOT_FILE_H) << 9) & MASK64) & opponent
    ray |= (((ray & NOT_FILE_H) << 9) & MASK64) & opponent
    ray |= (((ray & NOT_FILE_H) << 9) & MASK64) & opponent
    ray |= (((ray & NOT_FILE_H) << 9) & MASK64) & opponent
    if (((ray & NOT_FILE_H) << 9) & MASK64) & player:
        flipped |= ray

    # SW
    ray = (((pos_bb & NOT_FILE_A) << 7) & MASK64) & opponent
    ray |= (((ray & NOT_FILE_A) << 7) & MASK64) & opponent
    ray |= (((ray & NOT_FILE_A) << 7) & MASK64) & opponent
    ray |= (((ray & NOT_FILE_A) << 7) & MASK64) & opponent
    ray |= (((ray & NOT_FILE_A) << 7) & MASK64) & opponent
    ray |= (((ray & NOT_FILE_A) << 7) & MASK64) & opponent
    if (((ray & NOT_FILE_A) << 7) & MASK64) & player:
        flipped |= ray

    # NE
    ray = ((pos_bb & NOT_FILE_H) >> 7) & opponent
    ray |= ((ray & NOT_FILE_H) >> 7) & opponent
    ray |= ((ray & NOT_FILE_H) >> 7) & opponent
    ray |= ((ray & NOT_FILE_H) >> 7) & opponent
    ray |= ((ray & NOT_FILE_H) >> 7) & opponent
    ray |= ((ray & NOT_FILE_H) >> 7) & opponent
    if ((ray & NOT_FILE_H) >> 7) & player:
        flipped |= ray

    # NW
    ray = ((pos_bb & NOT_FILE_A) >> 9) & opponent
    ray |= ((ray & NOT_FILE_A) >> 9) & opponent
    ray |= ((ray & NOT_FILE_A) >> 9) & opponent
    ray |= ((ray & NOT_FILE_A) >> 9) & opponent
    ray |= ((ray & NOT_FILE_A) >> 9) & opponent
    ray |= ((ray & NOT_FILE_A) >> 9) & opponent
    if ((ray & NOT_FILE_A) >> 9) & player:
        flipped |= ray

    return flipped


def apply_move(player, opponent, pos):
    """Apply a move and return the new (player, opponent) tuple.

    The result has player/opponent swapped — ``player`` is always the
    next side to move.
    """
    if pos == PASS:
        return opponent, player
    flipped = flip(player, opponent, pos)
    new_player = opponent & (~flipped & MASK64)
    new_opponent = (player | flipped | ((1 << pos) & MASK64)) & MASK64
    return new_player, new_opponent


def popcount(x):
    return bin(x & MASK64).count("1")


def is_legal_move(player, opponent, move):
    if move == PASS:
        return mobility(player, opponent) == 0
    if move < 0 or move >= 64:
        return False
    return ((mobility(player, opponent) >> move) & 1) == 1


def is_game_over(player, opponent):
    if mobility(player, opponent) != 0:
        return False
    return mobility(opponent, player) == 0


def stones(player, opponent):
    return (player | opponent) & MASK64


def stone_sum(player, opponent):
    return popcount(stones(player, opponent))


def move_rotate90(move):
    if move == PASS:
        return move
    i, j = divmod(move, 8)
    return j * 8 + (7 - i)


def move_rotate180(move):
    if move == PASS:
        return move
    return 63 - move


def move_rotate270(move):
    if move == PASS:
        return move
    i, j = divmod(move, 8)
    return (7 - j) * 8 + i


def line_to_bitboards(line, is_black_turn):
    """Parse an FFO-style 64-char position string ('X', 'O', '-')."""
    if len(line) != 64:
        raise ValueError("line must be 64 characters")
    black = 0
    white = 0
    for i, ch in enumerate(line):
        if ch == "X":
            black |= 1 << i
        elif ch == "O":
            white |= 1 << i
    if is_black_turn:
        return black, white
    else:
        return white, black


def bitboards_to_ffo(player, opponent, is_black_turn):
    if is_black_turn:
        black, white = player, opponent
    else:
        black, white = opponent, player
    out = []
    for i in range(64):
        bit = 1 << i
        if black & bit:
            out.append("X")
        elif white & bit:
            out.append("O")
        else:
            out.append("-")
    return "".join(out)


def bitboards_to_dump(player, opponent, is_black_turn):
    if is_black_turn:
        black, white = player, opponent
    else:
        black, white = opponent, player
    rows = [" |abcdefgh", "-+--------"]
    for i in range(8):
        row = chr(ord("1") + i) + "|"
        for j in range(8):
            sq = i * 8 + j
            bit = 1 << sq
            if black & bit:
                row += "x"
            elif white & bit:
                row += "o"
            else:
                row += "."
        rows.append(row)
    return "\n".join(rows)


def iter_set_bits(bits):
    """Yield each set-bit position in a uint64, low-bit first."""
    bits &= MASK64
    while bits:
        lsb = bits & -bits
        yield lsb.bit_length() - 1
        bits ^= lsb
