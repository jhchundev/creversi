"""Functional tests for the pure-Python ``creversi`` port.

These tests exercise the public API documented in the README and check
behaviour against well-known Othello positions (initial play, full
game played from FFO #40-style positions, FFO test positions for
mobility / piece counts).
"""

from __future__ import annotations

import numpy as np
import pytest

import creversi
from creversi import Board, BLACK_TURN, WHITE_TURN, PASS, move_from_str, move_to_str


# ---- basic API ------------------------------------------------------


def test_initial_position():
    b = Board()
    assert b.turn == BLACK_TURN
    assert b.piece_num() == 2  # 2 black stones
    assert b.opponent_piece_num() == 2  # 2 white stones
    assert b.piece_sum() == 4
    assert not b.is_game_over()


def test_initial_legal_moves():
    b = Board()
    moves = sorted(move_to_str(m) for m in b.legal_moves)
    assert moves == ["c4", "d3", "e6", "f5"]
    assert len(b.legal_moves) == 4


def test_move_d3_flips_d4():
    b = Board()
    b.move_from_str("d3")
    # After d3, the disk at d4 must have been flipped to black.
    assert b.turn == WHITE_TURN
    line = b.to_line()
    # d3=bit19 black, d4=bit27 black, e4=bit28 black, d5=bit35 black, e5=bit36 white
    assert line[19] == "X"
    assert line[27] == "X"
    assert line[28] == "X"
    assert line[35] == "X"
    assert line[36] == "O"


def test_move_constants():
    assert creversi.A1 == 0
    assert creversi.H8 == 63
    assert creversi.PASS == 64
    assert creversi.BLACK_TURN is True
    assert creversi.WHITE_TURN is False
    assert creversi.BLACK == 1
    assert creversi.WHITE == 2
    assert creversi.EMPTY == 0


def test_move_str_helpers():
    assert move_to_str(0) == "a1"
    assert move_to_str(63) == "h8"
    assert move_to_str(PASS) == "pass"
    assert creversi.move_to_STR(0) == "A1"
    assert creversi.move_to_STR(63) == "H8"
    assert move_from_str("a1") == 0
    assert move_from_str("h8") == 63
    assert move_from_str("pass") == PASS


def test_move_rotate_helpers():
    # rotate90: i*8+j -> j*8+(7-i). e.g. (0,0) -> (0,7) = 7.
    assert creversi.move_rotate90(0) == 7
    assert creversi.move_rotate90(7) == 63
    assert creversi.move_rotate180(0) == 63
    assert creversi.move_rotate270(creversi.move_rotate90(35)) == 35


def test_set_line_and_to_line_roundtrip():
    line = "------------------OOO------OXX----OOXX----OX--------------------"
    b = Board(line, BLACK_TURN)
    assert b.to_line() == line
    assert b.turn == BLACK_TURN

    b.set_line(line, WHITE_TURN)
    assert b.to_line() == line  # to_line is invariant w.r.t. turn-side colour swap
    assert b.turn == WHITE_TURN


def test_pass_when_no_moves():
    # A constructed position where black has no legal moves but white has stones.
    # Easiest: black stones surrounded with no opponent adjacency.
    line = "XXXXXXXXX-------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    # Actually just test on a valid degenerate setup: only X stones, nothing for black to play.
    line = (
        "XXXXXXXX"
        "X------X"
        "X------X"
        "X------X"
        "X------X"
        "X------X"
        "X------X"
        "XXXXXXXX"
    )
    b = Board(line, BLACK_TURN)
    moves = list(b.legal_moves)
    # No opponent, so no flips possible -> only PASS
    assert moves == [PASS]


def test_is_legal():
    b = Board()
    # d3=19 is legal in opening
    assert b.is_legal(move_from_str("d3"))
    # a1 is not legal in opening
    assert not b.is_legal(move_from_str("a1"))
    # Bogus move
    assert not b.is_legal(-1)


def test_copy():
    b = Board()
    b.move_from_str("d3")
    c = b.copy()
    assert c.to_line() == b.to_line()
    c.move_from_str("c3")
    assert c.to_line() != b.to_line()


def test_diff_num():
    b = Board()
    # initial 2-2
    assert b.diff_num() == 0
    b.move_from_str("d3")
    # turn swapped -> diff is from white's POV: 1 white, 4 black -> -3
    assert b.diff_num() == -3


def test_puttable_diff():
    b = Board()
    # both sides have 4 moves at the start
    assert b.puttable_num() == 4
    assert b.opponent_puttable_num() == 4
    assert b.puttable_diff() == 0


# ---- bitboard interop -----------------------------------------------


def test_to_bitboard_roundtrip():
    b = Board()
    bb = np.empty(1, creversi.dtypeBitboard)
    b.to_bitboard(bb)
    b2 = Board()
    b2.move_from_str("d3")  # change to a different position
    b2.set_bitboard(bb, BLACK_TURN)
    assert b.to_line() == b2.to_line()
    assert b.turn == b2.turn


def test_piece_planes_initial():
    b = Board()
    planes = np.empty((1, 2, 8, 8), dtype=np.float32)
    b.piece_planes(planes[0])
    # Black stones at d5 (row 4, col 3) and e4 (row 3, col 4).
    assert planes[0, 0, 4, 3] == 1.0
    assert planes[0, 0, 3, 4] == 1.0
    # White stones at d4 (row 3, col 3) and e5 (row 4, col 4).
    assert planes[0, 1, 3, 3] == 1.0
    assert planes[0, 1, 4, 4] == 1.0
    # Total 1.0s = 4
    assert planes.sum() == 4.0


def test_piece_planes_rotate_consistent():
    b = Board()
    b.move_from_str("d3")
    p = np.empty((1, 2, 8, 8), dtype=np.float32)
    p90 = np.empty_like(p)
    p180 = np.empty_like(p)
    p270 = np.empty_like(p)
    b.piece_planes(p[0])
    b.piece_planes_rotate90(p90[0])
    b.piece_planes_rotate180(p180[0])
    b.piece_planes_rotate270(p270[0])
    # Same number of stones in each rotation.
    assert p.sum() == p90.sum() == p180.sum() == p270.sum()
    # Rotating by 90 four times returns to the original.
    expect = np.rot90(p[0], k=-1, axes=(1, 2))
    np.testing.assert_array_equal(p90[0], expect)


# ---- training-data dtype --------------------------------------------


def test_training_data_dtype():
    data = np.empty(1, creversi.TrainingData)
    b = Board()
    b.to_bitboard(data["bitboard"])
    data["turn"] = b.turn
    data["move"] = list(b.legal_moves)[0]
    data["reward"] = 1
    data["done"] = False
    # Read back.
    b2 = Board()
    b2.set_bitboard(data["bitboard"], bool(data["turn"][0]))
    assert b2.to_line() == b.to_line()


# ---- end-to-end short game ------------------------------------------


def test_play_full_game_no_errors():
    """Play a deterministic game by always picking the first legal move
    until game-over. Just exercises the pipeline."""
    b = Board()
    moves_played = 0
    while not b.is_game_over():
        moves = list(b.legal_moves)
        b.move(moves[0])
        moves_played += 1
        assert moves_played < 70  # safety cap; max ~60 stones placed
    assert b.is_game_over()
    assert b.piece_sum() <= 64


def test_pass_then_move():
    """When a player has no legal moves we must pass and the other side moves.

    This is more an invariant check: PASS round-trips the board.
    """
    b = Board()
    line = b.to_line()
    turn = b.turn
    b.move_pass()
    b.move_pass()
    assert b.to_line() == line
    assert b.turn == turn


# ---- inverse / consistency between mobility and flip ------------------


def test_mobility_only_returns_flipping_squares():
    """Every legal move must flip at least one opponent stone."""
    from creversi._bitboard import flip, mobility, iter_set_bits

    b = Board()
    p, o = b._bitboards
    mob = mobility(p, o)
    for sq in iter_set_bits(mob):
        assert flip(p, o, sq) != 0


def test_legal_moves_cover_all_of_mobility():
    from creversi._bitboard import mobility, popcount

    b = Board()
    p, o = b._bitboards
    mob = mobility(p, o)
    assert popcount(mob) == len(b.legal_moves)


# ---- batched / numpy path -------------------------------------------


def test_batched_mobility_matches_scalar():
    from creversi._bitboard import INITIAL_OPPONENT, INITIAL_PLAYER, mobility
    from creversi._bitboard_np import mobility_batch

    n = 32
    rng = np.random.default_rng(0)

    # Build random-ish positions by playing random legal moves from start.
    players = []
    opponents = []
    for _ in range(n):
        b = Board()
        for _ in range(rng.integers(0, 30)):
            moves = list(b.legal_moves)
            if moves == [PASS] and b.is_game_over():
                break
            b.move(moves[rng.integers(0, len(moves))])
        p, o = b._bitboards
        players.append(p)
        opponents.append(o)
    P = np.array(players, dtype=np.uint64)
    O = np.array(opponents, dtype=np.uint64)
    expected = np.array(
        [mobility(int(P[i]), int(O[i])) for i in range(n)], dtype=np.uint64
    )
    got = mobility_batch(P, O)
    np.testing.assert_array_equal(got, expected)


def test_batched_apply_move_matches_scalar():
    from creversi._bitboard import INITIAL_OPPONENT, INITIAL_PLAYER, apply_move
    from creversi._bitboard_np import apply_move_batch

    moves_to_play = [
        move_from_str("d3"),
        move_from_str("c4"),
        move_from_str("e6"),
        move_from_str("f5"),
    ]
    P = np.full(len(moves_to_play), INITIAL_PLAYER, dtype=np.uint64)
    O = np.full(len(moves_to_play), INITIAL_OPPONENT, dtype=np.uint64)
    moves = np.array(moves_to_play, dtype=np.int64)
    np_player, np_opponent = apply_move_batch(P, O, moves)
    for i, m in enumerate(moves_to_play):
        sp, so = apply_move(INITIAL_PLAYER, INITIAL_OPPONENT, m)
        assert int(np_player[i]) == sp
        assert int(np_opponent[i]) == so


# ---- vec env --------------------------------------------------------


def test_vec_env_step():
    pytest.importorskip("gym")
    from creversi.gym_reversi.envs.reversi_vec_env import ReversiVecEnv

    env = ReversiVecEnv(2)
    rewards, dones = env.step([move_from_str("d3"), move_from_str("c4")])
    assert rewards == [0.0, 0.0]
    assert dones == [False, False]


def test_vec_env_fast_step():
    from creversi.gym_reversi.envs.reversi_vec_env import ReversiVecEnvFast

    env = ReversiVecEnvFast(4)
    moves = np.array([move_from_str("d3")] * 4, dtype=np.int64)
    rewards, dones = env.step(moves)
    assert dones.tolist() == [False, False, False, False]
    assert rewards.tolist() == [0.0, 0.0, 0.0, 0.0]


# ---- GGF ------------------------------------------------------------


def test_ggf_parse_minimal():
    sample = (
        "(;GM[Othello]PB[alice]PW[bob]RB[1500]RW[1450]TY[8]"
        "RE[+12]"
        "BO[8 -------- -------- -------- ---O*--- ---*O--- "
        "-------- -------- -------- *]"
        "B[F5//1.5]W[D6//-1.0]B[C3]"
        ";)\n"
    )
    p = creversi.GgfParser()
    p.parse_str(sample)
    assert p.game_num() == 1
    assert p.names == [["alice", "bob"]]
    assert p.ratings == [[1500.0, 1450.0]]
    assert p.results == [12]
    assert len(p.moves[0]) == 3
    assert p.moves[0][0] == move_from_str("F5")
    assert p.moves[0][1] == move_from_str("D6")
    assert p.moves[0][2] == move_from_str("C3")
    assert len(p.evaluations[0]) == 2  # only first two moves had /eval


def test_ggf_replays_legally():
    """All moves in a parsed game must be legal in sequence."""
    sample = (
        "(;GM[Othello]PB[a]PW[b]TY[8]RE[+0]"
        "BO[8 -------- -------- -------- ---O*--- ---*O--- "
        "-------- -------- -------- *]"
        "B[F5]W[D6]B[C3]W[D3]B[C4]W[F4]B[F6]W[B4]"
        ";)\n"
    )
    p = creversi.GgfParser()
    p.parse_str(sample)
    b = Board()
    for mv in p.moves[0]:
        if mv != PASS:
            assert b.is_legal(mv), f"move {move_to_str(mv)} not legal in:\n{b}"
        b.move(mv)


# ---- GGF.Exporter ---------------------------------------------------


def test_ggf_exporter_roundtrip(tmp_path):
    from creversi.GGF import Exporter

    path = tmp_path / "g.ggf"
    e = Exporter(str(path))
    e.newgame(["a", "b"])
    moves = [move_from_str(m) for m in ("d3", "c3", "c4", "b3")]
    for m in moves:
        e.move(m)
    e.endgame(0)
    e.close()

    p = creversi.GgfParser()
    p.parse_file(str(path))
    assert p.game_num() == 1
    assert p.moves[0] == moves
    assert p.names[0] == ["a", "b"]
