"""Batched Reversi environment.

The implementation keeps two parallel paths:

* ``ReversiVecEnv`` — list-of-envs path (matches the original Cython
  semantics exactly: returns ``(rewards, dones)`` and resets done envs
  in place).
* ``ReversiVecEnvFast`` — opt-in NumPy/Numba batched path that operates
  on aligned ``np.uint64`` arrays for the whole batch at once. Useful
  for self-play / data generation pipelines.
"""

from __future__ import annotations

import numpy as np

from creversi.gym_reversi.envs.reversi_env import ReversiEnv

from creversi._bitboard import (
    INITIAL_OPPONENT,
    INITIAL_PLAYER,
    PASS,
)
from creversi._bitboard_np import (
    apply_move_batch,
    is_game_over_batch,
    mobility_batch,
)


class ReversiVecEnv:
    """Drop-in replacement for the Cython ``ReversiVecEnv``."""

    metadata = {"render.modes": ["line"]}

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [ReversiEnv() for _ in range(num_envs)]

    def reset(self):
        for env in self.envs:
            env.reset()

    def render(self, mode="line"):
        return [env.render(mode="line") for env in self.envs]

    def step(self, moves):
        rewards = []
        dones = []
        for i, move in enumerate(moves):
            _, reward, done, _ = self.envs[i].step(move)
            rewards.append(reward)
            dones.append(done)
            if done:
                self.envs[i].reset()
        return rewards, dones


class ReversiVecEnvFast:
    """NumPy-batched vector env. Same semantics as :class:`ReversiVecEnv` but
    keeps state in two ``uint64`` arrays of length ``num_envs``.

    Reward convention: from the perspective of the side that just moved
    (i.e. who is now waiting). ``+1`` win, ``-1`` loss, ``0`` draw.
    """

    metadata = {"render.modes": ["line"]}

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.player = np.full(num_envs, INITIAL_PLAYER, dtype=np.uint64)
        self.opponent = np.full(num_envs, INITIAL_OPPONENT, dtype=np.uint64)
        self.is_black_turn = np.ones(num_envs, dtype=bool)

    def reset(self, indices=None):
        if indices is None:
            self.player[:] = INITIAL_PLAYER
            self.opponent[:] = INITIAL_OPPONENT
            self.is_black_turn[:] = True
        else:
            self.player[indices] = INITIAL_PLAYER
            self.opponent[indices] = INITIAL_OPPONENT
            self.is_black_turn[indices] = True

    def legal_moves_mask(self):
        """Return a uint64 array of legal-move bitboards, one per env."""
        return mobility_batch(self.player, self.opponent)

    def step(self, moves):
        moves = np.asarray(moves, dtype=np.int64)
        # Apply every move in parallel.
        new_player, new_opponent = apply_move_batch(
            self.player, self.opponent, moves
        )
        self.player = new_player
        self.opponent = new_opponent
        self.is_black_turn = ~self.is_black_turn

        # Game-over check.
        done = is_game_over_batch(self.player, self.opponent)

        # diff_num is from the perspective of the new "player" (side to
        # move). The mover that just placed is now `opponent`. So reward
        # for the mover is the negation of diff_num().
        if done.any():
            popcnt = np.array(
                [
                    bin(int(self.player[k])).count("1")
                    - bin(int(self.opponent[k])).count("1")
                    for k in range(self.num_envs)
                ],
                dtype=np.int32,
            )
            rewards = np.where(
                done,
                np.where(popcnt > 0, -1.0, np.where(popcnt < 0, 1.0, 0.0)),
                0.0,
            ).astype(np.float32)
        else:
            rewards = np.zeros(self.num_envs, dtype=np.float32)

        # Reset done envs in place.
        if done.any():
            self.reset(np.where(done)[0])

        return rewards, done
