"""OpenAI Gym environment for a single Reversi board."""

from __future__ import annotations

try:
    import gym
    from gym import spaces
except ImportError:  # pragma: no cover - gym is optional
    gym = None
    spaces = None

import numpy as np

import creversi


if gym is not None:

    class ReversiEnv(gym.Env):
        metadata = {"render.modes": ["human", "svg", "ansi", "line"]}

        def __init__(self):
            super().__init__()
            self.board = creversi.Board()
            self.observation_space = spaces.Box(0, 2, (8, 8), dtype=np.uint8)
            # `action` accepts a move directly. `sample()` may return illegal moves.
            self.action_space = spaces.Discrete(65)

        def reset(self, line=None):
            if line:
                self.board.set_line(line)
            else:
                self.board.reset()
            return self.board

        def render(self, mode="human"):
            if mode == "svg":
                return self.board.to_svg()
            elif mode == "ansi":
                print(self.board)
            elif mode == "line":
                print(self.board.to_line())
            else:
                return self.board

        def step(self, move):
            assert self.board.is_legal(move)
            self.board.move(move)
            done = self.board.is_game_over()
            if done:
                d = self.board.diff_num()
                reward = 1.0 if d < 0 else (-1.0 if d > 0 else 0.0)
            else:
                reward = 0.0
            return self.board, reward, done, None

else:

    class ReversiEnv:  # pragma: no cover
        def __init__(self, *a, **kw):
            raise ImportError(
                "gym is required for ReversiEnv. Install with `pip install gym`."
            )
