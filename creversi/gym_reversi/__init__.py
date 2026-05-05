"""Optional Gym integration for ``creversi``.

If ``gym`` is installed, registers the ``Reversi-v0`` environment on
import. Otherwise this package still imports cleanly and the
NumPy-batched :class:`ReversiVecEnvFast` is available without ``gym``.
"""

try:
    from gym.envs.registration import register

    register(
        id="Reversi-v0",
        entry_point="creversi.gym_reversi.envs:ReversiEnv",
    )
except ImportError:
    pass
