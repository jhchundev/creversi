"""String <-> integer move conversions, matching the C++ ``hand`` API."""

from __future__ import annotations

PASS = 64
NOMOVE = -1


def to_hand(s):
    """Parse a move string ('a1'..'h8' or 'pass') into an int 0..64."""
    if s in ("ps", "pass", "pa", "PS", "PASS", "PA"):
        return PASS
    if len(s) != 2:
        raise ValueError("invalid hand_str: " + repr(s))
    c0, c1 = s[0], s[1]
    if "a" <= c0 <= "h":
        j = ord(c0) - ord("a")
    elif "A" <= c0 <= "H":
        j = ord(c0) - ord("A")
    else:
        raise ValueError("invalid hand_str: " + repr(s))
    if not ("1" <= c1 <= "8"):
        raise ValueError("invalid hand_str: " + repr(s))
    i = ord(c1) - ord("1")
    return i * 8 + j


def to_s(h):
    """Lowercase string form: 'a1'..'h8' or 'pass'."""
    if h == PASS:
        return "pass"
    return chr(ord("a") + (h % 8)) + chr(ord("1") + (h // 8))


def to_S(h):
    """Uppercase string form: 'A1'..'H8' or 'pass'."""
    if h == PASS:
        return "pass"
    return chr(ord("A") + (h % 8)) + chr(ord("1") + (h // 8))


def hand_from_diff(old_player, old_opponent, new_player, new_opponent):
    """Recover the played move by diffing two boards (placed-stone bit)."""
    old_bits = (old_player | old_opponent) & 0xFFFFFFFFFFFFFFFF
    new_bits = (new_player | new_opponent) & 0xFFFFFFFFFFFFFFFF
    diff = new_bits & ~old_bits
    if diff == 0:
        return PASS
    return (diff & -diff).bit_length() - 1
