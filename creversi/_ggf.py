"""GGF (Generic Game Format) parser for Othello records.

Mirrors the behavior of the C++ ``__GgfParser`` in ``ggf_parser.h``:
extracts player names, ratings, results, result flags, moves and
per-move evaluations.
"""

from __future__ import annotations

from ._hand import to_hand

NONE = 0
RESIGNED = 1
TIME_OUT = 2
MUTUAL_SCORE = 3


class GgfParser:
    """Iterative GGF parser. Use :meth:`parse_str` or :meth:`parse_file`."""

    def __init__(self):
        self.names = []
        self.ratings = []
        self.results = []
        self.result_flags = []
        self.moves = []
        self.evaluations = []

    def parse_file(self, path):
        with open(path, "r") as f:
            self.parse_str(f.read())

    def parse_str(self, s):
        self.names = []
        self.ratings = []
        self.results = []
        self.result_flags = []
        self.moves = []
        self.evaluations = []

        n = len(s)
        i = 0
        game_n = 0
        inside = False
        while i < n:
            c = s[i]
            if c == " " or c == "\r" or c == "\n":
                i += 1
                continue

            if i + 1 < n and (s[i:i + 2] == "B[" or s[i:i + 2] == "W["):
                start = i + 2
                j = start
                while j < n and s[j] != "]":
                    j += 1
                tok = s[start:j]
                slash1 = tok.find("/")
                if slash1 != -1:
                    move_str = tok[:slash1]
                    rest = tok[slash1 + 1:]
                    slash2 = rest.find("/")
                    if slash2 == -1:
                        eval_str = rest
                    else:
                        eval_str = rest[:slash2]
                    if eval_str == "":
                        self.evaluations[game_n].append(float("nan"))
                    else:
                        try:
                            self.evaluations[game_n].append(float(eval_str))
                        except ValueError:
                            self.evaluations[game_n].append(float("nan"))
                else:
                    move_str = tok
                try:
                    self.moves[game_n].append(to_hand(move_str))
                except (ValueError, IndexError):
                    pass
                i = j + 1
                continue

            if i + 1 < n and s[i:i + 2] == "(;":
                self.names.append(["", ""])
                self.ratings.append([0.0, 0.0])
                self.results.append(0)
                self.result_flags.append(NONE)
                self.moves.append([])
                self.evaluations.append([])
                inside = True
                i += 2
                continue

            if i + 1 < n and s[i:i + 2] == ";)":
                game_n += 1
                inside = False
                i += 2
                continue

            if s.startswith("GM[Othello]", i):
                i += 11
                continue

            for tag in ("PC[", "DT[", "TI[", "TB[", "TW["):
                if s.startswith(tag, i):
                    i += 3
                    while i < n and s[i] != "]":
                        i += 1
                    i += 1
                    break
            else:
                if s.startswith("PB[", i):
                    i += 3
                    start = i
                    while i < n and s[i] != "]":
                        i += 1
                    self.names[game_n][0] = s[start:i]
                    i += 1
                    continue
                if s.startswith("PW[", i):
                    i += 3
                    start = i
                    while i < n and s[i] != "]":
                        i += 1
                    self.names[game_n][1] = s[start:i]
                    i += 1
                    continue
                if s.startswith("RB[", i):
                    i += 3
                    start = i
                    while i < n and s[i] != "]":
                        i += 1
                    try:
                        self.ratings[game_n][0] = float(s[start:i])
                    except ValueError:
                        self.ratings[game_n][0] = 0.0
                    i += 1
                    continue
                if s.startswith("RW[", i):
                    i += 3
                    start = i
                    while i < n and s[i] != "]":
                        i += 1
                    try:
                        self.ratings[game_n][1] = float(s[start:i])
                    except ValueError:
                        self.ratings[game_n][1] = 0.0
                    i += 1
                    continue
                if s.startswith("TY[8]", i):
                    i += 5
                    continue
                if s.startswith("RE[", i):
                    i += 3
                    start = i
                    while i < n and s[i] != "]":
                        i += 1
                    body = s[start:i]
                    flag = NONE
                    colon = body.find(":")
                    if colon != -1:
                        f = body[colon + 1:]
                        if f == "r":
                            flag = RESIGNED
                        elif f == "t":
                            flag = TIME_OUT
                        elif f == "s":
                            flag = MUTUAL_SCORE
                        body = body[:colon]
                    try:
                        self.results[game_n] = int(float(body))
                    except ValueError:
                        self.results[game_n] = 0
                    self.result_flags[game_n] = flag
                    i += 1
                    continue
                if s.startswith("BO[", i):
                    # Skip the BO[...] header in either the spaced or
                    # compact form. The original C++ parser only
                    # recognised one specific form; we accept any
                    # 8x8 initial-position descriptor.
                    j = s.find("]", i)
                    if j != -1:
                        i = j + 1
                        continue

                # Unrecognised token: skip up to the next ';)' or '(;'.
                if inside:
                    j = s.find(";)", i)
                    if j == -1:
                        i = n
                    else:
                        # drop the partially-collected game
                        self.names.pop()
                        self.ratings.pop()
                        self.results.pop()
                        self.result_flags.pop()
                        self.moves.pop()
                        self.evaluations.pop()
                        inside = False
                        i = j + 2
                else:
                    j = s.find("(;", i)
                    if j == -1:
                        i = n
                    else:
                        i = j

    def game_num(self):
        return len(self.names)
