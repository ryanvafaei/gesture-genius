"""
Exercise 9: memory pairs (a card game for memory, played by pointing).

Face-down cards lie over her camera image. She points at one with her index
finger and holds still for a moment to turn it over, then looks for its
twin. Pointing with the left hand adds arm practice for free; either hand
works, and keys 1-8 turn a card over too (e.g. a helper, or on a tired day).

Levels: 2, 3 and then 4 pairs, one more after two good boards in a row
(no more than one turn too many). One rep = one finished board:
  range_high  accuracy (pairs / turns), 1 = no wrong turns
  raw_high    turns per pair (lower is better)
Nothing is a failure: two cards that do not match stay up for a moment
("Not a pair. Try to remember where they are.") and turn back.

The cards stay away from the left edge of the screen (spatial neglect is
common after a right-brain stroke). The pictures are the activity icons.
"""

import numpy as np

from rehab.exercises.base import Exercise, RepRecord, Say

ICONS = ("cup", "trowel", "pot", "book", "button", "phone")
# where the cards lie, as fractions of the camera image (x0, y0, x1, y1)
BOARD = (0.22, 0.14, 0.92, 0.74)
PAIR_PHRASES = ("That's a pair.", "You found a pair.", "Well remembered.")


def layout(n_cards, board=BOARD, gap=0.03):
    """Card rectangles (x0, y0, x1, y1) in two rows, in image fractions."""
    cols = max(1, n_cards // 2)
    x0, y0, x1, y1 = board
    w = (x1 - x0 - gap * (cols - 1)) / cols
    h = (y1 - y0 - gap) / 2
    return [(x0 + c * (w + gap), y0 + r * (h + gap), x0 + c * (w + gap) + w, y0 + r * (h + gap) + h)
            for r in range(2) for c in range(cols)]


class MemoryPairs(Exercise):
    name = "memory_pairs"
    title = "Memory pairs"
    instructions = (
        "Let's play a memory game with picture cards.",
        "Point at a card and hold still to turn it over.",
    )
    any_hand = True
    need_hand = False           # keys work without a hand in view
    uses_level = True
    progress_lower_is_better = True
    progress_phrase = "You found the pairs with {pct}% fewer turns than {when}."

    def __init__(self, *args, level=None, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = rng or np.random.default_rng()
        pairs = {int(k): v for k, v in self.params.get("level_pairs", {1: 2, 2: 3, 3: 4}).items()}
        self._level_pairs = pairs
        self.level = int(min(max(pairs), level or self.params.get("start_level", 1)))
        self._good_boards = 0
        self._board = None
        self._dwell = None
        self._now = 0.0
        self._last_phrase = None

    # --- board ------------------------------------------------------------

    @property
    def pairs(self):
        return self._level_pairs[self.level]

    def _new_board(self, now):
        icons = list(self.rng.choice(ICONS, size=self.pairs, replace=False)) * 2
        self.rng.shuffle(icons)
        rects = layout(len(icons))
        self._board = {
            "cards": [{"icon": str(i), "state": "down", "rect": r} for i, r in zip(icons, rects)],
            "first": None, "mismatch": None, "mismatch_until": None,
            "t_start": now, "turns": 0, "mismatches": 0, "hints": 0,
            "choice_times": [], "last_choice_t": now,
            "choices": 0, "affected_choices": 0,
        }
        self._dwell = None
        first = len(self.reps) == 0
        text = (f"Find the {self.pairs} pairs." if first
                else f"Here is a new board with {self.pairs} pairs.")
        return [Say(text, valid=self._while_board())]

    def _while_board(self):
        b = self._board
        return lambda: self._board is b

    @property
    def found(self):
        if self._board is None:
            return 0
        return sum(1 for c in self._board["cards"] if c["state"] == "matched") // 2

    def first_messages(self):
        return []                   # the board announces itself

    def resume_messages(self):
        return [Say("Point at a card and hold still.")]

    def start_set(self, set_no, now):
        super().start_set(set_no, now)
        self._board = None
        self._dwell = None

    def interrupt(self, now):
        super().interrupt(now)
        self._dwell = None

    # --- choosing a card ------------------------------------------------------

    def card_at(self, x, y):
        """Index of the face-down card under (x, y) in image fractions, or None."""
        if self._board is None:
            return None
        for i, c in enumerate(self._board["cards"]):
            x0, y0, x1, y1 = c["rect"]
            if x0 <= x <= x1 and y0 <= y <= y1 and c["state"] == "down":
                return i
        return None

    def select(self, i, now, affected=True):
        """Turn card i over (pointing, or keys 1-8). Returns what to say."""
        b = self._board
        if b is None or b["mismatch"] is not None:
            return []
        if not 0 <= i < len(b["cards"]) or b["cards"][i]["state"] != "down":
            return []
        self._progress(now)
        self._dwell = None
        b["cards"][i]["state"] = "up"
        b["choice_times"].append(now - b["last_choice_t"])
        b["last_choice_t"] = now
        b["choices"] += 1
        b["affected_choices"] += 1 if affected else 0
        if b["first"] is None:
            b["first"] = i
            return []
        first, b["first"] = b["first"], None
        b["turns"] += 1
        cards = b["cards"]
        if cards[first]["icon"] == cards[i]["icon"]:
            cards[first]["state"] = cards[i]["state"] = "matched"
            out = [Say("", "chime")]
            if self.found == self.pairs:
                return out + self._finish_board(now)
            phrases = [p for p in PAIR_PHRASES if p != self._last_phrase]
            self._last_phrase = str(self.rng.choice(phrases))
            return out + [Say(self._last_phrase, "praise", valid=self._while_board())]
        b["mismatches"] += 1
        b["mismatch"] = (first, i)
        b["mismatch_until"] = now + self.params.get("show_mismatch_s", 2.0)
        return [Say("Not a pair. Try to remember where they are.", "hint",
                    valid=self._while_board())]

    # --- per frame ---------------------------------------------------------------

    def update(self, f, now):
        out = []
        self._now = now
        if self._board is None and self.set_done:
            self.display = self._make_display(None)     # the session moves on
            return out
        if self._board is None:
            out += self._new_board(now)
            self._progress(now)
        b = self._board
        if b["mismatch"] is not None and now >= b["mismatch_until"]:
            for i in b["mismatch"]:
                b["cards"][i]["state"] = "down"
            b["mismatch"] = None
            self._progress(now)

        pointer = self._pointer(f)
        if pointer is not None and b["mismatch"] is None:
            out += self._dwell_update(pointer, now, affected=bool(getattr(f, "correct_hand", True)))
        else:
            self._dwell = None

        if self._board is b and self._stalled(now) and self._hints.ready("stall", now):
            b["hints"] += 1
            self._progress(now)
            out.append(Say("Point at a card and hold still.", "hint", valid=self._while_board()))

        self.display = self._make_display(pointer)
        return out

    @staticmethod
    def _pointer(f):
        """Her index fingertip in image fractions, or None."""
        if f is None or not getattr(f, "present", False) or f.image_points is None:
            return None
        size = f.image_size or (1.0, 1.0)
        tip = f.image_points[8]
        return float(tip[0] / size[0]), float(tip[1] / size[1])

    def _dwell_update(self, pointer, now, affected=True):
        i = self.card_at(*pointer)
        d = self._dwell
        radius = self.params.get("dwell_radius", 0.04)
        if i is None:
            self._dwell = None
            return []
        if d is None or d["card"] != i or np.hypot(pointer[0] - d["anchor"][0],
                                                   pointer[1] - d["anchor"][1]) > radius:
            # a new card, or she moved: start holding again from here
            self._dwell = {"card": i, "anchor": pointer, "since": now}
            return []
        if now - d["since"] >= self.params.get("dwell_s", 1.5):
            return self.select(i, now, affected=affected)
        return []

    def _finish_board(self, now):
        b = self._board
        times = b["choice_times"]
        rec = RepRecord(
            exercise=self.name, set_no=self.set_no, rep_no=self.reps_this_set + 1,
            t_start=b["t_start"], t_end=now,
            range_high=self.pairs / max(1, b["turns"]),
            raw_high=b["turns"] / self.pairs,
            movement_time=float(np.mean(times)) if times else float("nan"),
            hints=b["hints"],
            success=b["hints"] == 0,
            extra={"level": self.level, "pairs": self.pairs, "turns": b["turns"],
                   "mismatches": b["mismatches"],
                   "affected_share": round(b["affected_choices"] / max(1, b["choices"]), 2)},
        )
        self._add_rep(rec)
        out = [Say("Well done, you found all the pairs.", "praise", tag="rep_done")]
        if b["turns"] <= self.pairs + 1:
            self._good_boards += 1
        else:
            self._good_boards = 0
        if (self._good_boards >= self.params.get("boards_to_level_up", 2)
                and self.level < max(self._level_pairs)):
            self.level += 1
            self._good_boards = 0
            out.append(Say("Great memory. Next time there will be one more pair.", "praise"))
        self._board = None
        return out

    def _make_display(self, pointer):
        b = self._board
        cards = []
        if b is not None:
            dwell = self._dwell
            for i, c in enumerate(b["cards"]):
                progress = 0.0
                if dwell and dwell["card"] == i:
                    progress = min(1.0, (self._now - dwell["since"]) /
                                   self.params.get("dwell_s", 1.5))
                cards.append({"rect": c["rect"], "icon": c["icon"], "state": c["state"],
                              "key": str(i + 1), "dwell": progress})
        return {
            "kind": "cards",
            "cards": cards,
            "pointer": pointer,
            "level": self.level,
            "pairs": self.pairs,
            "found": self.found,
            "prompt": "Point at a card and hold still",
            "finger_colors": {"index": "target"} if pointer is not None else {},
            "demo_key": "point",
        }
