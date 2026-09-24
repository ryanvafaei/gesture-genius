"""
Guests (e.g. visitors at the marketplace): every guest is a new user with
a unique id, e.g. "guest-007-20260925-143012" (a running number that is easy
to write on a questionnaire, and the time the guest was created, which
keeps the id unique). Everything of that guest goes in one folder named
after the id:

    data/guests/<guest id>/                     profile, reps, history, sessions ...
    data/guests/<guest id>/<guest id>_report/   the guest's report (tools/report.py)
    data/guests/<guest id>/verbose/             detailed logs (python main.py --verbose)

data/guests/guests.csv lists every guest (id, when, hand) for later analysis.
Her own data (data/) is never touched.
"""

import csv
import re
from datetime import datetime
from pathlib import Path

from rehab import config, storage

PREFIX = "guest"
REGISTRY = "guests.csv"
_NUMBER = re.compile(rf"^{PREFIX}-(\d+)-")
MAIN_USER = "eleanor"           # her own data folder in the report and the logs


def guests_dir():
    return Path(config.GUESTS_DIR)


def new_guest_id(folder=None, now=None):
    """The next free id: guest-<running number>-<date>-<time>."""
    folder = Path(folder or guests_dir())
    numbers = [int(m.group(1)) for p in (folder.iterdir() if folder.is_dir() else [])
               if (m := _NUMBER.match(p.name))]
    n = max(numbers, default=0) + 1
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    guest_id = f"{PREFIX}-{n:03d}-{stamp}"
    while (folder / guest_id).exists():     # two guests in the same second
        n += 1
        guest_id = f"{PREFIX}-{n:03d}-{stamp}"
    return guest_id


def report_dir(folder, user):
    """Where one user's report goes: <folder>/<user>_report."""
    return Path(folder) / f"{user}_report"


def start_guest(hand=None, folder=None):
    """
    A new guest: a new id, its own data folder with a ready profile (no
    first-time questions: the first coach name and three activities are
    chosen), switched to with config.use_data_dir. Returns (id, folder).
    """
    root = Path(folder or guests_dir())
    guest_id = new_guest_id(root)
    path = root / guest_id
    path.mkdir(parents=True)
    config.use_data_dir(path)
    character = storage.load_content("character")
    activities = storage.load_content("activities")
    profile = storage.new_profile()
    profile["guest_id"] = guest_id
    profile["coach_name"] = (character.get("name") or (character.get("name_options") or [None])[0])
    profile["chosen_activities"] = list(activities.get("activities", {}))[:config.ACTIVITY_CHOICES]
    if hand:
        profile["affected_hand"] = hand
    storage.save_profile(profile)
    _register(root, guest_id, profile.get("affected_hand", config.AFFECTED_HAND))
    return guest_id, path


def _register(root, guest_id, hand):
    path = Path(root) / REGISTRY
    new = not path.is_file()
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["guest_id", "created", "hand", "folder"])
        w.writerow([guest_id, datetime.now().isoformat(timespec="seconds"), hand, guest_id])


def user_of(folder_name):
    """The user name of a guest folder (older folders were named by the time only)."""
    return folder_name if folder_name.startswith(PREFIX) else f"{PREFIX}-{folder_name}"
