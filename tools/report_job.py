"""
The report (tools/report.py) made from inside the app: the toolbar's
"Make report" button.

It runs `python -m tools.report` in its own process, so the camera and the
coach keep running while pandas and matplotlib work, and opens the report
folder when it is ready. Kept apart from tools/report.py so that the app
does not import pandas at start-up.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from rehab import config


def open_folder(path):
    """Show a folder in Finder / Explorer / the file manager. Failures are ignored."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform.startswith("win"):
            os.startfile(str(path))                     # noqa: S606 (Windows only)
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except (OSError, AttributeError):
        pass


def _shown(path):
    """The path as she would read it: relative to the project when it is inside it."""
    try:
        return str(Path(path).resolve().relative_to(config.ROOT_DIR))
    except ValueError:
        return str(path)


class ReportJob:
    """
    One run of tools.report in the background.

    status() -> (state, text): state is "running", "done" or "failed";
    text is a short line for the toolbar.
    """

    def __init__(self, data_dir=None, out=None, open_when_done=True, command=None, user=None):
        """user: one user's report (e.g. a guest's id: only data_dir, labelled with it)."""
        self.data_dir = Path(data_dir or config.BASE_DATA_DIR)
        self.out = Path(out or self.data_dir / "report")
        self.user = user or "everyone"
        self.open_when_done = open_when_done
        self._errors = tempfile.TemporaryFile()
        self._state = None
        cmd = command or ([sys.executable, "-m", "tools.report",
                           "--data", str(self.data_dir), "--out", str(self.out)]
                          + (["--user", user] if user else []))
        try:
            self._proc = subprocess.Popen(cmd, cwd=str(config.ROOT_DIR),
                                          stdout=subprocess.DEVNULL, stderr=self._errors)
        except OSError as e:
            self._proc = None
            self._state = ("failed", f"Report failed: {e}")

    def wait(self, timeout=120):
        """Block until the report is made (e.g. when the app closes). Returns status()."""
        if self._proc is not None and self._state is None:
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        return self.status()

    def status(self):
        if self._state is not None:
            return self._state
        code = self._proc.poll()
        if code is None:
            return "running", "Making the report..."
        if code == 0:
            self._state = ("done", f"Saved to {_shown(self.out)}")
            if self.open_when_done:
                open_folder(self.out)
        else:
            self._errors.seek(0)
            lines = self._errors.read().decode(errors="replace").strip().splitlines()
            print("tools.report failed:\n" + "\n".join(lines[-20:]), file=sys.stderr)
            self._state = ("failed", "Report failed (see the terminal)")
        self._errors.close()
        return self._state


def start_background(data_dir=None, out=None, open_when_done=True, user=None):
    """
    Start making the report of data_dir (default: her data folder with all
    guests), or with user only that folder, labelled with that user.
    """
    return ReportJob(data_dir, out, open_when_done, user=user)
