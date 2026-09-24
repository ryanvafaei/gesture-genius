"""
Numbers and charts for the report, from the saved CSV files.

Reads data/ (Eleanor, or whoever used the app normally) and every guest
folder in data/guests/ (python main.py --guest, e.g. at the marketplace):

    python -m tools.report                      # writes data/report/
    python -m tools.report --data data --out data/report

Writes
  summary.csv    per exercise and user group: sessions, reps, success rate,
                 range reached, hints and compensation per rep, movement
                 time, smoothness, and symmetry / turns where they apply
  sessions.csv   one row per session of everyone, with the 1-5 ratings
                 (exertion, enjoyment, ease) and whether Stop was pressed
  report.md      the same as plain tables, ready to paste into the report
  *.png          range and success over sessions per exercise, and the ratings

Objective measures come from the rep and exercise logs; subjective ones
from the questions at the end of each session. Practice data, not a
clinical assessment.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from rehab import config

RATINGS = ("exertion", "enjoyment", "ease")


def _read(path):
    try:
        return pd.read_csv(path) if Path(path).is_file() and Path(path).stat().st_size else None
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def load(data_dir):
    """All CSVs of data_dir and its guest folders, with a "user" column."""
    data_dir = Path(data_dir)
    folders = [("eleanor", data_dir)]
    guests = data_dir / "guests"
    if guests.is_dir():
        folders += [(f"guest-{p.name}", p) for p in sorted(guests.iterdir()) if p.is_dir()]
    tables = {}
    for name in ("sessions", "history", "reps"):
        parts = []
        for user, folder in folders:
            df = _read(folder / f"{name}.csv")
            if df is not None and len(df):
                df.insert(0, "user", user)
                df.insert(1, "group", "guests" if user.startswith("guest") else "eleanor")
                parts.append(df)
        tables[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return tables


def _extra_mean(rows, key):
    values = []
    for text in rows.get("extra", pd.Series(dtype=str)).dropna():
        try:
            v = json.loads(text).get(key)
        except (ValueError, AttributeError):
            continue
        if isinstance(v, (int, float)):
            values.append(float(v))
    return round(sum(values) / len(values), 3) if values else ""


def summary(history):
    """One row per group and exercise."""
    rows = []
    if history.empty:
        return pd.DataFrame()
    for (group, exercise), h in history.groupby(["group", "exercise"]):
        reps = h["reps_done"].sum()
        if not reps:
            continue
        rows.append({
            "group": group, "exercise": exercise,
            "sessions": h["session_id"].nunique(),
            "reps": int(reps),
            "success_rate": round((h["success_rate"] * h["reps_done"]).sum() / reps, 3),
            "range_reached": round(h["mean_range_high"].mean(), 3),
            "hints_per_rep": round(h["hints"].sum() / reps, 2),
            "compensation_share": round(h["compensation_reps"].sum() / reps, 3),
            "movement_time_s": round(h["mean_movement_time_s"].mean(), 2),
            "smoothness_peaks": round(h["mean_smoothness_peaks"].mean(), 2),
            "symmetry": _extra_mean(h, "symmetry"),
            "turns_per_board": _extra_mean(h, "turns"),
        })
    return pd.DataFrame(rows)


def ratings_summary(sessions):
    rows = []
    if sessions.empty:
        return pd.DataFrame()
    for group, s in sessions.groupby("group"):
        for key in RATINGS:
            if key not in s:
                continue
            v = pd.to_numeric(s[key], errors="coerce").dropna()
            if len(v):
                rows.append({"group": group, "question": key, "answers": len(v),
                             "mean": round(v.mean(), 2), "median": v.median(),
                             "min": int(v.min()), "max": int(v.max())})
    return pd.DataFrame(rows)


def markdown(df):
    """A plain Markdown table (no extra packages needed)."""
    if df is None or df.empty:
        return "_No data yet._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def charts(history, sessions, out):
    """PNG charts; skipped quietly when matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made = []
    mine = history[history["group"] == "eleanor"] if not history.empty else history
    for exercise, h in (mine.groupby("exercise") if not mine.empty else []):
        h = h.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(range(1, len(h) + 1), h["mean_range_high"], marker="o", label="range reached")
        ax.plot(range(1, len(h) + 1), h["success_rate"], marker="s", label="success rate")
        ax.set_xlabel("session")
        ax.set_ylim(0, 1.3)
        ax.set_title(exercise.replace("_", " "))
        ax.legend(loc="lower right")
        fig.tight_layout()
        path = Path(out) / f"progress_{exercise}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        made.append(path)
    present = [k for k in RATINGS if not sessions.empty and k in sessions]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(3.2 * len(present), 3), squeeze=False)
        for ax, key in zip(axes[0], present):
            for group, s in sessions.groupby("group"):
                v = pd.to_numeric(s[key], errors="coerce").dropna().astype(int)
                counts = [int((v == i).sum()) for i in range(1, 6)]
                offset = -0.2 if group == "eleanor" else 0.2
                ax.bar([i + offset for i in range(1, 6)], counts, width=0.4, label=group)
            ax.set_xticks(range(1, 6))
            ax.set_title(key)
        axes[0][0].legend()
        fig.tight_layout()
        path = Path(out) / "ratings.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        made.append(path)
    return made


def build(data_dir, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t = load(data_dir)
    summ = summary(t["history"])
    rate = ratings_summary(t["sessions"])
    summ.to_csv(out / "summary.csv", index=False)
    t["sessions"].to_csv(out / "sessions.csv", index=False)
    pictures = charts(t["history"], t["sessions"], out)
    s = t["sessions"]
    lines = [
        "# Coach data for the report\n",
        f"Sessions: {len(s)} ({(s['group'] == 'eleanor').sum() if len(s) else 0} normal, "
        f"{(s['group'] == 'guests').sum() if len(s) else 0} guest). "
        f"Stop pressed in {(s.get('safety_stop', pd.Series(dtype=str)) == 'yes').sum()} sessions.\n",
        "## Per exercise (objective)\n", markdown(summ),
        "## Ratings at the end of a session (1-5, subjective)\n",
        "exertion: 1 very easy - 5 very hard; enjoyment: 1 not at all - 5 very much; "
        "ease: 1 very hard - 5 very easy.\n", markdown(rate),
    ]
    if pictures:
        lines += ["## Charts\n"] + [f"![{p.stem}]({p.name})\n" for p in pictures]
    (out / "report.md").write_text("\n".join(lines))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(config.DATA_DIR))
    p.add_argument("--out", default=None, help="default: <data>/report")
    args = p.parse_args()
    out = build(args.data, args.out or Path(args.data) / "report")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
