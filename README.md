# Hand rehabilitation coach for Eleanor

A webcam coach that guides a stroke survivor through hand exercises at home. It watches the hand
with MediaPipe, **measures** each movement against her own calibrated range, counts and checks
every repetition, and talks her through the session in a slow, calm voice. A small garden grows
each time she shows up.

It was built for the *Socially Assistive Robots for Sport and Rehabilitation Coaching* assignment
(see `Socially Assistive Robots for Sport and Rehabilitation Coaching.md` and
`details_personas.pdf`). The system follows the **Sense → Think → Act** pattern and is designed
around one persona:

> **Eleanor**, 72, retired librarian. Ischemic stroke with weakness on her **left** side, and mild
> cognitive impairment (slower processing, occasional memory lapses). She wants to get back to
> cooking and gardening and to stay independent.

Everything in the app follows from that: it trains the left hand, speaks slowly and briefly, never
needs a keyboard, remembers what she did last time, and never makes a missed day feel like a
failure.

---

## Contents

- [What the app does](#what-the-app-does)
- [Quick start](#quick-start)
- [Controls](#controls)
- [A session, step by step](#a-session-step-by-step)
- [The exercises](#the-exercises)
- [How the hand is measured](#how-the-hand-is-measured)
- [Motivation features](#motivation-features)
- [Speech and screens](#speech-and-screens)
- [Saved data](#saved-data)
- [Changing settings and wording](#changing-settings-and-wording)
- [Code structure](#code-structure)
- [Tests and tools](#tests-and-tools)
- [Before the first real session](#before-the-first-real-session)
- [Limits](#limits)
- [What has been built so far](#what-has-been-built-so-far)

---

## What the app does

- **Six hand exercises** from stroke rehabilitation: grip and release, finger spreading, thumb
  bending, thumb-to-fingertip touches, finger tapping and a cloth squeeze. **Three more**: a
  bubble pinch (buttons, a pinch of salt), both hands together (the stronger hand leads) and a
  memory card game played by pointing. She can do one or all of today's exercises from a menu.
- **Shows, says and writes every instruction**: a demo hand loops through the movement before
  each exercise, and a small picture shows the position to reach during it. `R` repeats what
  was said.
- **Stop is always one key away**: `S` stops everything and shows the stroke warning signs and
  112. The app never decides it is an emergency and never calls anyone.
- **Measures, doesn't guess.** Each exercise is a continuous measure (an angle or a distance from
  the 21 hand landmarks) scaled to *her* calibrated range, not a healthy hand's. Reps are counted
  with hysteresis thresholds and hold timers, so jitter doesn't produce fake reps.
- **Checks quality**: whether the right hand is visible, close enough and facing the camera; hold
  steadiness, smoothness, and compensation (wrist moving, palm rotating, other fingers moving).
- **Talks to her** with slow speech, subtitles and a soft chime per good rep. Instructions always
  come before praise. Out-of-date messages are skipped instead of spoken late.
- **Answers without a keyboard**: thumbs up = yes, thumbs down = no.
- **Adapts**: targets rise and fall with her success, a "difficult day" mode makes the session
  easier when she's tired or has a bad day, and she is offered a rest when her range drops.
- **Motivates**: personal bests, praise for things that were really measured, links to daily
  activities she cares about ("This practises holding a cup of tea."), milestones, and a garden
  that grows from showing up and never wilts.
- **Remembers**: a coach with a name she chose, greetings based on when she last practised, and a
  history she (or her therapist) can look at in `data/`.
- **Asks how it went**: at the end of a session she rates how hard it was and how much she
  enjoyed it (1–5, a key or that many fingers held up). `tools/report.py` turns all of this into
  tables and charts for the report.
- **Guests keep their own data**: `--guest` (e.g. at the marketplace) uses a separate folder and
  skips the first-time questions, so her own data is never touched.
- **Works offline and on any OS**: macOS, Windows and Linux; no internet or account needed.

---

## Quick start

You need Python 3 with a [MediaPipe](https://pypi.org/project/mediapipe/) wheel for your
platform, and a webcam. A virtual environment is recommended.

```bash
python -m venv venv
source venv/bin/activate            # Windows PowerShell: .\venv\Scripts\activate
pip install -r requirements.txt

python main.py                      # full session: greeting, check-in, menu
```

Other ways to run it:

```bash
python main.py --camera 0                   # choose the webcam (default: CAMERA_INDEX in rehab/config.py)
python main.py --exercise grip_release      # only this exercise, no menu (can be given more than once)
python main.py --video recording.mp4        # run on a recording instead of the webcam
python main.py --video rec.mp4 --no-mirror  # a recording that is already mirrored
python main.py --no-speech                  # print what the coach says instead of speaking
python main.py --windowed                   # start in a window instead of full screen
python main.py --guest --short              # a visitor: own data folder, short sets
python main.py --guest --hand Right         # a visitor who trains the right hand
python -m tools.report                      # tables and charts from the saved data
python -m pytest tests                      # tests (synthetic hands, no camera needed)
```

> **Camera index:** `CAMERA_INDEX` in `rehab/config.py` is set to `1` for one of the team's
> laptops. If you see no image, run with `--camera 0` or change the setting.

Exercise names for `--exercise`: `grip_release`, `finger_abduction`, `thumb_flexion`,
`thumb_opposition`, `finger_tapping`, `grip_squeeze`, `bubble_pinch`, `two_hand_match`,
`memory_pairs`.

The MediaPipe model the app uses is included: `models/gesture_recognizer.task`.

---

## Controls

**Without a keyboard:** thumbs up = yes / good / carry on, thumbs down = no / not so good. Either
hand counts; hold the gesture for a moment. Ratings: hold up 1 to 5 fingers. Memory pairs: point
at a card and hold still.

| Key | What it does |
|---|---|
| space | yes / start / continue / pause; skips a rest |
| `y` / `n` | yes / no |
| `1`–`9` | in the menu: pick one exercise (two columns) |
| `0` or space | in the menu: all of today's exercises |
| `e` | in the menu: finish for today |
| `p` | in the menu: my profile (what the coach remembers) |
| `s` | **Stop / "I don't feel well"**: the safety screen, from anywhere |
| `r` | repeat what the coach said for this screen |
| `1`–`5` | answer a rating question (space skips) |
| `1`–`8` | memory pairs: turn that card over |
| `d`, then `y` | on the profile screen: delete the profile and start again from the first questions |
| `f` | full screen on / off |
| ↑ / ↓ | in the menu: move the highlight |
| `m` | back to the menu at any time |
| `q` or Esc | stop (everything done so far is kept) |

---

## A session, step by step

```
(first time only: she chooses the coach's name and three favourite activities)
greeting → check-in → menu or today's plan → [activity card → calibration → sets with rests] × n
→ summary → ratings (1–5) → garden → goodbye
```

1. **First time:** she picks the coach's name ("Iris" or "Robin") and keeps up to three of six
   daily activities (gardening, making tea, cooking, reading, getting dressed, using the phone).
2. **Greeting:** chosen by the days since her last session: first ever, same day, yesterday,
   2–6 days, 7+ days, or after a difficult day. It mentions at most one remembered fact, only from
   what is stored in `data/`. A gap is never called a failure; after 7+ days the coach says
   "let's start gently" and targets start two steps lower.
3. **Check-in:** "How is your hand feeling today?" A thumbs down switches on difficult day mode.
   No answer within 25 s means a normal day.
4. **Menu:** one exercise, all of today's exercises, "Finish for today", or "My profile". With `--exercise`
   there is no menu, only "Today we'll do one exercise." Today's plan (`DAILY_PLAN`) is the six
   hand exercises and memory pairs as a restful end; bubble pinch and two-hand match are in the
   menu only. Grip squeeze (strengthening) is planned only every other day.
5. **Each exercise:**
   - a full-screen **activity card** that links it to one of her activities, favourites first,
     with a **demo hand** looping through the movement and "Exercise 2 of 7" in a plan;
   - **calibration** the first time, or when it is older than 14 days; otherwise she can press
     space within 6 s to recalibrate. A clearly wrong calibration (e.g. "open" less open than
     "closed") is measured again, also when it came from an earlier session;
   - **sets with rests** (30 s between sets, 45 s between exercises). The target is adapted after
     every set. When her range drops over several reps, the coach offers a rest. A small still
     demo shows the position to reach now. Memory pairs needs no measuring and starts at once.
6. **Summary:** one highlight from today, plus one comparison with her own history
   ("Your hand opened 12% wider than last week").
7. **Ratings:** "How hard was today's practice?" and "How much did you enjoy it?" (guests also
   "How easy was the coach to use?"), 1–5 with a key or fingers held up, 25 s to answer, never
   required. Saved in `sessions.csv`.
8. **Garden:** a new seed (she picks one of two plants) or the next growth stage, with a short
   animation.
9. **Goodbye:** the same closing line every time, plus "Next time, we'll keep your rose growing."

---

## The exercises

| # | Exercise | Measure | One rep |
|---|---|---|---|
| 1 | `grip_release` — open and close your hand | average finger openness (MCP+PIP+DIP flexion), per finger scaled to her range | open, hold → close, hold |
| 2 | `finger_abduction` — spread your fingers | sum of the 3 gaps between fingers, angles in the palm plane | spread, hold → together, hold (pauses if fingers bend) |
| 3 | `thumb_flexion` — bend and stretch your thumb | thumb MCP+IP flexion combined with thumb tip → pinky MCP distance | in, hold → out, hold |
| 4 | `thumb_opposition` — touch your fingertips | thumb-to-fingertip distances / palm size | a sequence of touches: guided (level 1), then from memory (levels 2+) |
| 5 | `finger_tapping` — lift one finger at a time | fingertip height above the calibrated flat hand + MCP angle | a sequence of lifts: in order, called out, or a remembered pattern |
| 6 | `grip_squeeze` — squeeze a rolled cloth | finger closure between "holding" and "squeezing" the cloth | squeeze, hold 4 s → relax 4 s (every other day) |
| 7 | `bubble_pinch` — pinch the bubble | thumb tip → index tip distance / palm size, scaled between her "open" and "pinch" | pinch, hold 2 s (a bubble shrinks and pops) → let go |
| 8 | `two_hand_match` — both hands together | the affected hand's openness drives the reps; the other hand's openness gives the **symmetry** (1 − mean gap) | open both, hold → close both, hold |
| 9 | `memory_pairs` — memory pairs | pointing: index fingertip over a face-down card, held still 1.5 s (or keys 1–8) | one finished board of 2, 3 or 4 pairs |

Default sets and reps (from the rehabilitation video, low end of stage 4): 3 × 10 for exercises
1–3, 2 × 3 rounds for thumb opposition, 2 × 2 rounds for finger tapping, 2 × 8 for grip squeeze.
All are in `rehab/config.py`.

The two sequence exercises also train memory: thumb opposition moves up a level after two
error-free rounds in a row, and finger tapping logs reaction time and an **isolation score** (how
still the other fingers stayed). In thumb opposition every correct touch plays the next note of
"Twinkle, Twinkle, Little Star" (the **finger piano**); a wrong touch plays nothing.

Memory pairs moves up a pair after two good boards (no more than one turn too many). A pair
that does not match stays up for 2 s and turns back ("Try to remember where they are."). It logs
turns per pair, time per choice and how many cards were chosen with the affected hand. The cards
lie in the middle and right of the image, never at the left edge.

Two-hand match needs both hands in view ("Please show both hands to the camera."). Bubble pinch
and two-hand match default to 2 × 8.

**Logged for every rep:** range reached (% of calibration), the value held, movement time,
smoothness (speed peaks), hold stability, compensation (palm rotation, wrist movement, other
fingers moving), hints given, success, and exercise-specific details (lagging finger, isolation
score, correct/wrong touches, level).

---

## How the hand is measured

```
camera frame ─► MediaPipe GestureRecognizer ─► 21 landmarks (image + 3D world), handedness, gesture
            ─► features.py: angles, openness, spread, distances, palm size, palm facing, quality flags
            ─► One Euro filter (smooths jitter without lag)
            ─► exercise: scale to her calibrated range ─► hysteresis zones + hold timers ─► reps
```

- **Sense** (`rehab/Sense.py`): webcam or video file. The recognizer sees the camera's own,
  unmirrored image (so "Left" means her real left hand) and the landmarks are mirrored afterwards
  for a mirror-like display. The gesture label is used only for thumbs up / down answers and, in
  grip and release, as a logged secondary check.
- **Features** (`rehab/features.py`): joint angles from the 3D world landmarks, per-finger
  openness, finger spread in the palm plane, thumb distances divided by palm size, and quality
  flags: no hand, wrong hand, too far away, palm turned away.
- **Calibration** (`rehab/calibration.py`): for each exercise she holds each end position for 3 s
  and the median is stored. All thresholds are fractions of that range (0 = the least she could
  do, 1 = the most). Her calibrated range grows automatically when she holds beyond it.
- **Rep engines** (`rehab/exercises/base.py`): `TwoPhaseExercise` for the four range exercises and
  `SequenceExercise` for thumb opposition and finger tapping. A position counts once it is entered
  and held; it is left only after moving back by a hysteresis gap.
- **Coach** (`rehab/Think.py`): checks tracking quality first and asks her to adjust, calmly and
  not too often (a problem has to last 1.5 s). After 8 s without progress it gives one short hint.
- **Both hands**: `main.py` also extracts and smooths the other hand (its own One Euro filters)
  as `f.other`, for two-hand match and for counting raised fingers (`features.count_extended`).

---

## Motivation features

Think decides what happened and emits events (`rehab/events.py`); Act decides how to say it
(`rehab/feedback.py` with the phrase bank in `content/phrases.json`). Wording can change without
touching logic, and all logic is tested without a camera or speaker.

| Feature | Where | Rules (starting values in `rehab/config.py`, not clinical values) |
|---|---|---|
| Personal bests | `progress.py` | A rep's value is the **median during the hold** (raw measure, comparable across recalibrations), not the peak frame. A best must be ≥3% above the old one. Levels: today / this week / all time; only the highest is announced, at most one per set, after the rep. No announcements in her first session ("I've saved today as your starting point."). Also per exercise: reps without hints and hold steadiness. Speed is deliberately never a best. |
| Adaptive targets | `progress.py` | The target is the "high" threshold, a % of her calibrated range, evaluated once per set: ≥80% successful reps → +3 points, <50% → −3 (silently), floor 50%, ceiling 100%. A rep is successful when she reached and held the target without a hint. The session starts one step below last time (two after 7+ days). When her holds go beyond her calibrated maximum, the calibrated range grows. `AUTO_PROGRESSION = False` freezes targets. |
| Difficult day mode | `progress.py`, `Think.py` | Switched on by a thumbs down at the check-in, a first set 15% below the median of her last 5 normal sessions, the fatigue check, or <50% success in two sets in a row. It then stays on for the session: targets ×0.8, one set fewer, rests ×1.5, no grip squeeze, praise for effort, no comparisons. Said once: "Let's take it easier today." Difficult sessions never change saved targets and are left out of baselines. Three difficult days out of the last five add a note for the therapist in `sessions.csv`. |
| Specific praise | `feedback.py` | Only for events that were really measured ("Your ring finger opened more that time." needs the per-finger data). A soft chime every successful rep; one spoken phrase at most, the most important event, or a short generic one about every 3 reps. The last 3 phrases of a category are never reused. |
| Speech priority | `Act.py` | Instructions can jump ahead of waiting praise and cut off older praise. Praise never delays an instruction. Rep praise waiting over 3 s is dropped instead of played late. |
| Coach character | `content/character.json`, `memory.py` | She picks the name ("Iris" or "Robin") at the first session. Warm, calm and respectful, never childish. A simple face with three expressions (neutral, happy, encouraging). The same opening, check-in and closing lines every session. |
| Daily activities | `content/activities.json` | Six activity cards at the first session, and she keeps up to three. Each exercise shows its link (her favourites first) on a card and as a small icon. Milestones at 50, 100, 250 and 500 reps per activity. Framed as practice toward an activity, never a promise. Please have a therapist check the links. |
| Demo hand | `demo.py`, `ui.py` | The hand model from the tests, drawn from a three-quarter view: a loop on each exercise card, a still picture of the current position during the exercise. One instruction, three ways: text, voice, picture. |
| Finger piano | `sound.py` | Notes made with numpy, written once as small WAV files and played by the system player (`afplay`, `paplay`/`aplay`, `winsound`). pygame is not used: its SDL clashes with OpenCV's copy on macOS. |
| Garden | `garden.py`, `ui.py` | Grows from showing up, not performance: every session with a completed exercise waters it, difficult days too. Seed → sprout → leaves → bud → flower; 8 plots, then "a new season". Plants: rose, tulip, sunflower, lavender. A best adds a bee, a milestone a butterfly. Nothing ever wilts or goes backwards. A watering can shows the exercises done today. |

---

## Speech and screens

**Speech** (`rehab/Act.py`, `rehab/tts_util.py`) runs in its own thread so the camera never
freezes:

- macOS: the built-in `say -r 145`;
- Windows: `pyttsx3` (SAPI5 voices);
- Linux: `pyttsx3` (eSpeak), or the `espeak-ng` / `espeak` command when pyttsx3 is not available.

Voice and rate are per user in `data/profile.json` (default 145 words per minute). Because speech
is slow, every message is checked again just before it is spoken and skipped when she has already
done what it asks; one that goes out of date while it is being spoken is cut off by the next
instruction. When she moves on (answers, presses space, the next step starts, pause) whatever
the coach was still saying for the step she left is dropped and cut off, so she only hears what
belongs to the screen in front of her. Calibration only starts timing a position after its prompt
has been spoken. Everything that is said is also shown as a subtitle, and cards show the same
words she hears.

**Screens** (`rehab/Act.py`, `rehab/ui.py`):

- the mirrored camera image with the hand skeleton, and a side panel with a large progress bar
  and target line, per-finger bars, the finger sequence and big high-contrast text;
- full-screen cards for questions, activities, the summary and the garden, with the coach's face;
- a menu with large numbered items.
- **Stop / "I don't feel well"** (`s`, shown bottom right on every screen, never at the left
  edge): "Let's stop here. Please sit down and rest.", the stroke warning signs, **112** in large
  type ("Call 112 now, even if it passes"), and optionally a helper to call (`HELPER_NAME`,
  `HELPER_PHONE` in `config.py`). It stays until she says she feels fine (thumbs up or space)
  and is logged in `sessions.csv` for her therapist. The app never calls anyone itself.
- **ratings**: five large numbered boxes with a word under each, in neutral colours; the box
  for the fingers she holds up fills while she holds.
- **her profile** (`p` in the menu): everything the coach remembers (her name, trained hand,
  the coach's name and favourite activities she chose, when she started, sessions, measured
  exercises, personal bests, practice per activity, the garden). `d` asks whether to delete it;
  only the `y` key confirms (thumbs down, `n` or waiting keeps it). After deleting, the app starts
  again from the first-time questions.

The window opens **full screen** (`f` switches, `--windowed` starts in a window). Every screen is
drawn in the shape of the screen and the window scales it to fit, so nothing falls off the edge on
any screen size: the camera image gets dark space around it rather than being cut, and the
spoken subtitle shrinks to fit. Cards, the summary and the profile show her **camera image**
small, so she can check that her hand is in view while answering: the border turns green and a
bar fills while a thumbs up is held, and a line underneath says what the coach sees ("I can see
your hand", "Thumbs up - hold it there", "Now lower your hand", ...).

Text uses Pillow with Atkinson Hyperlegible (`assets/fonts`, SIL Open Font License), a font made
for low vision. Each line is rendered once and reused, so drawing a frame stays at about 2 ms.
The webcam is read in its own thread that keeps only the newest frame, so a slow frame never
leaves the screen behind her hand. Icons and garden pictures are drawn in calm, flat colours. PNGs (for example
exported from Figma) are used instead when present: `assets/icons/<icon>.png`,
`assets/garden/<plant>_<stage>.png` (stage 0–4), `assets/garden/{bee,butterfly,can,bed}.png`.

---

## Saved data

Personal data lives in `data/` and is **not committed** (see `.gitignore`).

| File | Contents |
|---|---|
| `profile.json` | name, affected hand, coach name, favourite activities, voice and rate, calibrations, targets, sequence levels, personal bests, reps practised per activity, last session |
| `reps.csv` | one row per rep with all quality measures (written after every rep) |
| `history.csv` | one row per exercise per session: means, bests, success rate, targets |
| `sessions.csv` | one row per session: duration, reps, check-in, difficult day, highlight, garden, therapist note, whether Stop was pressed, and the ratings (exertion, enjoyment, ease) |
| `garden.json` | the garden's plants and growth |
| `tracking_check.csv`, `validation.csv` | results of the tools below |

`profile.json`, `garden.json` and `sessions.csv` are written to a temporary file first and then
renamed, so a crash never leaves half a file. An unreadable file is set aside
(`*.broken-<time>`), and the coach starts as on a first day rather than remembering something
wrong. Older data files are upgraded automatically (new CSV columns, `thresholds` → `targets`).

**Guests** (`--guest`) get their own folder, `data/guests/<time>/`, with the same files. A
guest profile starts with the first coach name and three activities chosen, so there are no
first-time questions. `python -m tools.report` reads `data/` and every guest folder.

**Deleting the profile** (profile screen, `d` then `y`) removes `profile.json`, `garden.json`,
`reps.csv`, `history.csv` and `sessions.csv` together. By default they are moved to
`data/deleted/<time>/`, so a therapist can still put them back; set `KEEP_DELETED_PROFILE = False`
in `rehab/config.py` to erase them for good.

---

## Changing settings and wording

- **`rehab/config.py`**: everything a therapist might want to change: the trained hand, camera,
  sets, reps, thresholds, hold times, rest times, target steps, difficult-day rules, milestones
  and garden size; the menu (`SESSION_ORDER`) and today's plan (`DAILY_PLAN`); the short
  session for guests (`SHORT_SESSION`); the rating questions; the emergency number and helper. `AUTO_PROGRESSION = False` stops the app from changing targets by itself.
  `FULLSCREEN` and `DESIGN_HEIGHT` set how the window opens; `KEEP_DELETED_PROFILE` whether a
  deleted profile is kept as a backup.
- **`content/phrases.json`**: every sentence the coach says, grouped by event.
- **`content/character.json`**: the coach's name options, personality rules and face colours.
  Set `"name"` to fix the name instead of letting her choose.
- **`content/activities.json`**: the daily activities, their icons, and which exercise practises
  which activity.

---

## Code structure

```
main.py                    wiring: Sense → features → Think → Act (nothing imports it)
rehab/config.py            all settings and per-exercise parameters
rehab/Sense.py             camera / video, MediaPipe GestureRecognizer → HandObservation
rehab/features.py          landmarks → HandFeatures (angles, openness, spread, distances, quality flags,
                           the other hand, raised-finger count)
rehab/filters.py           One Euro filter
rehab/calibration.py       per-exercise capture of her range (median over a 3 s hold)
rehab/exercises/base.py    Exercise base, hysteresis, two-phase and sequence engines, rep quality measures
rehab/exercises/*.py       the nine exercises
rehab/Think.py             Coach (quality checks, logging, rep events) + SessionManager (session flow)
rehab/events.py            events with priorities: what Think tells Act
rehab/progress.py          personal bests, adaptive targets, difficult days, milestones
rehab/memory.py            greetings and what the coach remembers
rehab/garden.py            the garden's state (grows, never wilts)
rehab/Act.py               speech with a priority queue + drawing (skeleton, bar, sequence, subtitles)
rehab/tts_util.py          text-to-speech backend by OS
rehab/feedback.py          events → words from content/phrases.json (no repeats, praise frequency)
rehab/ui.py                Pillow text, full-screen cards, summary, garden, ratings, safety screen,
                           demo hand, stop hint, watering can, coach face
rehab/demo.py              the demo hand's positions and loops per exercise
rehab/handmodel.py         a kinematic hand (joint angles → 21 landmarks), for the demo and the tests
rehab/sound.py             finger piano notes (WAV files, played by the system player)
rehab/storage.py           profile.json, reps.csv, history.csv, sessions.csv, garden.json
content/                   character.json, phrases.json, activities.json (edit wording here)
assets/                    fonts (Atkinson Hyperlegible), optional icons and garden PNGs
models/                    MediaPipe models (the app uses gesture_recognizer.task)
tools/tracking_check.py    detection rate and jitter of each measure with your camera
tools/validate.py          program rep count vs. a count by hand, on recorded videos
tools/report.py            tables, charts and report.md from all saved data (including guests)
tests/                     unit and session tests driven by a synthetic 3D hand
```

---

## Tests and tools

```bash
pip install pytest
python -m pytest tests
```

The 163 tests need no camera or speaker. `rehab/handmodel.py` builds a 3D hand in any pose,
so exercises, calibration, whole sessions, speech priority, the motivation rules, storage and the
text-to-speech backend choice are all tested with made-up hands.

- `python -m tools.tracking_check --seconds 30` shows how often the hand is detected and how much
  each measure jitters with your camera and light. `--record file.mp4` also saves the video,
  `--video` analyses a recording.
- `python -m tools.validate video.mp4 --exercise grip_release --expected 10 --calibrate 12`
  compares the program's rep count with a count by hand. Results are appended to
  `data/validation.csv` (`--note` for conditions, `-v` to print what the coach would say).
- `python -m tools.report` writes `data/report/`: `summary.csv` (per exercise, Eleanor and guests
  apart: sessions, reps, success rate, range, hints and compensation per rep, movement time,
  smoothness, symmetry, turns), `sessions.csv` with the ratings, `report.md` with plain tables,
  and PNG charts of progress and ratings.

---

## Before the first real session

1. `python -m tools.tracking_check --seconds 30`. Hold up the **left** hand, palm to the camera.
   The overlay should say `hand OK`. If it complains about the wrong hand, set
   `RECOGNIZE_UNMIRRORED = False` in `rehab/config.py`. (On MediaPipe's sample photos, Tasks labels
   the real hand in the unmirrored view, so the recognizer gets the unmirrored frame by default.)
2. Record finger tapping and thumb opposition with `--record` and check the jitter table. Finger
   tapping depends heavily on the camera angle: a tilted phone stand or a camera looking down helps.
3. For the report: record test videos, count reps by hand, and run `tools.validate` on them.
4. Have a therapist check the exercises, sets and reps, and the activity links.

---

## Limits

- One webcam gives estimated depth. The angles are good for relative progress over time, not
  clinical goniometer measurements.
- Grip squeeze measures holding and timing, not force.
- Finger tapping depends heavily on the camera angle.
- The Brunnstrom stage and the default sets and reps come from the rehabilitation video and should
  be confirmed by a therapist before real use.
- All motivation numbers (best threshold, target step, success rates, difficult-day drop) are
  starting values for testing, not clinical values.
- The app gives no medical advice. Repeated difficult days only leave a note for her therapist
  or family.
- Adapting hold time instead of range is not needed: every range exercise measures range. The two
  sequence exercises keep their own levels.

---

## What has been built so far

The project started from the course's elbow-flexion template (Sense / Think / Act with a balloon
animation) and was rebuilt step by step into the hand coach:

1. **Setup:** MediaPipe Tasks models, mirrored camera view, a first hand gesture recogniser.
2. **Measuring the hand** ([PR #1](https://github.com/ryanvafaei/stroke-rehab/pull/1)): gesture classification replaced by continuous measures,
   per-user calibration, One Euro smoothing, hysteresis rep counting, quality checks, the six
   exercises, rep logging, the tracking and validation tools, and the synthetic-hand tests.
3. **Choosing an exercise** ([PR #2](https://github.com/ryanvafaei/stroke-rehab/pull/2)): a start menu to pick one exercise, all of today's, or finish.
4. **Voice that matches the moment** ([PR #3](https://github.com/ryanvafaei/stroke-rehab/pull/3)): messages are re-checked before speaking and skipped
   when out of date; calibration waits for its prompt; wrong calibrations are redone.
5. **Speech on every OS** ([PR #4](https://github.com/ryanvafaei/stroke-rehab/pull/4)): `say` on macOS, pyttsx3 on Windows and Linux, espeak fallback.
6. **Motivation** ([PR #5](https://github.com/ryanvafaei/stroke-rehab/pull/5)): personal bests, adaptive targets, difficult day mode, specific praise,
   a speech priority queue, a coach character, daily activity cards and milestones, the garden,
   thumbs up / down answers, and full-screen cards in Atkinson Hyperlegible.
7. **Full screen and profile** ([PR #8](https://github.com/ryanvafaei/stroke-rehab/pull/8)): opens full screen, the camera on question cards, her profile screen.
8. **Testing and persona fit**: ratings at the end of a session, guest mode and `tools/report.py`,
   a demo hand for every exercise, Stop / "I don't feel well" with 112, "Exercise 2 of 7" and a
   repeat key, the finger piano, and three new exercises: bubble pinch, two-hand match and
   memory pairs.
