# Hand and arm rehabilitation coach for Eleanor

A webcam coach that guides a stroke survivor through hand and arm exercises at home. It watches
the hand and body with MediaPipe, **measures** each movement against her own calibrated range and
against published clinical benchmarks, counts and checks every repetition, and talks her through
the session in a slow, calm voice. A demo hand shows each movement, a memory card game trains her
memory, and a small garden grows each time she shows up. At the end she rates the session, and a
report tool turns everything into numbers for the report.

It was built for the *Socially Assistive Robots for Sport and Rehabilitation Coaching* assignment
(see `Socially Assistive Robots for Sport and Rehabilitation Coaching.md` and
`details_personas.pdf`). The system follows the **Sense → Think → Act** pattern and is designed
around one persona:

> **Eleanor**, 72, retired librarian. Ischemic stroke with weakness on her **left** side, and mild
> cognitive impairment (slower processing, occasional memory lapses). She wants to get back to
> cooking and gardening and to stay independent.

Everything in the app follows from that: it trains the left hand and arm, speaks slowly and
briefly, shows every movement as well as saying it, never needs a keyboard, keeps Stop one key
away, keeps important things away from the left edge of the screen, remembers what she did last
time, and never makes a missed day feel like a failure.

---

## Contents

- [What the app does](#what-the-app-does)
- [Quick start](#quick-start)
- [Controls](#controls)
- [A session, step by step](#a-session-step-by-step)
- [The exercises](#the-exercises)
- [How the hand is measured](#how-the-hand-is-measured)
- [Arm exercises and movement benchmarks](#arm-exercises-and-movement-benchmarks)
- [Motivation features](#motivation-features)
- [Speech and screens](#speech-and-screens)
- [Saved data](#saved-data)
- [Changing settings and wording](#changing-settings-and-wording)
- [Code structure](#code-structure)
- [Tests and tools](#tests-and-tools)
- [Before the first real session](#before-the-first-real-session)
- [At the marketplace](#at-the-marketplace)
- [Limits](#limits)
- [What has been built so far](#what-has-been-built-so-far)

---

## What the app does

- **Six hand exercises** from stroke rehabilitation: grip and release, finger spreading, thumb
  bending, thumb-to-fingertip touches, finger tapping and a cloth squeeze. **Three more**: a
  bubble pinch (buttons, a pinch of salt), both hands together (the stronger hand leads) and a
  memory card game played by pointing. She can do one or all of today's exercises from a menu.
- **Eight arm exercises** (arm raises, hand to mouth and head, elbow and wrist, reaching, finger
  to nose), measured in degrees with body tracking and judged against published benchmarks: the
  Fugl-Meyer assessment's form rules, normal and everyday ranges of motion, and the measurement
  error of MediaPipe in stroke survivors. See
  [Arm exercises and movement benchmarks](#arm-exercises-and-movement-benchmarks).
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
- **Every guest is a new user**: `--guest` or "New guest" in the toolbar (e.g. at the
  marketplace) creates a guest with a unique id (`guest-007-20260925-143012`), its own folder and
  its own report, and skips the first-time questions, so her own data is never touched.
- **Toolbar in the menu**: hidden until `i` or the icon at the top right is pressed. Short
  sessions on / off, a new guest, back to her own profile, and the report (with its charts)
  without a terminal.
- **Verbose log for tuning**: `--verbose` logs every frame (landmarks, measures, detector
  state), every event and the camera video; `tools/verbose_summary.py` turns a log into tracking
  and detection numbers with suggested settings.
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
python main.py --exercise shoulder_flexion_raise   # an arm exercise (also the ones for the therapist)
python main.py --video recording.mp4        # run on a recording instead of the webcam
python main.py --video rec.mp4 --no-mirror  # a recording that is already mirrored
python main.py --no-speech                  # print what the coach says instead of speaking
python main.py --windowed                   # start in a window instead of full screen
python main.py --guest --short              # a new guest (unique id, own folder), short sets
python main.py --guest --hand Right         # a new guest who trains the right hand
python main.py --verbose                    # log everything (and the video) for tuning
python main.py --verbose --no-video         # the same without the camera video
python -m tools.report                      # tables and charts from the saved data
python -m tools.verbose_summary --latest    # what the newest verbose log says
python -m pytest tests                      # tests (synthetic hands, no camera needed)
```

> **Camera index:** `CAMERA_INDEX` in `rehab/config.py` is set to `1` for one of the team's
> laptops. If you see no image, run with `--camera 0` or change the setting.

Exercise names for `--exercise`: `grip_release`, `finger_abduction`, `thumb_flexion`,
`thumb_opposition`, `finger_tapping`, `grip_squeeze`, `bubble_pinch`, `two_hand_match`,
`memory_pairs`, and the arm exercises `shoulder_flexion_raise`, `shoulder_abduction_raise`,
`hand_to_mouth`, `hand_to_head`, `elbow_extension`, `wrist_extension`, `tabletop_reach`,
`finger_to_nose_timed`.

The MediaPipe models the app uses are included: `models/gesture_recognizer.task` (hands) and
`models/pose_landmarker_lite.task` (body, loaded only for the arm exercises).

---

## Controls

**Without a keyboard:** thumbs up = yes / good / carry on, thumbs down = no / not so good. Either
hand counts; hold the gesture for a moment. Ratings: hold up 1 to 5 fingers. Memory pairs: point
at a card and hold still.

| Key | What it does |
|---|---|
| space | yes / start / continue / pause; skips a rest |
| `k` | skip: the exercise (from its card to the last set; reps done are kept) or the rest (also a **Skip** button to click) |
| `i` | in the menu: open / close the toolbar (or click the icon at the top right) |
| `t` / `g` / `b` / `o` | toolbar open: short sessions on / off; **new guest** (a new session for a new guest with a unique id and folder); back to her own profile (while a guest is active); make the report and open its folder (a guest's own report while a guest is active) |
| `y` / `n` | yes / no |
| `1`–`9` | in the menu: pick one exercise (two columns) |
| `0` or space | in the menu: all of today's exercises |
| `a` | in the menu: arm exercises (a second menu; `0` or `m` goes back) |
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
4. **Menu:** one exercise, all of today's exercises, "Arm exercises" (a second menu, key `a`),
   "Finish for today", or "My profile". With `--exercise`
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
| 4 | `thumb_opposition` — touch your fingertips | per finger, thumb-to-fingertip distance (3D and in the picture) between her calibrated touch of that finger and her open hand | a sequence of touches: guided (level 1), then from memory (levels 2+) |
| 5 | `finger_tapping` — lift one finger at a time | fingertip height above the flat hand and MCP angle, averaged, against a threshold per finger (the ring finger lifts least) | a sequence of lifts: in order, called out, or a remembered pattern |
| 6 | `grip_squeeze` — squeeze a rolled cloth | finger closure between "holding" and "squeezing" the cloth | squeeze, hold 4 s → relax 4 s (every other day) |
| 7 | `bubble_pinch` — pinch the bubble | thumb tip → index tip distance / palm size, scaled between her "open" and "pinch" | pinch, hold 2 s (a bubble shrinks and pops) → let go |
| 8 | `two_hand_match` — both hands together | the affected hand's openness drives the reps; the other hand's openness gives the **symmetry** (1 − mean gap) | open both, hold → close both, hold |
| 9 | `memory_pairs` — memory pairs | pointing: index fingertip over a face-down card, held still 1.5 s (or keys 1–8) | one finished board of 2, 3 or 4 pairs |

Default sets and reps (from the rehabilitation video, low end of stage 4): 3 × 10 for exercises
1–3, 2 × 3 rounds for thumb opposition, 2 × 2 rounds for finger tapping, 2 × 8 for grip squeeze.
All are in `rehab/config.py`. Short sessions (`--short` or the toolbar): 1 set of 3 reps (1 round
for the sequences and the memory game) and a 5 s rest between exercises.

Detection details: the thumb-opposition calibration measures a touch on **each** fingertip (the
ring and little fingers are further from the thumb and tracked less well), and when the prompted
finger and a neighbour are both at the thumb, the prompted finger counts. Finger tapping scales
each finger's lift threshold (`finger_scale`: ring 0.55, little 0.6, middle 0.85 of the index),
lets the flat position follow a hand that settles on the table, and prefers the prompted finger
when a neighbour rises with it; neighbours may move a little without lowering the isolation
score. Finger counting for the ratings looks at how far each fingertip reaches beyond its middle
joint, which stays reliable when the fingers point at the camera.

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

## Arm exercises and movement benchmarks

The arm exercises follow `benchmarks/BENCHMARK_PLAN.md`. Every number they use is in
`benchmarks/benchmarks.json` (rebuilt from the CSVs in `benchmarks/data` by
`python benchmarks/tools/build_benchmarks_json.py`) with its source, or is marked **proposed**:
a project default to tune in testing. The logic is in `rehab/benchmarks.py`.

| Layer | Question | Source | In the app |
|---|---|---|---|
| 1. Form | Was it done the right way? | Fugl-Meyer (FMA-UE) 2026 manual | an FMA-style score 0/1/2 per rep: 90° for the arm raises, a straight elbow, no trunk lean or shoulder hike, the lower score when the peak is within the tolerance of the threshold |
| 2. Range | How far is far enough? | Soucie 2011, Gates 2016, Bain 2015, Levin 2004 | normal range as the ceiling and to reject tracking errors; the angles daily tasks need as milestones ("That's about the arm lift you need to drink from your cup of tea."); the trunk's share of a reach |
| 3. Personal | What is her target today? | FMA: unaffected side first | the right arm, then the left, is measured; targets climb from her own baseline to her right arm's range, in steps no smaller than the tolerance |
| Tolerance | Is the change real? | Lazem 2026, Jayavel 2025 | within a rep, differences smaller than the 95% limits of agreement are noise (5–9°); between sessions "your arm lifts higher" is only said for a change of at least the minimum detectable change (MDC, 11–33°) |

| Exercise | View | Measures | Scores and milestones |
|---|---|---|---|
| `shoulder_flexion_raise` | side-on | shoulder elevation, elbow, arm in plane, trunk | FMA 13 (and 16); 71 / 86 / 90 / 105 / 108° |
| `shoulder_abduction_raise` | facing | shoulder elevation, elbow, shoulder hike, trunk | FMA 15 |
| `hand_to_mouth` | side-on | elbow flexion (and shoulder) | 81 / 100 / 121° (drinking); empty cup only |
| `hand_to_head` | side-on | elbow flexion, wrist to ear | FMA 7: the hand reaches the ear |
| `elbow_extension` | side-on | elbow flexion down to 0° | FMA 10; a contracture of 30° or more: not testable |
| `wrist_extension` | side-on, forearm on the table | hand against the forearm (hand + body tracking) | FMA 19 (at most 1: no resistance); 15 / 33 / 40° |
| `tabletop_reach` | halfway | hand and trunk movement | Reaching Performance Scale trunk score 0–3 |
| `finger_to_nose_timed` | facing | touches, time, touch error | FMA 31–33 (adapted: eyes open), right arm first |

**A session with an arm exercise:** activity card → **calibration** (the first time and then
weekly, as the assessment: the right arm first, 2 practice reps and 3 recorded, then 3 with the
left; about 10 s rest between reps; she may need to turn her chair) → **setup check** (the camera
must see the arm, not at the edge, from the right direction: "Please turn your chair so your left
arm is nearest the screen.") → sets. A rep is: rest position held → move → today's target (hold)
→ back. After a rep she hears at most one cue ("Keep your elbow straight.", "Keep your back
against the chair."), and at most one technique cue per set. A rep with a compensation still
counts. When the camera loses her arm it says so as its own fault: "I can't see your arm. Please
move into the box."

**Tracking in a real room** (`rehab/body.py`, checked on photos with the app's pose model):

- **No hips needed.** Seated at a table the hips are hidden or below the picture; MediaPipe then
  guesses them (visibility 0.1–0.2), and the guess tilts the shoulder angle by up to 20°. The
  arm exercises only need the arm itself: without visible hips the shoulder angle is measured
  against the vertical (she sits upright), and a trunk lean is seen from the shoulders moving
  (the hips stay on the chair). An arm raise only needs the shoulder and elbow in view during
  a set, so a hand lifted out of the top of the picture does not stop it.
- **Guessed points don't count.** A measure is left out while one of its points is not really
  seen (the far arm side-on, a wrist out of the picture), and those points are not drawn. The
  arm being trained is thick and labelled "LEFT ARM" on the camera image.
- **Which arm.** Side-on, the arm nearest the camera is worked out from which way she faces (and
  from depth). A left/right swap by the model, or a mirrored video, is put back. Hands are
  matched to the arm whose wrist they are at, not to MediaPipe's handedness label.
- **Views.** A real torso facing the camera measures 0.5–0.6 shoulder width per torso length, so
  "facing" starts at 0.45 (it was 0.60, which asked people who faced the camera to face it); the
  setup check accepts a little more (`VIEW_ACCEPT`). A needed point off the picture gets "Please
  move back a little" instead of "I can't see your arm".
- **Reps.** A rep ends when the arm is back in its start posture or has come back 75% of the way
  (`ARM_RETURN_SHARE`); before, it had to reach the exact start angle ±5°, and many reps never
  ended. Movements under two tolerances that never reach the target are jitter, not reps
  (`ARM_MIN_REP_TOLERANCES`). A pause on the way up is still one rep, and the hold is only left
  two tolerances below the target.
- **Speech in step.** The prompt to move ("Now lift ...", "And again.") waits until the rep count
  and cue have been said, and is skipped when she has already started. "And hold." is an
  instruction, never dropped because the coach was talking. The 6 Hz filter follows the real
  frame rate (pose and hands together often run at 10–15 fps, not the camera's 30).

**Targets (levels):** target = her baseline + level × tolerance, up to the lowest of her right
arm, the therapist's limit and the norm (160° active shoulder flexion). The level changes once
per session: +1 after two sessions in a row with at least 80% clean reps (and compensation not
rising), −1 below 50% success or when most reps were compensated, one lower after two missed
sessions and on a difficult day. Difficult days never change the saved level. (Proposed rules.)

**Therapist profile** (`benchmarks/therapist_profile.json`): her sex and age for the norms, the
affected side, seat type, lab or home MDC, passive limits and contractures, and per arm exercise
whether she may do it alone and its sets, reps, rest and hold. Exercises not cleared for her
alone show "with your therapist" in the menu and start only with `--exercise`.

**Hand exercises:** grip and release also gets FMA 25/24-style scores and Bain 2015's "open enough
/ closed enough for 90% of daily tasks" as milestones, and "your hand opened wider" is only said
for 20° per joint or a rise over three sessions (finger angles are not validated). Every held
pinch in bubble pinch, and every index touch in thumb opposition, gets an FMA 28-style pinch
score (at most 1). The arm exercises' intro card shows a seated figure doing the movement, like
the demo hand of the hand exercises.

The scores are **FMA-style**, for coaching and the report's measurements, not a clinical FMA-UE.

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
| `benchmark_reps.csv` | arm (and benchmark) reps: level, target, tolerance, start and peak angle, FMA-style score, coaching success, compensation, rules broken, trunk / shoulder / plane measures, cue, tracking quality |
| `benchmark_sessions.csv` | per session with arm exercises: seat, camera check, clean success and compensation rates, bests, improvement claims, milestones, level changes |
| `tracking_check.csv`, `validation.csv`, `body_check.csv`, `angle_agreement.csv` | results of the tools below |

`profile.json` also keeps the arm calibrations (both sides, her baseline, symmetry), the weekly
assessments, the levels and the milestones already announced.

The two benchmark logs are the objective measures for the report's testing section:
repetitions, clean success, compensation and the FMA-style profile over time.

`profile.json`, `garden.json` and `sessions.csv` are written to a temporary file first and then
renamed, so a crash never leaves half a file. An unreadable file is set aside
(`*.broken-<time>`), and the coach starts as on a first day rather than remembering something
wrong. Older data files are upgraded automatically (new CSV columns, `thresholds` → `targets`).

**Guests** (`--guest`, or "New guest" in the toolbar) are each a new user with a unique id,
`guest-<number>-<date>-<time>` (e.g. `guest-007-20260925-143012`: the number is easy to write on
a questionnaire, the time keeps it unique). Everything of a guest is in one folder named after
the id:

```
data/guests/guests.csv                          every guest: id, created, hand
data/guests/<id>/                               profile, reps, history, sessions ... (as above)
data/guests/<id>/<id>_report/                   the guest's own report (made when the guest's
                                                session ends, and by "Make report")
data/guests/<id>/verbose/<id>_<session>/        verbose logs of the guest's sessions
```

A guest profile starts with the first coach name and three activities chosen, so there are no
first-time questions. `sessions.csv` has a `user_id` column (the guest's id, or `eleanor`).
`python -m tools.report` reads `data/` and every guest folder and lists each guest as its own
user (`users.csv`, "Per user"); `python -m tools.report --data data/guests/<id> --user <id>`
makes one guest's report.

**Verbose logs** (`python main.py --verbose`) are for tuning the calibration, the detection and
the tracking after a session. One folder per session, `<data>/verbose/<user>_<session>/`:

| File | What |
|---|---|
| `session.json` | who and when, software versions, camera, the whole `config.py`, the stored calibrations; at the end frames, frame rate, loop time and dropped video frames |
| `frames.jsonl.gz` | one line per camera frame: timing, every hand seen (21 image + 21 world landmarks, handedness, gesture), the measures the exercises use, the quality problem, the stage, and the detector's inner state (lift scores and flat baseline, touch closeness, candidate, prompted finger, finger counts during ratings, calibration step) |
| `events.jsonl` | stage changes, keys, clicks, every line the coach spoke, each exercise's settings and thresholds, calibration results, detections (finger, prompted finger, right or wrong), reps, ratings, skips, toolbar actions |
| `video.mp4` | the camera image (not mirrored) on a real-time timeline; `video_frame` in `frames.jsonl.gz` is its frame number. `python main.py --video <folder>/video.mp4` replays the session (`--no-video` leaves it out) |

`python -m tools.verbose_summary <folder>` (or `--latest`) writes `summary.md` and
`summary.json` into the folder: frame rate and loop time; how often the hand was seen, the
right hand, too far, palm turned away, handedness flips and drop-outs; the noise floor while the
hand was held still in calibration (jitter per measure and per landmark, in mm); per exercise and
finger the prompts, found, wrong fingers, how strong the prompted finger's signal was against its
threshold and the neighbours', near misses, and a suggested setting where the numbers point to one
(e.g. `finger_scale` for a finger whose lifts stay close to the threshold); and how steady the
finger counts were during the ratings. The video shows the person: record only with consent and
delete the logs when they are no longer needed.

**Deleting the profile** (profile screen, `d` then `y`) removes `profile.json`, `garden.json`,
`reps.csv`, `history.csv`, `sessions.csv` and the two benchmark logs together. By default they are moved to
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
- **`benchmarks/therapist_profile.json`**: the arm exercises she may do alone, their sets, reps,
  rest and hold, her side, sex and age for the norms, and any limits or contractures.
- **`benchmarks/data/*.csv`**: the benchmark numbers; rebuild `benchmarks.json` afterwards.

---

## Code structure

```
main.py                    wiring: Sense → features → Think → Act (nothing imports it)
rehab/config.py            all settings and per-exercise parameters
rehab/Sense.py             camera / video, MediaPipe GestureRecognizer → HandObservation, PoseLandmarker → PoseObservation
rehab/features.py          landmarks → HandFeatures (angles, openness, spread, distances, quality flags,
                           the other hand, raised-finger count)
rehab/body.py              pose landmarks → BodyFeatures (arm angles in pixels, trunk, view, setup check)
rehab/filters.py           One Euro filter; 6 Hz low-pass for body angles
rehab/benchmarks.py        benchmarks.json, FMA-style rep scoring, ladder, MDC claims, progression, RPS, FMA 31-33
rehab/calibration.py       per-exercise capture of her range (median over a 3 s hold); arm: right then left, setup check
rehab/exercises/base.py    Exercise base, hysteresis, two-phase and sequence engines, rep quality measures
rehab/exercises/*.py       the nine hand exercises
rehab/exercises/arm.py     arm exercise engine: rep state machine, form and compensation rules, cues
rehab/exercises/arm_raise.py, elbow_bend.py, wrist_extension.py, tabletop_reach.py, finger_to_nose.py
                           the eight arm exercises
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
benchmarks/                BENCHMARK_PLAN.md, benchmarks.json, data/ (sources), therapist_profile.json
assets/                    fonts (Atkinson Hyperlegible), optional icons and garden PNGs
models/                    MediaPipe models (the app uses gesture_recognizer.task)
tools/tracking_check.py    detection rate and jitter of each measure with your camera
tools/validate.py          program rep count vs. a count by hand, on recorded videos
tools/report.py            tables, charts and report.md from all saved data (including guests)
tools/report_job.py        the toolbar's "Make report": runs tools.report in the background
tools/verbose_summary.py   tracking, noise and detection numbers from a --verbose log
rehab/guests.py            a unique id and folder for every guest
rehab/verbose.py           the --verbose log: frames, events and the camera video
tools/body_check.py        arm angle jitter vs. tolerance, and agreement with a goniometer
tests/                     unit and session tests driven by a synthetic 3D hand and a synthetic body
```

---

## Tests and tools

```bash
pip install pytest
python -m pytest tests
```

The 240 tests need no camera or speaker. `rehab/handmodel.py` builds a 3D hand in any pose
and `tests/synthetic_body.py` a seated body with any arm angles, so exercises, calibration, whole
sessions, the benchmark scoring, speech priority, the motivation rules, storage and the
text-to-speech backend choice are all tested with made-up hands and bodies.

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
- `python -m tools.body_check --seconds 20` shows the arm angles live and how much they jitter
  compared with the tolerance. With `--joint left.shoulder_elevation --goniometer 90 --note ...`
  it logs the camera's angle next to a phone goniometer's in `data/angle_agreement.csv`.

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
5. Try the new parts with the real camera: holding up 1–5 fingers for a rating (the screen says
   "I see 3 fingers"), pointing and holding still over a card in memory pairs, both hands in view
   for two-hand match, and whether the finger-piano notes and the voice can be heard together.
6. Fill in `HELPER_NAME` and `HELPER_PHONE` in `rehab/config.py` if the Stop screen should show a
   person to call.
7. Arm exercises: set up as in the plan (camera about 1.5 m away, lens about 90 cm high, even
   light, short sleeves). Run `python -m tools.body_check` sitting still in each view: the
   filtered jitter should stay below the tolerance. Then hold an arm at about 5 angles measured
   with a phone goniometer, on 2 people (`--goniometer`), and compare the differences with
   Lazem 2026. Repeat a calibration on two days to see whether the MDC fits your setup.
8. Have the therapist fill in `benchmarks/therapist_profile.json` (which arm exercises she may do
   alone, limits, contractures, seat).

## At the marketplace

1. Start the first visitor with `python main.py --guest --short` (add `--hand Right` for someone
   who wants to use the right hand); for every next visitor press **New guest** in the menu's
   toolbar (`i`, then `g`). Each visitor gets a unique id and a fresh folder in `data/guests/`
   (the id is shown in the toolbar: write it on their questionnaire), and Eleanor's own data
   stays untouched.
2. Visitors play Eleanor: the coach greets them by her name, and the first-time questions are
   skipped. They still measure their own hand once per exercise.
3. After their session they answer the ratings on screen (guests also get "How easy was the coach
   to use?"). Hand out a paper usability questionnaire (e.g. SUS) as well for the report.
4. Each guest's own report is made in `data/guests/<id>/<id>_report/` when their session ends.
   Afterwards run `python -m tools.report` (or "Make report" in the toolbar from her own
   profile) for everyone together, and give `data/report/report.md` and the charts to whoever
   writes the report.

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
- The arm scores are "FMA-style", not a clinical FMA-UE. Items that need resistance are capped at
  1. The normative data stop at age 69 and are passive range; the daily-task angles come from
  young healthy adults with 3D motion capture, and 2D webcam angles only approximate them.
  MediaPipe finger angles and wrist change thresholds are not validated; Lazem's reference was
  Kinovea, not 3D motion capture. See `benchmarks/BENCHMARK_PLAN.md` section 15.
- Pain is not asked yet, so the plan's "pause when pain rises" rule is in
  `benchmarks.next_level()` but has no input. The right hand is not measured as a reference for
  grip and release, and the tremor score of finger to nose is only a rough stand-in.
- Adapting hold time instead of range is not needed: every range exercise measures range. The two
  sequence exercises keep their own levels.
- Finger counting for the ratings needs clearly straight fingers; keys 1–5 always work. A thumbs up
  counts as one finger.
- Memory pairs is a memory game, not a test of cooking skills: its numbers show game practice,
  not recovery.
- The two-hand symmetry compares each hand with its own range, so it shows how the hands move
  together, not whether they are equally strong.
- The demo hand is a drawn model, not a video of a real person.
- The Stop screen does not decide whether something is an emergency and never calls anyone.

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
8. **Testing and persona fit** ([PR #9](https://github.com/ryanvafaei/stroke-rehab/pull/9)): ratings at the end of a session, guest mode and `tools/report.py`,
   a demo hand for every exercise, Stop / "I don't feel well" with 112, "Exercise 2 of 7" and a
   repeat key, the finger piano, and three new exercises: bubble pinch, two-hand match and
   memory pairs.
9. **Movement benchmarks**: body tracking and eight arm exercises scored against the
   Fugl-Meyer form rules, normative and daily-task ranges and MediaPipe's measurement error;
   right-then-left calibration as a weekly assessment, a target ladder with levels, MDC-gated
   progress claims, benchmark logs, hand benchmark scores, and `tools/body_check.py`.
