# Hand rehabilitation coach (Eleanor)

`main.py` now runs a hand coach that **measures** the hand instead of classifying gestures.
Every exercise is a continuous measure (an angle or distance from the 21 hand landmarks),
scaled to Eleanor's own calibrated range, with hysteresis thresholds for each rep phase.
The built-in gesture recognizer is still in the pipeline, but only as a logged secondary check.

```bash
pip install -r requirements.txt
python main.py                              # starts with a menu to choose the exercise
python main.py --exercise grip_release      # one exercise, no menu
python main.py --video recording.mp4        # run on a recording
python main.py --no-speech                  # print instead of speaking
python -m pytest tests                      # tests (synthetic hands, no camera needed)
```

**Answers without a keyboard:** thumbs up = yes / good, thumbs down = no / not so good (either
hand, held for a moment). **Keys:** space = yes / continue / pause (and skips a rest), `y` / `n`
= yes / no, `q`/Esc stops (what was done is kept). In the menu, press a number (`1`–`6`) to pick
one exercise, `0` / space for all of today's exercises, or `7` to finish for today; the up and down
arrows move the highlight. `m` goes back to the menu at any time.
Speech uses `say -r 145` on macOS and `pyttsx3` on Windows and Linux (`rehab/tts_util.py` detects
the system; on Linux the `espeak` command is used when pyttsx3 is not available). Voice and rate
are per user in `data/profile.json`.
Speech is slow, so every message is checked again just before it is spoken and skipped when
she has already done what it asks. Calibration only starts timing a position after its prompt
has been spoken, and a clearly wrong calibration (e.g. "open" less open than "closed") is measured
again, also when it was stored by an earlier session.

### Exercises

| Exercise | Measure | One rep |
|---|---|---|
| `grip_release` | average finger openness (MCP+PIP+DIP flexion), per finger scaled to her range | open, hold → close, hold |
| `finger_abduction` | sum of the 3 gaps between fingers, angles in the palm plane | spread, hold → together, hold (pauses if fingers bend) |
| `thumb_flexion` | thumb MCP+IP flexion combined with thumb tip → pinky MCP distance | in, hold → out, hold |
| `thumb_opposition` | thumb-to-fingertip distances / palm size | a sequence of touches (guided, then memory levels) |
| `finger_tapping` | fingertip height above the calibrated flat hand + MCP angle | a sequence of lifts (in order, called out, or a remembered pattern) |
| `grip_squeeze` | finger closure between "holding" and "squeezing" the cloth | squeeze, hold 4 s → relax 4 s (every other day) |

Per rep it logs: range reached (% of calibration), movement time, smoothness (speed peaks),
hold stability, compensation (palm rotation, wrist movement, other fingers moving), hints given,
and exercise-specific details (lagging finger, isolation score, correct/wrong touches, level).

### Session flow

```
(first time: choose the coach's name, pick 3 favourite activities)
greeting → check-in → menu or today's plan → [activity card → exercise → rest] × n
→ summary → garden → goodbye
```

- **Greeting:** chosen by the days since her last session (first ever, same day, 1 day, 2–6 days,
  7+ days, after a difficult day). It mentions at most one remembered fact, and only facts stored
  in `data/`. A gap is never called a failure; after 7+ days the coach says "let's start gently",
  and targets start two steps lower.
- **Check-in:** "How is your hand feeling today?" Thumbs down switches on difficult day mode.
- **Menu:** one exercise, all of today's, or "Finish for today". With `--exercise` there is no
  menu, only "Today we'll do one exercise."
- **Each exercise:** a full-screen activity card ("This practises holding a cup of tea."), then
  calibration (first time, or when older than 14 days; otherwise space within 6 s to recalibrate),
  then sets with rests. The target is adapted after every set.
- **Summary:** one highlight from today, plus one comparison with her own history ("Your hand opened
  12% wider than last week").
- **Garden:** a new seed (she picks one of two plants) or the next growth stage, with a short
  animation.
- **Goodbye:** the same closing line every time, plus "Next time, we'll keep your rose growing."

When her range drops over several reps the coach offers a rest.

### Motivation features

Think decides what happened and emits events (`rehab/events.py`); Act decides how to say it
(`rehab/feedback.py` with the phrase bank in `content/phrases.json`). Wording can change without
touching logic, and all logic is tested without a camera or speaker.

| Feature | Where | Rules (starting values in `rehab/config.py`, not clinical values) |
|---|---|---|
| Personal bests | `progress.py` | A rep's value is the **median during the hold** (raw measure, comparable across recalibrations), not the peak frame. A best must be ≥3% above the old one. Levels: today / this week / all time; only the highest is announced, at most one per set, after the rep. No announcements in her first session ("I've saved today as your starting point."). Also per exercise: reps without hints and hold steadiness. Speed is deliberately never a best. |
| Adaptive targets | `progress.py` | The target is the "high" threshold, a % of her calibrated range, evaluated once per set: ≥80% successful reps → +3 points, <50% → −3 (silently), floor 50%, ceiling 100%. A rep is successful when she reached and held the target without a hint. The session starts one step below last time (two after 7+ days). When her holds go beyond her calibrated maximum, the calibrated range grows. `AUTO_PROGRESSION = False` freezes targets. |
| Difficult day mode | `progress.py`, `Think.py` | Switched on by a thumbs down at the check-in, a first set 15% below the median of her last 5 normal sessions, the existing fatigue check, or <50% success in two sets in a row. It then stays on for the session: targets ×0.8, one set fewer, rests ×1.5, no grip squeeze, praise for effort, no comparisons. Said once: "Let's take it easier today." Difficult sessions never change saved targets and are left out of baselines. Three difficult days out of the last five add a note for the therapist in `sessions.csv`. |
| Specific praise | `feedback.py` | Only for events that were really measured ("Your ring finger opened more that time." needs the per-finger data). A soft chime every successful rep; one spoken phrase at most, the most important event, or a short generic one about every 3 reps. The last 3 phrases of a category are never reused. |
| Speech priority | `Act.py` | Instructions can jump ahead of waiting praise and cut off older praise. Praise never delays an instruction. Rep praise waiting over 3 s is dropped instead of played late. |
| Coach character | `content/character.json`, `memory.py` | She picks the name ("Iris" or "Robin") at the first session. Warm, calm and respectful. A simple face with three expressions (neutral, happy, encouraging). The same opening, check-in and closing lines every session. |
| Daily activities | `content/activities.json` | Six activity cards at the first session, and she keeps up to three. Each exercise shows its link (her favourites first) on a card and as a small icon. Milestones at 50, 100, 250 and 500 reps per activity. Framed as practice toward an activity, never a promise. Please have a therapist check the links. |
| Garden | `garden.py`, `ui.py` | Grows from showing up, not performance: every session with a completed exercise waters it, difficult days too. Seed → sprout → leaves → bud → flower; 8 plots, then "a new season". A best adds a bee, a milestone a butterfly. Nothing ever wilts or goes backwards. A watering can shows the exercises done today. |

Screens use Pillow with Atkinson Hyperlegible (`assets/fonts`, SIL Open Font License). Icons and
garden pictures are drawn in calm, flat colours. PNGs exported from Figma are used instead when
present: `assets/icons/<icon>.png`, `assets/garden/<plant>_<stage>.png` (stage 0–4),
`assets/garden/{bee,butterfly,can,bed}.png`.

### Files

```
main.py                    wiring: Sense → features → Think → Act (nothing imports it)
rehab/config.py            all settings and per-exercise parameters (sets, reps, thresholds, hold times)
rehab/Sense.py             camera / video, MediaPipe GestureRecognizer → HandObservation
rehab/features.py          landmarks → HandFeatures (angles, openness, spread, distances, quality flags)
rehab/filters.py           One Euro filter
rehab/calibration.py       per-exercise capture of her range (median over a 3 s hold)
rehab/exercises/base.py    Exercise base, hysteresis, two-phase and sequence engines, rep quality measures
rehab/exercises/*.py       the six exercises
rehab/Think.py             Coach (quality checks, logging, rep events) + SessionManager (session flow)
rehab/events.py            events with priorities: what Think tells Act
rehab/progress.py          personal bests, adaptive targets, difficult days, milestones
rehab/memory.py            greetings and what the coach remembers
rehab/garden.py            the garden's state (grows, never wilts)
rehab/Act.py               speech with a priority queue + drawing (skeleton, bar, sequence, subtitles)
rehab/tts_util.py          text-to-speech backend by OS (`say` on macOS, pyttsx3 elsewhere)
rehab/feedback.py          events -> words from content/phrases.json (no repeats, praise frequency)
rehab/ui.py                Pillow text, full-screen cards, summary, garden, watering can, coach face
rehab/storage.py           data/profile.json, reps.csv, history.csv, sessions.csv, garden.json
content/                   character.json, phrases.json, activities.json (edit wording here)
assets/                    fonts (Atkinson Hyperlegible), optional icons and garden PNGs
tools/tracking_check.py    Phase 1: detection rate and jitter of each measure with your camera
tools/validate.py          Phase 8: program rep count vs. a count by hand, on recorded videos
tests/                     unit and session tests driven by a synthetic 3D hand
```

`data/` holds personal data and is not committed. `reps.csv` is written after every rep; the
profile, `garden.json` and `sessions.csv` go to a temporary file first and are then renamed, so a
crash never leaves half a file. An unreadable file is set aside (`*.broken-<time>`), and the coach
starts as on a first day rather than remembering something wrong. Older data files are upgraded
automatically (new CSV columns, `thresholds` → `targets`).

### Before the first real session

1. `python -m tools.tracking_check --seconds 30`. Hold up the **left** hand, palm to the camera.
   The overlay should say `hand OK`. If it complains about the wrong hand, set
   `RECOGNIZE_UNMIRRORED = False` in `rehab/config.py`. (On MediaPipe's sample photos, Tasks labels
   the real hand in the unmirrored view, so the recognizer gets the unmirrored frame by default.)
2. Record finger tapping and thumb opposition with `--record` and check the jitter table. Finger
   tapping depends heavily on the camera angle: a tilted phone stand or a camera looking down helps.
3. For the report: record test videos, count reps by hand, and run
   `python -m tools.validate video.mp4 --exercise grip_release --expected 10 --calibrate 12`.
   Results are appended to `data/validation.csv`.

### Limits

- One webcam gives estimated depth. The angles are good for relative progress over time, not
  clinical goniometer measurements.
- Grip squeeze measures holding and timing, not force.
- Finger tapping depends heavily on the camera angle.
- The Brunnstrom stage and the default sets and reps come from the video and should be confirmed
  by a therapist before real use.
- All motivation numbers (best threshold, target step, success rates, difficult-day drop) are
  starting values for testing, not clinical values.
- The app gives no medical advice. Repeated difficult days only leave a note for her therapist
  or family.
- The plan's fallback of adapting hold time instead of range is not needed: every range exercise
  measures range. The two sequence exercises keep their own levels.

The original elbow template (`coach/`) is unchanged below.

---

# **Interactive Coaching System for Post-Stroke Rehabilitation**

## **Project Overview**
This project implements an interactive coaching system for post-stroke rehabilitation. The system tracks joint movements using a camera and provides feedback to users. It features the following components:

- **Sense**: Uses Mediapipe to detect and measure joint angles (specifically elbow flexion/extension).
- **Think**: Implements a state machine to track movement patterns (flexion/extension).
- **Act**: Visualizes the user's movements and provides graphical feedback via a balloon animation that inflates with successful repetitions and explodes after 10 successful transitions.

### **Main Features**
- Tracks elbow flexion/extension movements in real-time.
- State machine logic (based on transitions between flexion and extension).
- Balloon visualization as graphical feedback for exercise progress.
- Text-to-speech (TTS) audio feedback to encourage users.

### **Dependencies**

This project uses the following additional dependencies to take care of certain functionality:

- OpenCV: For camera input and rendering the visual feedback.
- Mediapipe: For joint detection and motion tracking.
- pyttsx3: For text-to-speech functionality to provide audio feedback.
- transitions: For implementing the finite state machine to track movement states.

---

## **Setup Instructions**

### **1. Prerequisites**
Ensure you have the latest version of **Python 3.14** installed on your machine. We have recently migrated to the newest version of mediapipe which uses a [Tasks-based approach](https://developers.google.com/edge/mediapipe/solutions/guide) for performing common recognition tasks. This repository includes a lightweight pose recognition model.

Download and unzip the [project files](https://github.com/utwente-interaction-lab/FIT-Interactive-Coaching-System/archive/refs/heads/main.zip) from this GitHub repository (or `git clone https://github.com/utwente-interaction-lab/FIT-Interactive-Coaching-System`)

### **2. Installing Dependencies**

It is strongly recommended to create a virtual environment for your project, this may prevent issues with conflicting dependencies on your system (see https://docs.python.org/3/library/venv.html for
detailed instructions how to create a virtual environment on other OS). Instructions for Windows PowerShell:

```powershell
py -m venv venv/
```

Make sure to activate the environment each time you (re)open a terminal.

```powershell
.\venv\Scripts\activate
```

$${\color{red}Troubleshooting}$$: If you get a warning about the script being disabled due to the powershell execution policy, please [update the policy](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.5) accordingly:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```


Before installing dependencies you may need to first update pip to the latest version.

```powershell
py -m pip install --upgrade pip
```
You can install the required Python libraries using a package manager like `pip` for Python. Make sure your system has access to these libraries, and ensure the camera is working correctly.

```powershell
pip install opencv-python mediapipe pyttsx3 transitions
```

or you can try install the requirements

```powershell
pip install -r requirements.txt
```

---

## **How to Run the Program**

Once the dependencies are installed:
1. Ensure that your webcam is functional and has access permissions.
2. Navigate to the directory containing the project files.
3. Run the script in your Python environment.
4. The program will start tracking your elbow movements, and a balloon will inflate after each flexion-extension cycle. After 10 cycles, the balloon will explode to provide feedback.

---

## **Project Structure**
```md

├── main.py               # Entry point for the program
├── sense.py              # Handles joint detection and angle calculation using Mediapipe
├── think.py              # Implements the state machine to track flexion/extension and timeout events
├── act.py                # Handles visual and auditory feedback (balloon animation and TTS)
├── README.md             # Documentation (this file)
└── requirements.txt      # Optional: Lists the dependencies for the project
```

Each component of the system (Sense, Think, Act) is modular, allowing for easy updates and extensions. Here's a breakdown of each file's role:

* main.py: This is the main script that ties all the components together. It initializes the Sense, Think, and Act components and runs the main loop.
* sense.py: Responsible for using Mediapipe to detect joint positions, compute angles, and process the motion input.
* think.py: Contains the decision-making logic using a state machine. This tracks transitions between flexion and extension and handles timeouts for inactivity.
* act.py: Manages the visual and audio feedback (e.g., the balloon animation and text-to-speech encouragement).
* README.md: The project documentation, which provides setup instructions, project structure, and guidance for extending the code.
* requirements.txt: (Optional) Lists the Python dependencies, making it easier to install everything needed to run the project.

## **How to Extend the Project**

### **1. Add More Joints**
Currently, the system tracks only the elbow joint. You can extend the system to track multiple joints (e.g., shoulders, knees) by modifying the **Sense** component.

- **Sense Component**: Modify `sense.py` to detect additional joints using Mediapipe. Calculate and analyze other angles or movements.
  - Example: Add shoulder tracking to monitor overall arm movement.
  
### **2. Advanced Visualization**
The current visualization includes a simple balloon animation. You can improve this by implementing more complex feedback mechanisms, such as:
- Showing different animations based on specific goals.
- Incorporating 3D graphics or more interactive agents.
- Tracking the user’s progress visually over time.

- **Act Component**: Modify `act.py` to include more sophisticated visualizations. For example, you can create an animated character that mimics the user’s movement.

### **3. More Complex State Logic**
The decision-making component is based on a simple state machine that tracks flexion and extension transitions. You can expand this logic by implementing:
- More states (e.g., slow motion detection or incorrect movement feedback).
- Add more complex behavior trees or state charts to represent more nuanced movements.
- Adaptive difficulty settings based on user performance.

- **Think Component**: Modify `think.py` to introduce more states or more complex decision-making. Consider using behavior trees to model more intricate feedback.

### **4. Enhanced Feedback and Audio**
Currently, the system uses text-to-speech for basic feedback. You can extend this by:
- Adding more detailed and context-aware feedback based on the user's performance.
- Introducing different TTS voices or background music for motivation.
- Dynamic feedback based on user improvement (e.g., congratulating the user after milestones).

- **Think/Act Components**: Modify both `think.py` and `act.py` to add more intelligent feedback mechanisms. You could use different motivational messages depending on performance or vary the frequency of feedback.

---

## **Troubleshooting**

1. **No Camera Detected**: Ensure that your webcam is properly connected and that you've granted permission for the camera to be used by the program.
2. **TTS Audio Issues**: If you're not hearing the text-to-speech output:
   - Ensure you have an active internet connection (if you are using, e.g., gTTS).
   - Verify that your audio system is working and correctly configured.
   - pyttsx3 is blocking the program. Think about running it in a thread.
3. **Performance Issues**: If the program runs slowly:
   - Close unnecessary programs that might be using system resources.

For further troubleshooting you can check the official [mediapipe documentation](https://developers.google.com/edge/mediapipe/framework/getting_started/troubleshooting)

---
