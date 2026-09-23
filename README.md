# Hand rehabilitation coach (Eleanor)

`main.py` now runs a hand coach that **measures** the hand instead of classifying gestures.
Every exercise is a continuous measure (an angle or distance from the 21 hand landmarks),
scaled to Eleanor's own calibrated range, with hysteresis thresholds for each rep phase.
The built-in gesture recognizer is still in the pipeline, but only as a logged secondary check.

```bash
pip install -r requirements.txt
python main.py                              # full session
python main.py --exercise grip_release      # one exercise
python main.py --video recording.mp4        # run on a recording
python main.py --no-speech                  # print instead of speaking
python -m pytest tests                      # tests (synthetic hands, no camera needed)
```

**Keys:** space starts, pauses and continues (and skips a rest). `q`/Esc stops.
Speech uses `say -r 140` on macOS, `espeak` on Linux, `pyttsx3` as a fallback.

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

greeting → for each exercise: short instruction → calibration (first time, or when older than
14 days; otherwise space within 6 s to recalibrate) → sets with rests → summary with a progress
message compared with her own earlier sessions ("Your hand opened 12% wider than last week").
When her range drops over several reps the coach offers a rest. After a good session the target
is raised a little (`AUTO_PROGRESSION` in `rehab/config.py`, which a therapist can switch off).

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
rehab/Think.py             Coach (quality checks, logging, fatigue) + SessionManager (session flow)
rehab/Act.py               speech thread + drawing (skeleton, bar with target line, sequence, subtitles)
rehab/storage.py           data/profile.json, data/reps.csv, data/history.csv
tools/tracking_check.py    Phase 1: detection rate and jitter of each measure with your camera
tools/validate.py          Phase 8: program rep count vs. a count by hand, on recorded videos
tests/                     unit and session tests driven by a synthetic 3D hand
```

`data/` holds personal data and is not committed.

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
