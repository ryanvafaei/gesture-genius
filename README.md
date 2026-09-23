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
