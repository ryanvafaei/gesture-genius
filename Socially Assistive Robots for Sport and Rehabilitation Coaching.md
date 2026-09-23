# Topic 2: Socially Assistive Robots for Sport and Rehabilitation Coaching  
## The Challenge: 
In this assignment, your group will design and develop a coaching system aimed at ischemic stroke rehabilitation. The system should help potential patients recover motor functions by coaching the users, tracking their joint movements, analyzing their movements, and providing real-time feedback. The system will follow the Sense - Think - Act paradigm and cater to a specific persona: older adult, adult, or young adult. The feedback should be tailored to the persona’s needs and can be provided through text, speech, sound, a visualization, haptics, and/or a virtual agent. 

### Assignment Overview: 
**Design:** <br>
- Design a *coaching system* for your persona guiding them through a rehabilitation exercise.
- Research suitable *rehabilitation exercises*. 
- Think about *motivational strategies* to *engage* the user that you can implement in your system.  

**Sense:**
- Track the user's joint movements using your laptop webcam, python and the mediapipe package. 
- Think about ways to make the joint tracking more robust to *jitter*.  
- Detect **when** and **where** the user may need physical or motivational guidance.

**Think:** 
- Analyze the user’s **joint angles** and **performance** against *predefined benchmarks* for ischemic stroke rehabilitation, such as *"range of motion, joint flexibility, and correct exercise form"*.
- Your system should **analyze the captured data** to assess the user's *movement patterns, posture, or performance*. <br> This analysis will help determine if the user follows the **system's instructions, performs the activity correctly, or needs adjustments**.  
- Decide whether **feedback** is needed, what should be said, where touch feedback might be applied, and what direction or type of **guidance** would help.

**Act:**
- Provide *personalized, real-time feedback* (motivational or corrective) through the selected medium (e.g., GUI, audio signals, speech, avatar, virtual hand, vibration cues). 
- Ensure that the feedback is tailored to the persona's age and cognitive capabilities. 
 
### Materials: 
Python codesLinks to an external site. with templates for the sense, think, act components:
- The **sense component** uses *mediapipe* to track the user joints,
- the **think component** uses the *transition library* to model a state machine making the decisions on the user’s posture state, 
- the **act component** uses *opencv*  to visualize the user's skeleton and prints text that instructs the user what to do. 
<br><br>
These are all very simple and basic implementations that **you need to enhance and adjust** for your use cases. <br>
You will receive a short description of what stroke and post-stroke symptoms are and rehabilitation and persona that describes the condition of your target user, the needs and goals. Use this persona to develop your system. 
 

### What you need to have for the challenge: 
- Everyone has a computer and the software running 
- Your system design on paper 
- By the start of the tutorial, you should have started the implementation and have a system running. 
 
---

## Persona
### Brief
- Eleanor, Older Adult / Retired (72 years)
- Challenges: Reduced flexibility, slower movement, and potential cognitive decline. 
- Feedback: Simple, clear, and possibly slower-paced instructions with motivational encouragement to avoid frustration. 
### Demography
- Age: 72 (Older Adult)
- Occupation: Retired Librarian
- Stroke Type: Ischemic stroke affecting the right side
- Mobility Status: Hemiparesis (weakness) on the left side, uses a cane for walking.
- Cognitive Status: Mild cognitive impairment (slowed processing, occasional memory lapses).

### Goals
- Improve left-side mobility and hand function for daily activities like cooking  and gardening.
- Maintain independence and reduce the risk of falls.
- Stay socially engaged to prevent isolation.

### Needs
- Rehabilitation Focus: Motor recovery, balance improvement, and cognitive stimulation.
- Technology Comfort Level: Low to medium; familiar with using smartphones but hesitant about new technology.
- Support System: Family (children and grandchildren) is supportive, but she values her independence.

### Rehabilitation Strategy
- Physical Therapy: Focus on exercises to strengthen the left leg and arm, along with balance training.
- Occupational Therapy: Fine motor skills rehab with everyday tasks like buttoning clothes and holding utensils.
- Cognitive Exercises: Memory games or brain exercises that can be done in short sessions.
 
### Group Formation and Task Assignment 
- Form a group of 4 members. 
- Assign roles within your group (e.g., data analysis, decision algorithm development, interaction management, feedback mechanism development, output modality implementation). 
- Discuss and outline the key tasks to be completed. 
- Detailed tasks 

## System Design 
- Choose a rehabilitation exercise commonly used for ischemic post-stroke patients (e.g., arm or leg raises, hand grip exercises, or balance training) that suits your persona.  You don’t have to research what an ischemic stroke is in detail.  
- What task do they have, how can the system motivate them to do the exercises and provide feedback? 
- Define the joints to track based on your exercise (e.g., shoulder, elbow, knee) and design the logic for movement analysis and how and when your system should give feedback to the user to help them (e.g., what constitutes correct vs. incorrect performance). 
- Decide on the type of feedback and the method of delivery (text, speech, visual feedback, virtual agent), and feedback style suitable for your persona. Think about the trade-off between creating visually appealing and engaging feedback modalities vs. simplicity and on-point feedback delivery.  
- Design the system based on the needs and capabilities of your assigned persona. Examples: 
	- Older Adults: Focus on slower, simplified movements with encouraging feedback. 
	- Adult: Focus on accurate feedback for precision and progress tracking. 
	- Children: Use gamification, virtual characters, or playful feedback to keep the child engaged. 
 

 
## Implementation 
- Sense: Set up a motion tracking system using MediaPipeLinks to an external site.. See if there are additional modules that are of interest to you besides skeleton tracking. Ensure the system can reliably detect and track the necessary joints. 

- Think: Implement the logic to analyze the tracked movements. This could involve setting thresholds or conditions for correct/incorrect movement, transition between state in the rehabilitation exercises, and your logic to decide when to provide feedback. 

- Act: Develop a feedback and instruction mechanism to guide the user through the exercise. Develop an output modality suitable for your persona with motivating features (e.g., a GUI with virtual playground, verbal feedback for the user, a virtual representation of the user guiding them through the exercise, a virtual avatar).  

## Testing and Refinement 

- Test the system with different users mimicking the assigned persona. 
- Check that the system correctly tracks movements and provides appropriate feedback. 
- Refine the movement analysis and feedback for better accuracy and user engagement 
- Reflect on the purpose of your system and identify objective task or behavior measurements (e.g., number of repetitions, persistence, execution accuracy) and subjective measurements, such as system usability, physical activity enjoyment, and perceived exertion, that could help you to identify the effectiveness of your coaching system.  
 
---

## Deliverables and grading: 
### Report and code (100 points) 
- Write a report (2-3.5 pages A4 (excluding tables and figures), Arial, 11pt) describing your system and design considerations. Submit the code of your working prototype coaching system. 
- Describe your design considerations including the chosen exercise and task, the instruction and feedback strategies. (20 points) 
- Describe and reflect on how you implemented the system’s functionality. How did you implement reliable movement tracking, how did you analyze the movements and provided feedback. Reflect on the reliability of your system. (25 points) 
- Reflect on the user experience and innovation. Are the instructions clear and engaging for potential users? How does your system adjust to user performances? How engaging is your system? What is the complexity of adjustment of your system to provide feedback based on well or poor performances and user’s skill level? How does your system provide varying non monotonous feedback?  Is it easy to use or do users need further instructions? (30 points) 
- How would you improve your coaching system based on your user testing and other groups testing your system? (15 points) 
- Reflect on the marketplace session, testing and discussion with other groups. Which difference where there in approaching the challenge? What did you learn? (10 points) 
- Marketplace Demonstration and Testing 

### Presentation
- Groups for each persona will present their work in a marketplace and the groups of the other two personas can go around and test your system. 
- Let them try it out, think out loud and ask them questions while they give you feedback. Observe the user experience when others interact with your system and when you interact with the other groups' systems, as it will provide valuable insights into the impact of your design decisions and the different groups' design choices and implementation. Give them small questionnaires to assess your system’s usability and user experience. 
- Discuss common design choices for motion tracking, analysis, and feedback. Compile a list of potential improvements or extensions for your systems. 
 

## Evaluation Criteria 

| Category | Description | Max Points |
| :--- | :--- | :--- |
| **System Design and Persona Customization** | How well is your rehabilitation coaching system designed to your persona's cognitive, physical and motivational needs? | 20 |
| **Functionality** | How well does your system track relevant joints? How accurate is the movement analysis? How does it handle edge cases? Is your feedback delivered in realtime? | 25 |
| **User Experience** | How clear and concise are the system's instructions and feedback? How motivating is your system in engaging users? Does it take the persona's specific needs and performance into account? Is it intuitive to use? How does your system's output modality implementation enhance the user interaction for your persona (e.g., engaging visuals, adaptive virtual agents, tailored speech)? | 30 |
| **Testing and refinement suggestions** | How thoroughly have you tested your system and identified key areas of improvement to accommodate for your persona? | 15 |
| **Reflection on Cross Persona Testing** | How was your experience and what did you learn from testing other groups' systems? Have you experienced differences and similarities between your system and the system of the other personas? | 10 |

 

---

## Tools and Resources: 
User tracking technology (i.e., MediaPipe,) 
Programming environment (e.g., Python, PyCharm) 
Text-to-Speech libraries (e.g., Google TTS, pyttsx3) 
Speech-to-text (e.g., speech_recognition) 
Animation or game development (e.g., PyGame, cocos2d, pyglet) 
Machine Learning Libraries (e..g, scikit-learn, pytorch-lightning) 
Interaction and behavior modelling (e.g., pytrees, transition) 