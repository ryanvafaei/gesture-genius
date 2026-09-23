from transitions import Machine
import collections


# Think Component: Decision Making
# Things you need to improve: Add states and transitions according to your intervention/rehabilitation coaching design

class Think(object):

    def __init__(self, act_component, flexion_threshold=90, extension_threshold=120):
        """
        Initializes the state machine and sets up the transition logic.
        :param act_component: Reference to the Act component to trigger visual feedback
        :param flexion_threshold: threshold for entering the flexion state
        :param extension_threshold: threshold for entering the extension state
        """


        # Define initial state and thresholds for transitions
        self.state = 'flexion'  # Initial state
        self.previous_state = self.state
        self.flexion_threshold = flexion_threshold  # Elbow angle threshold for flexion
        self.extension_threshold = extension_threshold  # Elbow angle threshold for extension

        # Transition counter
        self.flexion_to_extension_count = 0  # Count transitions from flexion to extension
        self.extension_to_flexion_count = 0  # Count transitions from extension to flexion

        # Act component for visualization
        self.act_component = act_component

        # Define the state machine with states 'flexion' and 'extension'
        states = ['flexion', 'extension']

        # Initialize the state machine
        self.machine = Machine(model=self, states=states, initial='flexion')

        # Define transitions
        self.machine.add_transition(trigger='angle_decreasing', source='extension', dest='flexion',
                                    after='increment_ext_to_flexion')
        # conditions='is_flexion_threshold_reached',

        self.machine.add_transition(trigger='angle_increasing', source='flexion', dest='extension',
                                    after='increment_flexion_to_extension')
        # conditions='is_extension_threshold_reached',

    # Condition for transition to 'flexion' state
    def is_flexion_threshold_reached(self, angle):
        """
        Checks if the flexion threshold has been reached to trigger a state change.

        :param angle: The current elbow angle
        :return: True if flexion threshold is reached, False otherwise
        """
        return angle < self.flexion_threshold # Flexion is detected when the elbow is bent

    # Condition for transition to 'extension' state
    def is_extension_threshold_reached(self, angle):
        """
                 Checks if the extension threshold has been reached to trigger a state change.

                 :param angle: The current elbow angle
                 :return: True if extension threshold is reached, False otherwise
                 """
        return angle > self.extension_threshold # Extension is detected when the elbow is almost straight

    # Method to update the FSM state based on the current elbow angle
    def update_state(self, current_angle, previous_angle):
        """
        Updates the state machine based on the current angle (flexion or extension).

        :param current_angle: The current elbow joint angle (in degrees)
        :param previous_angle: The previous elbow joint angle (in degrees)
        """

        # Check if the angle is decreasing and we're in the extension state
        if current_angle < previous_angle and self.state == 'extension':
            if self.is_flexion_threshold_reached(current_angle):
                self.angle_decreasing()

        # Check if the angle is increasing and we're in the flexion state
        elif current_angle > previous_angle and self.state == 'flexion':
            if self.is_extension_threshold_reached(current_angle):
                self.angle_increasing()

        # Debug: Print the current state and transition counts
        # print(f"Elbow Angle: {current_angle}, State: {self.state}")
        # print(f"Transitions - Flexion to Extension: {self.flexion_to_extension_count}, "
        #       f"Extension to Flexion: {self.extension_to_flexion_count}")

    # Methods to increment transition counters (no arguments needed)
    def increment_flexion_to_extension(self):
        """
        Callback when transitioning from flexion to extension. Increments the repetition count.
        """
        self.flexion_to_extension_count += 1
        self.act_component.handle_balloon_inflation() # Inflate the balloon

    def increment_ext_to_flexion(self):
        """
        Callback when transitioning from extension to flexion.
        """
        self.extension_to_flexion_count += 1
        self.act_component.handle_balloon_inflation() # Reset timeout trigger
