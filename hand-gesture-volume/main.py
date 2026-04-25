import cv2              # Opens webcam and displays video
import mediapipe as mp  # Google's hand tracking library

# MediaPipe setup - Loading the hand-detector
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils    # helper to draw 21 dots

# Starting Webcam (0 = default laptop webcam)
cap = cv2.VideoCapture(0)

# Start the hand detector
with mp_hands.Hands(
    max_num_hands=1,        # only track one hand
    min_detection_confidence=0.7,   # how confident it needs to be to detect a hand
    min_tracking_confidence=0.7     # how confident it needs to be to keep tracking
) as hands:
    
    while cap.isOpened():           # keep looping while webcam is on
        success, frame = cap.read() # grab a frame from the webcam
        if not success:
            break

        # Flip the frame so it acts like a mirror
        frame = cv2.flip(frame, 1)

        # Convert to RGB because MediaPipe expects RGB, webcam gives BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run the hand detector on this frame
        results = hands.process(rgb_frame)

        # If a hand is detected, draw the landmarks on screen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame in a window
        cv2.imshow("Hand Tracker", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()