import cv2
import mediapipe as mp

# Initialize MediaPipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup camera capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Increase frame size (width=1280, height=720 for HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Use the hands model for detecting hand landmarks
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        result = hands.process(rgb_frame)

        # If hands are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on the hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark coordinates (wrist and tips of each finger)
                landmarks = hand_landmarks.landmark

                # Finger landmarks: [4, 8, 12, 16, 20] (Thumb, Index, Middle, Ring, Pinky)
                finger_tips = [4, 8, 12, 16, 20]
                fingers = []

                # Check if fingers are open (landmark y-position of finger tip < that of knuckle)
                for i in range(1, 5):  # For fingers (excluding thumb)
                    if landmarks[finger_tips[i]].y < landmarks[finger_tips[i] - 2].y:
                        fingers.append(1)  # Finger is open
                    else:
                        fingers.append(0)  # Finger is closed

                # Check thumb (landmark x-position for thumb)
                if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 1].x:
                    fingers.append(1)  # Thumb is open
                else:
                    fingers.append(0)  # Thumb is closed

                # Count the number of fingers that are open
                finger_count = fingers.count(1)

                # Display the number of fingers in orange color (BGR = (0, 165, 255))
                cv2.putText(frame, f'Fingers: {finger_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        # Show the frame with the finger count
        cv2.imshow("Finger Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
