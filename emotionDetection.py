import cv2
from fer import FER

# Initialize the emotion detector
detector = FER()

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Use the FER library to detect emotions
    emotions = detector.detect_emotions(frame)

    # Draw rectangles and labels for each detected face and their emotions
    for emotion in emotions:
        (x, y, w, h) = emotion["box"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Get the dominant emotion
        dominant_emotion = emotion["emotions"]
        emotion_name = max(dominant_emotion, key=dominant_emotion.get)
        
        # Display the emotion name
        cv2.putText(frame, emotion_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
