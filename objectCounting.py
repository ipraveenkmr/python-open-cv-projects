import cv2
import numpy as np

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of objects (contours)
    object_count = len(contours)

    # Draw contours and display the count on the frame
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    # Display the object count on the frame
    cv2.putText(frame, f'Objects Count: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Counting', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
