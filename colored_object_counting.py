import cv2
import numpy as np

# Define the color ranges for the objects to count
# Example: counting red objects
# You can change the color range for other colors as needed
color_ranges = {
    "red": [(0, 100, 100), (10, 255, 255)],  # Lower red range
    "red2": [(160, 100, 100), (180, 255, 255)],  # Upper red range
}

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    object_count = 0

    # Process each color range
    for color, (lower, upper) in color_ranges.items():
        # Create a mask for the specified color
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count the number of objects (contours) for the specified color
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                object_count += 1
                # Draw the contour and a label
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                # Get the centroid of the contour for labeling
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(frame, color, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the object count on the frame
    cv2.putText(frame, f'Count of {list(color_ranges.keys())[0]}: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Counting', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
