import cv2
import numpy as np

# Define the color ranges for the objects to count
color_ranges = {
    "golden": [(10, 100, 100), (25, 255, 255)],  # HSV range for golden
    "white": [(0, 0, 200), (180, 25, 255)],      # HSV range for white
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

    object_count = {color: 0 for color in color_ranges}

    # Process each color range
    for color, (lower, upper) in color_ranges.items():
        # Create a mask for the specified color
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count the number of objects (contours) for the specified color
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                object_count[color] += 1
                # Draw the contour and a label
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                # Get the centroid of the contour for labeling
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(frame, color, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the object counts on the frame
    for idx, (color, count) in enumerate(object_count.items()):
        cv2.putText(frame, f'Count of {color}: {count}', (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Counting', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
