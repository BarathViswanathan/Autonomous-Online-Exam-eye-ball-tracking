import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("Error: Failed to capture frame!")
        break  # Exit loop if frame is empty

    # Convert to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Define Region of Interest (ROI) for eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            # Extract eye region for pupil detection
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Apply thresholding to isolate pupil
            _, thresh = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)

            # Find contours of the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            if contours:
                # Find the largest contour (likely the pupil)
                (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
                
                # Convert to integer values
                cx, cy, radius = int(cx), int(cy), int(radius)

                # Draw a circle around the pupil
                cv2.circle(eye_color, (cx, cy), radius, (255, 0, 0), 2)

                # Draw crosshairs on the pupil
                cv2.line(eye_color, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 2)  # Horizontal line
                cv2.line(eye_color, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 2)  # Vertical line

    # Show the frame
    cv2.imshow("Eyeball Tracking", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
