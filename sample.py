import cv2
import dlib
import numpy as np
from datetime import datetime

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Initialize a list to store landmark points
landmarks_prev = []

# Define the threshold value for motion detection
MOTION_THRESHOLD = 3000  # Adjust this value based on testing

# Initialize variables to store the previous gray frame
prev_gray = None
frame_stability = []

while True:
    # Read frame by frame
    ret, img = cap.read()

    # Convert image into grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find faces
    faces = detector(gray)

    for face in faces:
        # Look for the landmarks
        landmarks = predictor(image=gray, box=face)

        # Initialize an array to store current landmarks
        landmarks_curr = []

        # Loop through all the points
        for n in range(0, 81):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Add current landmarks to the array
            landmarks_curr.append((x, y))

        # If previous landmarks are available, apply smoothing
        if landmarks_prev:
            landmarks_smoothed = []

            # Calculate the smoothed landmarks
            for curr, prev in zip(landmarks_curr, landmarks_prev):
                smoothed_x = int(0.6 * curr[0] + 0.4 * prev[0])
                smoothed_y = int(0.6 * curr[1] + 0.4 * prev[1])
                landmarks_smoothed.append((smoothed_x, smoothed_y))

            # Use smoothed landmarks for drawing
            for idx, point in enumerate(landmarks_smoothed):
                cv2.circle(img=img, center=point, radius=3, color=(0, 255, 255), thickness=-1)
                cv2.putText(img=img, text=str(idx), org=point, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.3, color=(255, 255, 255), thickness=1)
        else:
            # Use current landmarks for drawing if no previous data is available
            for idx, point in enumerate(landmarks_curr):
                cv2.circle(img=img, center=point, radius=3, color=(0, 255, 255), thickness=-1)
                cv2.putText(img=img, text=str(idx), org=point, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.3, color=(255, 255, 255), thickness=1)

        # Update the previous landmarks
        landmarks_prev = landmarks_curr

    # If previous frame is available, calculate the difference
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        movement = np.sum(diff)
        frame_stability.append(movement)

        # If the movement is below the threshold, capture the frame
        if movement < MOTION_THRESHOLD and len(frame_stability) > 10:
            # Ensure the stability over the last few frames
            if all(stability < MOTION_THRESHOLD for stability in frame_stability[-10:]):
                # Capture and save the frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"stable_frame_{timestamp}.png"
                cv2.imwrite(filename, img)
                print(f"Stable frame captured and saved as {filename}")
                # You can break the loop here if you only need one stable frame
                # break

    # Update the previous frame
    prev_gray = gray

    # Display the image with landmarks
    cv2.imshow(winname="Face Landmark Detection", mat=img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
print(frame_stability)