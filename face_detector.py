import numpy as np
import cv2
import face_recognition

from face_defined import get_face_defender

image = face_recognition.load_image_file("2020-03-11-214159.jpg")
known_face = face_recognition.face_encodings(image)[0]

# learn
face_defender = get_face_defender()


# Get a reference to webcam
video_capture = cv2.VideoCapture("/dev/video0")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        i = face_defender(np.linalg.norm(face_encoding - known_face))
        if i >= 0.5:
            name = f'Ruslan {round(i * 100, 1)}%'
        else:
            name = f'Unknown {round(i * 100, 1)}%'
        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
