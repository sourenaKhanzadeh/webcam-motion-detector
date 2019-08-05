import cv2, time
import numpy as np

first_frame = None
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    status = 0
    if check:
        frame = cv2.resize(frame, (600, 600))

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)


        if first_frame is None:
            first_frame = gray_frame
            continue

        delta_frame = cv2.absdiff(first_frame, gray_frame)
        thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 1000:
              continue
            status = 1
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+ w, y + h), (0,255,0), 3)

        faces = face_detector.detectMultiScale(gray_frame,
            scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in faces:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0 ,0xff))

        cv2.imshow("Capturing", frame)
        cv2.imshow("Delta Frame", delta_frame)
        cv2.imshow("Threshold Frame", thresh_frame)


    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    print(status)

video.release()
cv2.destroyAllWindows()