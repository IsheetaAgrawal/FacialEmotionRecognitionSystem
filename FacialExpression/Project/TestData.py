def Test():
    import cv2
    import numpy as np
    from keras.models import model_from_json

    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # load json and create model
    json_file = open('.\Project\\model\\trained_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # load weights into new model
    emotion_model.load_weights(".\Project\\model\\trained_model.h5")
    print("Loaded model from disk")
    
    # start the webcam feed
    cap = cv2.VideoCapture(0)

    # pass here your video path
    # cap = cv2.VideoCapture("C:\\Users\\Harsh Pathak\\Downloads\\Emotion.mp4")

    emotions = [0] * 7
    counter = 0

    while counter <= 10:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('.\Project\\FaceDetection\\FaceDetectionTrainedData.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            emotions[maxindex] += 1

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter += 1

    cap.release()
    cv2.destroyAllWindows()

    max_pos = 0
    sec_max_pos = -1
    for i in range(7):
        if emotions[i] > emotions[max_pos]:
            max_pos = i

    for i in range(7):
        if i != max_pos and (sec_max_pos == -1 or emotions[i] > emotions[max_pos]):
            sec_max_pos = i

    if max_pos == 4:
        if emotions[sec_max_pos] > 20:
            max_pos = sec_max_pos

    if emotions[max_pos] == 0:
        return "No Face detected"
    
    print(str(emotion_dict[max_pos]))
    return str(emotion_dict[max_pos])