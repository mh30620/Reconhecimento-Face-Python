import face_recognition
import cv2
import numpy as np

# acessa a web cam para 
video_capture = cv2.VideoCapture(0)

# carrega a imagem e aprende a reconhece-la 
wt_img = face_recognition.load_image_file("obama.jpg")
wtimg_face_encoding = face_recognition.face_encodings(wt_img)[0]

#  carrega a segunda imagem e aprender a reconhece-la 
wt_img2_img = face_recognition.load_image_file("biden.jpg")
wtimg2_face_encoding = face_recognition.face_encodings(wt_img2_img)[0]

# cria arrays para saber as imagens encoding e nomear 
known_face_encodings = [wtimg_face_encoding,wtimg2_face_encoding]
known_face_names = [ "Exemplo","Exemplo2"]

# Inicializando variaveis 
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
  # pegar um quadro de video 
while True:
    ret, frame = video_capture.read()

    # redimendionando o video para 1/4 para ir mais rapido 
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # converntendo a imagem de bgr para  rbg 
    rgb_small_frame = small_frame[:, :, ::-1]

    # Processe apenas todos os outros quadros de vídeo para economizar tempo
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
       
        top *= 4
        right *= 4
        bottom *= 4

        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



