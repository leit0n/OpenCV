import cv2
import os
import numpy as np

base_dir = "reference_images"
face_size = (100, 100)
threshold = 80  # Ajustável: maior = mais permissivo

haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)

# Preparar dados para treino
faces = []
labels = []
label_map = {}
label_counter = 0

for nome in os.listdir(base_dir):
    person_dir = os.path.join(base_dir, nome)
    if not os.path.isdir(person_dir):
        continue
    label_map[label_counter] = nome
    for file in os.listdir(person_dir):
        if file.lower().endswith(".png"):
            path = os.path.join(person_dir, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.equalizeHist(img)
            faces.append(img)
            labels.append(label_counter)
    label_counter += 1

if not faces:
    print("Nenhuma foto de referência encontrada. Execute capture_photos_lbph_final.py primeiro.")
    exit()

faces = np.array(faces)
labels = np.array(labels)

# Criar e treinar LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
recognizer.train(faces, labels)
print("Modelo LBPH treinado com sucesso.")

# Abrir webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Face Recognition LBPH", cv2.WINDOW_NORMAL)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces_detected:
            face_crop = cv2.resize(gray[y:y+h, x:x+w], face_size)
            face_crop = cv2.equalizeHist(face_crop)

            label_pred, conf = recognizer.predict(face_crop)

            if conf < threshold:
                nome = label_map[label_pred]
            else:
                nome = "Desconhecido"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{nome} ({int(conf)})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Face Recognition LBPH", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrompido pelo utilizador")

finally:
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
