import cv2
import numpy as np
import os

# ==== Config ====
BASE_DIR = "reference_images"
FACE_SIZE = (100, 100)
THRESHOLD = 80
ANIMAL_DETECT = False  # Toggle animal detection

# ==== Load models ====
print("Loading models...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if ANIMAL_DETECT:
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
    ANIMALS = {"gato", "cao", "passaro", "cavalo", "vaca", "ovelha"}
    CLASSES = ["background", "aviao", "bicicleta", "passaro", "barco",
               "garrafa", "autocarro", "carro", "gato", "cadeira", "vaca",
               "mesa", "cao", "cavalo", "mota", "pessoa", "planta",
               "ovelha", "sofa", "comboio", "monitor"]

# ==== Load faces ====
def load_faces(base_dir=BASE_DIR):
    faces, labels, label_map = [], [], {}
    for idx, name in enumerate(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        label_map[idx] = name
        for file in os.listdir(path):
            if file.lower().endswith(".png"):
                img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.equalizeHist(img)
                    faces.append(img)
                    labels.append(idx)
    return np.array(faces), np.array(labels), label_map

faces, labels, label_map = load_faces()
if len(faces) == 0:
    raise SystemExit("Nenhuma foto de referência encontrada.")

# ==== Train recognizer ====
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
recognizer.train(faces, labels)
print("Modelo LBPH treinado com sucesso.")

# ==== Detection loop ====
cap = cv2.VideoCapture(0)
cv2.namedWindow("Detecção Humano/Animal", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_detected = face_cascade.detectMultiScale(gray, 1.1, 5)
    human_found = False

    for (x, y, w, h) in faces_detected:
        human_found = True
        roi = cv2.equalizeHist(cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE))
        label_pred, conf = recognizer.predict(roi)
        name = label_map.get(label_pred, "Desconhecido") if conf < THRESHOLD else "Humano desconhecido"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({int(conf)})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if ANIMAL_DETECT and not human_found:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            if label in ANIMALS:
                (h, w) = frame.shape[:2]
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Animal: {label}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Detecção Humano/Animal", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
