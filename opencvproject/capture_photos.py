import cv2
import os

# --- Configurações ---
base_dir = "reference_images"
num_photos = 700        # Fotos por pessoa
face_size = (100, 100)
brightness_threshold = 80
center_margin = 0.2

haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)

# Perguntar nomes
nomes_input = input("Digite os nomes separados por vírgula: ")
nomes = [n.strip() for n in nomes_input.split(",")]

cap = cv2.VideoCapture(0)
cv2.namedWindow("Capture Photos LBPH", cv2.WINDOW_NORMAL)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def is_face_centered(x, y, w, h):
    cx = x + w/2
    cy = y + h/2
    return (frame_width*(0.5-center_margin) < cx < frame_width*(0.5+center_margin) and
            frame_height*(0.5-center_margin) < cy < frame_height*(0.5+center_margin))

for nome in nomes:
    person_dir = os.path.join(base_dir, nome)
    os.makedirs(person_dir, exist_ok=True)
    print(f"\nCapturando fotos para: {nome}")
    count = 0

    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            if brightness < brightness_threshold or not is_face_centered(x, y, w, h):
                cv2.putText(frame, "Rosto centralizado e bem iluminado",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                continue

            face_crop = cv2.resize(gray[y:y+h, x:x+w], face_size)
            face_crop = cv2.equalizeHist(face_crop)

            file_path = os.path.join(person_dir, f"{count}.png")
            cv2.imwrite(file_path, face_crop)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{nome}: {count}/{num_photos}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Capture Photos LBPH", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("\nFotos de referência capturadas com sucesso.")
