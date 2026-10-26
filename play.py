import json
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224

model = tf.keras.models.load_model("gesture_model.keras")
class_names = json.load(open("class_names.json"))
BAR_X = 20 + max(cv2.getTextSize(c, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for c in class_names)


def predict(roi):
    img = cv2.cvtColor(cv2.resize(roi, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    probs = model.predict(img[None].astype("float32"), verbose=0)[0]
    return class_names[int(np.argmax(probs))], float(np.max(probs)), probs


cap = cv2.VideoCapture(0)  # camera padrao do sistema
if not cap.isOpened():
    raise SystemExit("Nao consegui abrir a webcam.")

recent = deque(maxlen=7)  

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2
    roi = frame[y1:y1 + side, x1:x1 + side]

    label, conf, probs = predict(roi)
    recent.append(label)
    stable = Counter(recent).most_common(1)[0][0]

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), (0, 255, 0), 3)
    cv2.putText(frame, f"Modelo ve: {label} ({conf:.0%})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # barras de probabilidade (otimo para mostrar a "incerteza" do modelo)
    bar_max = max(0, x1 - 10 - BAR_X)  # barra cheia termina antes da caixa de captura
    for i, c in enumerate(class_names):
        bw = int(probs[i] * bar_max)
        cv2.putText(frame, c, (10, 63 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (BAR_X, 50 + i * 22), (BAR_X + bw, 64 + i * 22), (0, 200, 255), -1)

    cv2.putText(frame, f"Letra: {stable}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    cv2.imshow("Reconhecimento de Libras", frame)

    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q ou ESC
        break

cap.release()
cv2.destroyAllWindows()
