"""
Desafio 2: Alarme de Sonolência com EAR (MediaPipe Face Landmarker)

Requer o arquivo 'face_landmarker.task' na mesma pasta deste script.
Pressione 'q' para sair.
"""
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuração do Detector de Pontos Faciais (API Moderna Tasks)
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(options)

# IDs na ordem [p1, p2, p3, p4, p5, p6]
OLHO_ESQ = [33, 160, 158, 133, 153, 144]
OLHO_DIR = [362, 385, 387, 263, 373, 380]  # extra: usar os dois olhos deixa o EAR mais estável


def distancia(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# TODO 1: cálculo do EAR
def calcular_ear(landmarks_list, indices, largura, altura):
    # landmarks_list[i].x e .y vêm normalizados (0 a 1); multiplicamos pela
    # largura/altura do frame para obter coordenadas em pixels.
    pts = []
    for i in indices:
        lm = landmarks_list[i]
        pts.append((lm.x * largura, lm.y * altura))
    p1, p2, p3, p4, p5, p6 = pts

    vertical = distancia(p2, p6) + distancia(p3, p5)   # abertura das pálpebras
    horizontal = 2.0 * distancia(p1, p4)               # largura do olho (normalização)
    return vertical / horizontal if horizontal > 0 else 0.0


url = "teste.mp4"
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Erro: Webcam indisponível.")
    exit()

LIMIAR_EAR = 0.22        # Abaixo disso = olho fechado
FRAMES_PARA_ALERTA = 20  # Frames consecutivos para disparar o alerta
contador_fechado = 0

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    altura, largura = frame.shape[:2]

    # O MediaPipe moderno exige imagens no formato mp.Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    resultado = detector.detect(mp_image)

    if resultado.face_landmarks:
        for rosto_landmarks in resultado.face_landmarks:

            # TODO 2: EAR atual (média dos dois olhos)
            ear_esq = calcular_ear(rosto_landmarks, OLHO_ESQ, largura, altura)
            ear_dir = calcular_ear(rosto_landmarks, OLHO_DIR, largura, altura)
            ear = (ear_esq + ear_dir) / 2.0

            # Desenha os pontos dos olhos para conferência visual
            for i in OLHO_ESQ + OLHO_DIR:
                lm = rosto_landmarks[i]
                cv2.circle(frame, (int(lm.x * largura), int(lm.y * altura)), 2, (0, 255, 0), -1)

            cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # TODO 3: máquina de estados / acumulador de quadros consecutivos
            if ear < LIMIAR_EAR:
                contador_fechado += 1
            else:
                contador_fechado = 0   # piscada normal: zera o contador

            if contador_fechado >= FRAMES_PARA_ALERTA:
                cv2.putText(frame, "ALERTA: SONOLENCIA!", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                cv2.rectangle(frame, (0, 0), (largura - 1, altura - 1), (0, 0, 255), 8)
                # Alerta sonoro (somente Windows). Descomente se quiser:
                # import winsound; winsound.Beep(1000, 200)
    else:
        contador_fechado = 0  # sem rosto detectado: não acumula

    cv2.imshow('Desafio 2: Alarme de Sonolencia', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
