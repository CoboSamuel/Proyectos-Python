import cv2
import os

# Cargar los detectores de rostros y sonrisas de OpenCV
detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector_sonrisas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Iniciar la cámara con DirectShow para Windows
camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("[+] Filtro de Realidad Aumentada Activo.")
print("[*] Pon la ventana de la cámara en primer plano y prueba tus expresiones...")

# Cargar las imágenes en memoria desde el inicio
img_feliz = cv2.imread("feliz.jpg")
img_triste = cv2.imread("triste.jpg")

# Verificar que las imágenes existan antes de correr el bucle
if img_feliz is None or img_triste is None:
    print("[Error] Asegúrate de tener 'feliz.jpg' y 'triste.jpg' en la misma carpeta.")
    camara.release()
    exit()

# Estado inicial: empezamos asumiendo seriedad
imagen_actual = img_triste

while True:
    ret, frame = camara.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = detector_rostros.detectMultiScale(gris, 1.3, 5)

    for (x, y, w, h) in rostros:
        # Cortar la región del rostro en gris para buscar la sonrisa
        region_rostro_gris = gris[y:y+h, x:x+w]
        sonrisas = detector_sonrisas.detectMultiScale(region_rostro_gris, 1.7, 22)

        # Cambiar la "máscara" actual dependiendo de la expresión
        if len(sonrisas) > 0:
            imagen_actual = img_feliz
        else:
            imagen_actual = img_triste

        try:
            # 1. Redimensionar la imagen (feliz/triste) para que tenga exactamente
            # el mismo ancho (w) y alto (h) que tu rostro detectado en ese instante.
            mascara_redimensionada = cv2.resize(imagen_actual, (w, h))

            # 2. Superponer la imagen encima de tu cara en el cuadro original (frame)
            frame[y:y+h, x:x+w] = mascara_redimensionada
        except Exception as e:
            # Por si ocurre algún error matemático al redimensionar en los bordes de la pantalla
            pass

    # Mostrar el resultado final con la máscara puesta
    cv2.putText(frame, "Presiona 'q' para salir", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Filtro Facial de Animo Tracking", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()
