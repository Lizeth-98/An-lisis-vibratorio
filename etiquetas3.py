import cv2
import pytesseract
import re
 
# Configura la ruta de Tesseract OCR si es necesario
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
 
# Verifica si la imagen es legible
def check_legibility(image):
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var > 100  # Ajustar el umbral deacuerdo a la necesario
 
# Extrae texto del número de serie usando OCR
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary, config='--psm 6')
    return text.strip()
 
# Valida el formato del número de serie
def validate_serial_number(serial_number):
    #definir patron 
    pattern = r'^SN-\d{4}-[A-Z]{2}$'  # ejemplo simulado SN-1234-AB, falta para que pueda detectar todas las series y no solo una
    return bool(re.match(pattern, serial_number))
 
# Sistema de vision en tiempo real
def real_time_serial_validation():
    cap = cv2.VideoCapture(0) 
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el video.")
            break
 
        # legibilidad
        if not check_legibility(frame):
            result = "Imagen borrosa. Intente ajustar la cámara."
        else:
            # Extraer y validar num de serie
            serial_text = extract_text(frame)
            if not serial_text:
                result = "No se pudo leer el número de serie."
            elif not validate_serial_number(serial_text):
                result = f"Número de serie inválido: {serial_text}"
            else:
                result = f"Número de serie válido: {serial_text}"
 
        # Mostrar el resultado en el video
        cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Validación de Número de Serie", frame)
 
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    real_time_serial_validation()
