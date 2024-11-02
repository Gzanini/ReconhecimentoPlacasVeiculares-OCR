import cv2
import logging
from typing import List, Tuple

def processar_contornos(imagem_original, imagem_processada) -> List[Tuple]:
    """Processa os contornos na imagem para identificar possíveis placas.

    Args:
        imagem_original: Imagem original carregada.
        imagem_processada: Imagem após processamento.

    Returns:
        List[Tuple]: Lista de possíveis placas detectadas.
    """
    logging.info("Processando contornos da imagem")
    contornos, _ = cv2.findContours(imagem_processada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    possiveis_placas = []

    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        area = cv2.contourArea(contorno)
        x, y, w, h = cv2.boundingRect(contorno)

        if h > w or h < (w * 0.2) or area < 10000 or area > 70000:
            continue

        if len(aprox) >= 4 and len(aprox) < 10:
            cv2.drawContours(imagem_original, [aprox], -1, (0, 255, 0), 2)
            imagem_recortada = imagem_original[y:y + h, x:x + w]
            imagem_recortada_cinza = cv2.cvtColor(imagem_recortada, cv2.COLOR_BGR2GRAY)

            _, imagem_recortada_limiarizada = cv2.threshold(
                imagem_recortada_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            imagem_recortada_processada = cv2.morphologyEx(
                imagem_recortada_limiarizada, cv2.MORPH_CLOSE, kernel)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            imagem_recortada_processada = cv2.morphologyEx(
                imagem_recortada_processada, cv2.MORPH_OPEN, kernel)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            imagem_recortada_processada = cv2.dilate(imagem_recortada_processada, kernel, iterations=1)
            imagem_recortada_processada = cv2.erode(imagem_recortada_processada, kernel, iterations=1)

            possiveis_placas.append((imagem_recortada, imagem_recortada_processada))

    logging.info(f"Encontrou {len(possiveis_placas)} possíveis placas")
    return possiveis_placas
