import cv2
import logging

def processar_imagem(imagem_placa) -> any:
    """Processa a imagem da placa para facilitar a identificação de caracteres.

    Args:
        imagem_placa: Imagem da placa original.

    Returns:
        Imagem após processamento para destacar os caracteres.
    """
    logging.info("Convertendo imagem para tons de cinza e aplicando filtros")
    imagem_cinza = cv2.cvtColor(imagem_placa, cv2.COLOR_BGR2GRAY)
    imagem_cinza = cv2.bilateralFilter(imagem_cinza, 9, 75, 75)
    imagem_limiarizada = cv2.adaptiveThreshold(
        imagem_cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return imagem_limiarizada
