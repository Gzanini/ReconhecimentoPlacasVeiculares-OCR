import os
import cv2
import pytesseract
from aplicar_ocr import aplicar_ocr
from processar_contornos import processar_contornos
from processar_imagem import processar_imagem
from utils import exibir_resultado
import logging

# Configurar o caminho do executável Tesseract (se necessário)
os.environ["TESSDATA_PREFIX"] = "C:/Program Files/Tesseract-OCR/tessdata/"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configurar logging
logging.basicConfig(level=logging.INFO)

def detectar_placa(imagem_path: str) -> None:
    """Processa a imagem para detectar a placa do veículo.

    Args:
        imagem_path (str): Caminho para o arquivo de imagem.
    """
    logging.info(f"Processando imagem: {imagem_path}")
    imagem_original = cv2.imread(imagem_path)
    imagem_processada = processar_imagem(imagem_original)
    possiveis_placas = processar_contornos(imagem_original, imagem_processada)

    if len(possiveis_placas) == 0:
        logging.warning("Nenhuma placa foi encontrada na imagem.")
        return

    placa_detectada, placa_recortada, placa_recortada_processada = aplicar_ocr(possiveis_placas)
    exibir_resultado(imagem_original, imagem_processada, placa_recortada, placa_recortada_processada, placa_detectada)

if __name__ == "__main__":
    pasta_imagens = "images"
    lista_imagens = os.listdir(pasta_imagens)
    logging.info(f"Iniciando processamento de imagens na pasta: {pasta_imagens}")

    for imagem_file in lista_imagens:
        if imagem_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            imagem_path = os.path.join(pasta_imagens, imagem_file)
            detectar_placa(imagem_path)
