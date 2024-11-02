import cv2
import re
import pytesseract
import logging
from utils import substituir_letras_por_numeros, gerar_possibilidades_mercosul
from typing import List, Tuple, Union

def encontrar_placa(string: str) -> Union[str, None]:
    """Procura uma placa no formato antigo (ABC1234) em uma string.

    Args:
        string (str): Texto onde procurar a placa.

    Returns:
        Union[str, None]: Placa encontrada ou None se não houver correspondência.
    """
    padrao = r'[A-Z]{3}\d{4}'
    placas_encontradas = re.findall(padrao, string)
    return placas_encontradas[0] if placas_encontradas else None

def encontrar_placa_mercosul(string: str) -> Union[str, None]:
    """Procura uma placa no formato Mercosul (ABC1D23) em uma string.

    Args:
        string (str): Texto onde procurar a placa.

    Returns:
        Union[str, None]: Placa Mercosul encontrada ou None se não houver correspondência.
    """
    padrao = r'[A-Z]{3}[0-9][0-9A-Z][0-9]{2}'
    placas_encontradas = re.findall(padrao, string)
    return placas_encontradas[0] if placas_encontradas else None

def aplicar_ocr(possiveis_placas: List[Tuple]) -> Tuple[Union[str, None], any, any]:
    """Aplica OCR nas imagens de possíveis placas para identificar o texto.

    Args:
        possiveis_placas (List[Tuple]): Lista de tuplas com imagens recortadas e processadas das possíveis placas.

    Returns:
        Tuple[Union[str, None], any, any]: Texto da placa detectada, imagem recortada, imagem processada.
    """
    for placa_recortada, placa_recortada_processada in possiveis_placas:
        logging.info("Aplicando OCR na imagem da placa")
        x, y, w, h = cv2.boundingRect(placa_recortada_processada)

        if h > 120:
            placa_recortada_processada = placa_recortada_processada[30:]
            placa_recortada_processada = placa_recortada_processada[:-10]

        # OCR em português
        resultado_tesseract_por = pytesseract.image_to_string(
            placa_recortada_processada, lang='por',
            config=r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6 --oem 3')
        placa_detectada_por = "".join(filter(str.isalnum, resultado_tesseract_por))

        # Verifica formato Mercosul e antigo
        placa_mercosul = encontrar_placa_mercosul(placa_detectada_por)
        if placa_mercosul:
            logging.info(f"Placa Mercosul detectada: {placa_mercosul}")
            return placa_mercosul, placa_recortada, placa_recortada_processada

        placa_antiga = encontrar_placa(placa_detectada_por)
        if placa_antiga:
            logging.info(f"Placa antiga detectada: {placa_antiga}")
            return placa_antiga, placa_recortada, placa_recortada_processada

        # OCR em inglês
        resultado_tesseract_eng = pytesseract.image_to_string(
            placa_recortada_processada, lang='eng',
            config=r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6 --oem 3')
        placa_detectada_eng = "".join(filter(str.isalnum, resultado_tesseract_eng))

        # Verifica formato Mercosul e antigo em inglês
        placa_mercosul = encontrar_placa_mercosul(placa_detectada_eng)
        if placa_mercosul:
            logging.info(f"Placa Mercosul detectada no OCR inglês: {placa_mercosul}")
            return placa_mercosul, placa_recortada, placa_recortada_processada

        placa_antiga = encontrar_placa(placa_detectada_eng)
        if placa_antiga:
            logging.info(f"Placa antiga detectada no OCR inglês: {placa_antiga}")
            return placa_antiga, placa_recortada, placa_recortada_processada

        # Geração de possibilidades de substituição de letras por números
        ultimos_4_caracteres = placa_detectada_por[-4:]
        possibilidades = substituir_letras_por_numeros(ultimos_4_caracteres)
        result = ""
        for possibilidade in possibilidades:
            result += placa_detectada_por[:3] + possibilidade + "\n"

        result += "\nMercosul:\n"
        possibilidades_mercosul = gerar_possibilidades_mercosul(ultimos_4_caracteres)
        for possibilidade in possibilidades_mercosul:
            result += placa_detectada_por[:3] + possibilidade + "\n"

        logging.info("Possíveis substituições de letras por números geradas para placa.")
        return result, placa_recortada, placa_recortada_processada

    logging.warning("Nenhuma placa foi identificada nas imagens fornecidas.")
    return None, None, None
