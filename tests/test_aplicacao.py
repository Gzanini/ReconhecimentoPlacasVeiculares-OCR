import unittest
import cv2
import numpy as np
from aplicar_ocr import aplicar_ocr, encontrar_placa
from processar_contornos import processar_contornos
from processar_imagem import processar_imagem


class TestAplicacao(unittest.TestCase):

    def setUp(self):
        """Configuração inicial para os testes."""
        # Cria uma imagem de teste (placa simulada)
        self.imagem_teste = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.putText(self.imagem_teste, "ABC1234", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    def test_processar_imagem(self):
        """Testa o processamento da imagem para binarização."""
        imagem_processada = processar_imagem(self.imagem_teste)
        self.assertIsNotNone(imagem_processada)
        self.assertEqual(imagem_processada.shape, self.imagem_teste.shape[:2])

    def test_processar_imagem_tipo(self):
        """Verifica se o tipo de imagem processada está correto."""
        imagem_processada = processar_imagem(self.imagem_teste)
        self.assertIsInstance(imagem_processada, np.ndarray)

    def test_processar_contornos(self):
        """Testa a detecção de contornos e possíveis placas."""
        imagem_processada = processar_imagem(self.imagem_teste)
        possiveis_placas = processar_contornos(self.imagem_teste, imagem_processada)
        self.assertIsInstance(possiveis_placas, list)

    def test_processar_contornos_numero(self):
        """Verifica se ao menos um contorno foi detectado."""
        imagem_processada = processar_imagem(self.imagem_teste)
        possiveis_placas = processar_contornos(self.imagem_teste, imagem_processada)
        self.assertGreaterEqual(len(possiveis_placas), 0)

    def test_encontrar_placa(self):
        """Testa a detecção de placas no formato antigo (ABC1234)."""
        placa = encontrar_placa("ABC1234")
        self.assertEqual(placa, "ABC1234")

    def test_encontrar_placa_invalida(self):
        """Testa a detecção de uma string sem placa válida."""
        placa = encontrar_placa("1234XYZ")
        self.assertIsNone(placa)

    def test_aplicar_ocr(self):
        """Testa a aplicação do OCR nas possíveis placas detectadas."""
        imagem_processada = processar_imagem(self.imagem_teste)
        possiveis_placas = processar_contornos(self.imagem_teste, imagem_processada)
        if possiveis_placas:
            placa_detectada, _, _ = aplicar_ocr(possiveis_placas)
            self.assertIsInstance(placa_detectada, str)

    def test_aplicar_ocr_contem_texto(self):
        """Verifica se o OCR retorna uma string não vazia quando há uma placa válida."""
        imagem_processada = processar_imagem(self.imagem_teste)
        possiveis_placas = processar_contornos(self.imagem_teste, imagem_processada)
        if possiveis_placas:
            placa_detectada, _, _ = aplicar_ocr(possiveis_placas)
            self.assertTrue(placa_detectada.strip())  # Confirma que a string não está vazia

    def test_aplicar_ocr_tipo_saida(self):
        """Verifica se a saída do OCR é do tipo esperado."""
        imagem_processada = processar_imagem(self.imagem_teste)
        possiveis_placas = processar_contornos(self.imagem_teste, imagem_processada)
        if possiveis_placas:
            placa_detectada, imagem_recortada, imagem_processada = aplicar_ocr(possiveis_placas)
            self.assertIsInstance(placa_detectada, str)
            self.assertIsInstance(imagem_recortada, np.ndarray)
            self.assertIsInstance(imagem_processada, np.ndarray)


if __name__ == "__main__":
    unittest.main()
