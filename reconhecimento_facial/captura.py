from cv2 import cv2 
import numpy as np

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier("haarcascade_eye.xml") # Melhoria da eficiência da algoritmo, detecção de olho para não capturar faces invisíveis.

camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostras = 25
id = input('Digite seu identificador: ')
largura, altura = 220, 220
print("Capturando as faces...")

while True:
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150,150))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = imagem[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho) # A Detecção dos olhos é para melhorar a eficiência do algoritmo.
        for (o_x, o_y, o_l, o_a) in olhosDetectados:
            cv2.rectangle(regiao, (o_x, o_y), (o_x + o_l, o_y + o_a), (0, 255, 0), 2)
        
            if cv2.waitKey(1) & 0xFF == ord('p'):
                if np.average(imagemCinza) > 110:  # "Luminosidade que o OPENCV recomenda para captura de faces, ou seja, o ambiente tem que estar bem claro"
                    imagemFace  = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                    cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                    print("[foto " + str(amostra) + " capturada com sucesso]")
                    amostra += 1
                else:
                   print("Luminosidade do ambiante é insuficiente, ascenda uma luz!!")    

    cv2.imshow("Captura de Face", imagem)
    cv2.waitKey(1)  # Necessário para dar mais tempo para que a imagem seja processada e copiada para a tela durante a execução.
    if (amostra >= numeroAmostras):
        break

print("Faces capturadas com sucesso.")
camera.release()
cv2.destroyAllWindows()    