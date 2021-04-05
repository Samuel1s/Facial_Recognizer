from cv2 import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemById():
    caminhos = [os.path.join("fotos", f) for f in os.listdir("fotos")]
    faces = []
    ids = []

    for caminhoImagem in caminhos:
        read_image = cv2.imread(caminhoImagem)
        
        if read_image is not None:  # Para evitar erro em cvtColor.
            imagemFace = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
            faces.append(imagemFace)   
        
        find_id = os.path.split(caminhoImagem)[-1].split('.')[1] 

        if find_id != 'DS_Store': # A existência do DS_Store dentro de fotos pode prejudicar na conversão de string para int.
            id = int(find_id)
            ids.append(id)  

    return np.array(ids), faces

ids, faces = getImagemById()

print("Treinando... ")
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')


print("Treinamento Realizado")

