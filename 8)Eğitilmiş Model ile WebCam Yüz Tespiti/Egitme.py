import cv2,os
import numpy as np
from PIL import Image # Image tanımda kullanılan kütüphane

recognizer = cv2.face.LBPHFaceRecognizer_create() # Eğitim metodunu tanımlıyoruz.
cascadePath = "face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
path = 'YuzVerileri'

def get_images_and_labels(path):
     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
     images = []
     labels = []
     for image_path in image_paths:
         image_pil = Image.open(image_path).convert('L') # Resimlerimizi alıyoruz.
         image = np.array(image_pil, 'uint8')
         nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
         print(nbr)
         faces = faceCascade.detectMultiScale(image) # Resimdeki yüzü tespit ediyoruz.
         for (x, y, w, h) in faces:
             images.append(image[y: y + h, x: x + w])
             labels.append(nbr)
             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
             cv2.waitKey(10)
     return images,labels




images, labels = get_images_and_labels(path)
cv2.imshow('test',images[0])
cv2.waitKey(1)

# Çekilen yüz fotoğrafları içerisinde yüzleri tespit ederek, tespit edilen yüzlerin konumları ile resimler eğitim metoduna verilerek eğitilmiş bir dosya oluşturuyoruz.
recognizer.train(images, np.array(labels))
recognizer.write('training/trainer.yml')
cv2.destroyAllWindows()

