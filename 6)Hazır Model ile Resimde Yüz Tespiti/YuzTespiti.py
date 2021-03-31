import cv2
import numpy as np

img= cv2.imread('kalabalik.jpg') # Resmimizi alıyoruz.

### Algoritmanın daha eğitilmesi gerekmektedir.
yuz_casc=cv2.CascadeClassifier('haarcascade_face.xml') # Haarcascade eğitilmiş dosyamızı alıyoruz.

griton=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Resmimizi grilendiriyoruz.
yuzler=yuz_casc.detectMultiScale(griton,1.1,2) # detectMultiScale metoduyla yüzleri  buluyoruz. Scalafactor ile yüzleri yüzde 10 büyütüyoruz(1.1). 4 seferde teyit alıypruz.

# Bulunan yüzleri dikdörtgen içerisine alıyoruz.
for(x,y,w,h) in yuzler:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

cv2.imshow('CIKTI',img)
cv2.waitKey(0)
cv2.destroyAllWindows()