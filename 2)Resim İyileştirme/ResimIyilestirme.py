import cv2
import numpy as np

img=cv2.imread('sayfa.jpg')
ret, threshold=cv2.threshold(img,12,255,cv2.THRESH_BINARY) # thresholding uyguluyoruz. Yani eşik değer altındaki kısımları siyah yapıyoruz.

griton=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Resmi grileştiriyoruz.

ret, thresholdgri=cv2.threshold(griton,12,255,cv2.THRESH_BINARY) # Grileştirilen resme thresholding uygulanıyor.

gaus=cv2.adaptiveThreshold(griton,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1) # Grileştirilen resme gaussian thresholding uyguluyoruz.

ret,otsu=cv2.threshold(griton,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Grileştirilen resme Otsu thresholding uyguluyoruz.

# En optimum thresholding resmi grileştirip gaussian thresholding değerini uygulamaktır. Durumdan duruma değişiyor thresholding yöntemlerinin uygun kullanımı.
cv2.imshow('GIRDI',img)
#cv2.imshow('thresholding',thresholdgri)
#cv2.imshow('otsu_thresholding',otsu)
cv2.imshow('CIKTI',gaus) # gaussian thresholding

cv2.waitKey(0)
cv2.destroyAllWindows()
