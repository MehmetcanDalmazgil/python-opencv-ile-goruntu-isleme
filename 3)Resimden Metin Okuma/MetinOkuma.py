import cv2
import numpy as np
from PIL import Image
import pytesseract #Metin okuma için gerekli olan kütüphane

pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

kaynak_yolu=""
def metin_oku(img_yolu):

    img=cv2.imread(img_yolu)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel=np.ones((1,1),np.uint8)
    img=cv2.erode(img,kernel,iterations=1)
    cv2.imshow('GIRDI', img)
    img = cv2.dilate(img, kernel, iterations=1)

    img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
    cv2.imwrite(kaynak_yolu+'gurultusuz.png',img)

    #Gürültüsüz resmin içerisinden pytesseract.image_to_string metodu ile metin ifadesini alıyoruz.
    sonuc=pytesseract.image_to_string(Image.open(kaynak_yolu+'gurultusuz.png'),lang='tur')
    return sonuc

print("---------------------------------")
print("metin okuma")
print("---------------------------------")
print(metin_oku('yazi.png'))

print("---------------------------------")
print("tamamlandı")
print("---------------------------------")

cv2.waitKey(0)
cv2.destroyAllWindows()


