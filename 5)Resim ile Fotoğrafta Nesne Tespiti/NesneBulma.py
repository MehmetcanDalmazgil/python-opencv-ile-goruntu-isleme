import cv2
import numpy as np


### Resmimizi alıyoruz ve grileştiriyoruz.
img_rgb=cv2.imread('ana_resim.jpg')
img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)


nesne=cv2.imread('bulunacak.jpg',0) # Bulacağımız nesneyi alıyoruz. (Gri olarak)

w,h=nesne.shape[::-1] # Nesnemizin yüksekliğini ve genişliğini alıyoruz.

res=cv2.matchTemplate(img_gray,nesne,cv2.TM_CCOEFF_NORMED) # Resmimizin gri tonuyla bu nesenyi eşleştirmeye çalışıyoruz.
threshold=0.80 # Eşik değer oluşturuyoruz. (%80 doğruluk payıyla bunu bulmaya çalış diyoruz. )

loc=np.where(res>threshold) # Bulunan nesneler içerisinde eşik değerden yüksek eşleşme gösteren değerleri loc değişkenine alıyoruz.

for n in zip(*loc[::-1]): # Belirlenen bu nesneler etrafında dikdörtgen çizdirmek için yükseklik ve genişlik değerlerini alıyoruz.
    cv2.rectangle(img_rgb,n,(n[0]+w,n[1]+h),(0,255,0),2) # Sütun,Satır sırasıyla değerler verilmektedir.
    cv2.putText(img_rgb,"Dugme",(n[0]+w,n[1]+h),cv2.QT_FONT_NORMAL,1, (255,255,255),1)

### Resmimizi ekrana bastırıyoruz.
cv2.imshow('CIKTI',img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
