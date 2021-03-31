import cv2
import numpy as np

### Kameramızı çalıştırdık.
kamera= cv2.VideoCapture(0)

while(1):
    ### Çalışan kameramızı okuyoruz.
    ret, frame=kamera.read()

    ### Okunan videoyu filtreleme için hsv formatına çeviriyoruz.
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    ### Maskeleme(filtreleme) için düşük ve yüksek renk değerlerini seçiyorıuz. Şu an mavi rengi filtreliyoruz. Yani mavi rengi ayırt etmek istiyoruz.
    dusuk_mavi=np.array([100,60,60]) # dusuk_kırmızı [150,30,30] # dusuk_beyaz [0,0,140]
    ust_mavi=np.array([140,255,255]) # ust_kırmızı [190,255,255] # ust_beyaz [256,60,256]

    ### Videoda mavi renki olan kısımları maskeliyoruz yani mavi olan kısımları beyaz geri kalan kısımları siyah yapıyoruz.
    mask=cv2.inRange(hsv,dusuk_mavi,ust_mavi)

    ### Ardından normal video ile maskelenmiş frame'leri birleştirerek aslında mavi olan ama maskeleme ile beyaz renge dönüşmüş kısımları tekrar maviye döndürüyoruz.
    son_resim=cv2.bitwise_and(frame,frame,mask=mask)

    ### Gürültüyü artırma ve azaltma işlemleri için blurlaştırma kullanıyoruz.
    kernel = np.ones((5,5),np.uint8)
    ### erosion gürültüyü silmektedir.
    erosion = cv2.erode(mask,kernel,iterations=1)
    ### dilation gürültüyü artırmaktadır.
    dilation = cv2.dilate(mask,kernel,iterations=1)
    ### opening filtreleme içerisinde uymayan kısımları(gürültüleri) iyice belirginleştirmektedir. (Siyah yapmaktadır)
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    ### closing filtrleme içerisindeki gürültüleri kapatmaktadır.(Beyaz yapmaktadır)
    closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

    ### Tüm frame'ler arasındaki farkı anlamak için görüntülüyoruz.
    cv2.imshow('GIRDI',frame)
    cv2.imshow('MASKELEME', mask)
    cv2.imshow('CIKTI', son_resim)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
kamera.release()
cv2.destroyAllWindows()