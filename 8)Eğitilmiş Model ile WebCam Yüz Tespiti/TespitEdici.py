import cv2


recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training/trainer.yml') # Bu eğitim dosyasını alıyoruz
cascadePath = "face.xml" # Artı olarak face cascade dosyamızıda alıyoruz.
faceCascade = cv2.CascadeClassifier(cascadePath)
path = 'yuzverileri' # Resimleri alıyoruz.

cam = cv2.VideoCapture(0)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) # WebCam'den görüntüyü alıp griye çeviriyoruz.
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE) #Cascade ile yüzleri tespit ettik.
    for(x,y,w,h) in faces: # Tespit edilen yüzleri kullanarak işlemlerimizi gerçekleştiriyoruz.
        tahminEdilenKisi, conf = recognizer.predict(gray[y:y + h, x:x + w]) # Tespit edilen yüzler ile trainer dosyasındaki yüzler ile karşılaştırıyoruz.
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2) # Dikdörtgen çizdiriyoruz.
        if(tahminEdilenKisi==1):
             tahminEdilenKisi= 'Mehmetcan'
        elif (tahminEdilenKisi == 2):
            tahminEdilenKisi = 'Aziz Sancar'
        else:
            tahminEdilenKisi= 'Bilinmeyen kişi'

        # Video üzerinden alınan frame'ler üzerinede gerekli adlandırmayı yapıyoruz.
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255, 255, 255)
        cv2.putText(im, str(tahminEdilenKisi), (x, y + h), fontFace, fontScale, fontColor)
        cv2.imshow('CIKTI',im)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()









