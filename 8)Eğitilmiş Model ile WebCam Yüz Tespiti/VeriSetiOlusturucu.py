import cv2
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('face.xml') # Eğitilmiş algoritmamızı dahil ettik.
i=0

kisi_id=input('ID numarası giriniz')
while True:
    _, img =cam.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Videomuzu grileştiriyoruz.
    # Scala faktör ile minimum komuşu sayısıda tespit için yeterli olmaktadır.
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE) # Parametre ifadelerini veriyoruz.
    # En az 5 kere yüz olduğunu tespit ederse kaydedecektir.

    for(x,y,w,h) in faces:
        i=i+1 # Yüz tespiti için kaç tane fotoğraf çekilmektedir.
        cv2.imwrite("yuzverileri/face-" + kisi_id + '.' + str(i) + ".jpg", gray[y:y + h , x :x + w]) # Satır, Sütun
        cv2.rectangle(img, (x , y), (x + w, y + h), (225, 0, 0), 2) # Sütun, Satır
        cv2.imshow('resim', img[y :y + h, x :x + w])
        cv2.waitKey(100)
    if i>20:
        cam.release()
        cv2.destroyAllWindows()
        break

