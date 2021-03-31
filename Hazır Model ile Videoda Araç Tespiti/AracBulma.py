import cv2
cascade_src='haarcascade_car.xml'
video_src='otoban.avi'

cap=cv2.VideoCapture(video_src) #Kod içerisine videomuzu dahil ediyoruz.
cars_cascade=cv2.CascadeClassifier(cascade_src) #Arac bulma için hazır haaracascade arac modulünü kullanıyoruz.

while True:
    _, img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Her frame'i grileştiriyoruz.
    cv2.imshow('gri_hal', gray)
    araclar=cars_cascade.detectMultiScale(gray,1.1,1) #Her frame'de haarcascade ile bulunana araçların konum bilgilerini bir değişkene kaydediyoruz.

    #Tespit edilen araç bilgilerini kullanarak nesne etraflarına dikdörtgen çizdiriyoruz.
    for(x,y,w,h) in araclar:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2) #Dikdörtgenimizi çizdiriyoruz.
        cv2.putText(img, "arac", (x,y-3), cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255), 1)
    cv2.imshow('CIKTI',img)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()