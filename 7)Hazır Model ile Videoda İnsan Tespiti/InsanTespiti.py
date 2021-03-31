import cv2
import numpy as np

cap=cv2.VideoCapture('video.avi')
human_cascade=cv2.CascadeClassifier('haarcascade_fullbody.xml') # Algoritmamızı dahil ediyoruz.

while True:
    _,frame=cap.read()
    griton=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # Videoyu grileştiriyoruz.
    insan=human_cascade.detectMultiScale(griton,1.1,4) # Cascade ile insan tespiti yapıyoruz. # Scalama ve örnekleme sayısını değiştirdiğimizde tespit edilen nesneler değişmektedir.

    # Tespit edilen insanlar çevresinde dikdörtgen çizdiriyoruz.
    for (x,y,w,h) in insan:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),3)
    cv2.imshow('CIKTI',frame)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()