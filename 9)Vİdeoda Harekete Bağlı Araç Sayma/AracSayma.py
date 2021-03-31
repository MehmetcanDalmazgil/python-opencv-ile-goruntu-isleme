import cv2
backsub = cv2.createBackgroundSubtractorMOG2()
capture = cv2.VideoCapture("arac_video.avi")
i = 0
minArea=2600 #Herhangi bir aracın momenti 2600 üzernde çıkmaktadır.

#Videoda hareketli nesnelerin ağırlık merkezinin, çizilen doğru merkezleri üzerinden geçip geçmemesine bağlı olarak araç sayma işlemi gerçekleştirilmiştir.
while True:
    ret, frame = capture.read()

    fgmask = backsub.apply(frame, None, 0.02)
    erode=cv2.erode(fgmask,None,iterations=4)
    moments=cv2.moments(erode,True)

    #Harketli nesnelerin ağırlık merkezine bağlı olarak araç sayısını artıracağız.
    area=moments['m00']
    # yatay ust
    cv2.line(frame,(40,0),(40,176),(255,0,0),2)
    cv2.line(frame, (55, 0), (55, 176), (255, 0, 0), 2)
    # diket ust
    cv2.line(frame,(0,50),(320,50),(255,0,0),2)
    cv2.line(frame, (0, 65), (320, 65), (255, 0, 0), 2)
    # yatay alt
    cv2.line(frame, (100, 0), (100, 176), (0, 255, 255), 2)
    cv2.line(frame, (115, 0), (115, 176), (0, 255, 255), 2)
    # dikey alt
    cv2.line(frame, (0, 105), (320, 105), (0, 255, 255), 2)
    cv2.line(frame, (0, 130), (320, 130), (0, 255, 255), 2)

    if moments['m00'] >= minArea:
        x = int(moments['m10'] / moments['m00']) # X konumunu hesaplıyoruz.
        y = int(moments['m01'] / moments['m00']) # Y konumunu hesaplıyoruz.

        print("Momentum :" + str(moments['m00']) + "x :" + str(x) + " y : " + str(y))

        if x > 40 and x < 55 and y > 50 and y < 65:
            i = i + 1
            print(i)
            #print("ust" + str(i))
        elif x > 105 and x < 115 and y > 105 and y < 130:
            i = i + 1
            print(i)
            #print("alt" + str(i))
        print("---------------------------------------------------------------------")

        cv2.putText(frame, 'Sayi: %r' % i, (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("CIKTI", frame)
        cv2.imshow("MASKELEME", fgmask)

    key = cv2.waitKey(25)
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
