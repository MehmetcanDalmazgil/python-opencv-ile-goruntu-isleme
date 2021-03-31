import cv2
import numpy as np
img1 = cv2.imread('messi.jpg')
img2 = cv2.imread('logo.jpg')

#Logo'nun satır ve sütun değerlerine göre img1'in sol üst resim parçasını alıyoruz.
satir,sutun,kenar = img2.shape
roi = img1[0:satir,0:sutun]
#cv2.imshow('roi',roi)

#Logo'yu gri hale getiriyoruz.
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#cv2.imshow('img2gray',img2gray)

# Logo'daki siyah kısımlar haricindeki kısımları beyaza çeviriyoruz.
ret, mask = cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY)
#cv2.imshow('mask',mask)

# Gürültüyü silebilmek için görüntüye blurlaştırma uyguluyoruz.
kernel = np.ones((3,2),np.uint8)
# erosion gürültüyü silmektedir.
mask_erosion = cv2.erode(mask,kernel,iterations=1)
#cv2.imshow('mask',erosion)

#Logo'nun arka planı siyah durumda. Biz arka planın roi ile kaplı olmasını istediğimiz için ters maskeleme işlemi uyguluyoruz.
mask_ters = cv2.bitwise_not(mask_erosion)
#cv2.imshow('mask_ters',mask_ters)

# Arka planı düzenlemek için mask_ters ile messi'nin bulunduğu arka planı birleştiriyoruz.
sonuc_arkaplan = cv2.bitwise_and(roi,roi,mask = mask_ters)
#cv2.imshow('sonuc_arkaplan',sonuc_arkaplan)


"""
# Ön planı düzenlemek için logo ile mask'ı birleştiriyoruz.
sonuc_onplan = cv2.bitwise_and(img2,img2,mask = mask_erosion)
cv2.imshow('sonuc_onplan',sonuc_onplan)
"""

#Eklemek istediğimiz logo'nun arka plan ve ön planını birleştiriyoruz..
eklenecek_resim =  cv2.add(sonuc_arkaplan,img2) # img2 yerine yukarıda tanımlanan sonuc_onplan'da kullanılabilir.
#cv2.imshow('sonuc_onplan',eklenecek_resim)

#Son olarakta messi resminin sol üst kısmına oluşturuğumuz resmi ekliyoruz.
img1[0:satir,0:sutun] = eklenecek_resim
cv2.imshow('CIKTI',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

