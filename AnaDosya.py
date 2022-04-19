from NesneTanimaModulu import *

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# cap.set(10,70)

while True:
    success, img = cap.read()
    sonuc,objectInfo = nesneleriBul(img, 0.45, objects=["portakal", "makas"])

    cv2.imshow("Output", img)
    cv2.waitKey(1)