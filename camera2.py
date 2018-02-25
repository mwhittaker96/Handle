mport numpy as np
import cv2


cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

while(True):
        #capture frame-by-frame
        ret, frame = cap.read()

        #make frame gray
        fgmask = fgbg.apply(frame)
        #display frame
        cv2.imshow('frame', fgmask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#when done
cap.release()
cv2.destroyAllWindows()
