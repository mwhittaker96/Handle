import os
import numpy as np
import cv2
import sys

black_pixel = [0,0,0]
white_pixel = [255,255,255]


""""
def paintpixel(pixel):

    if pixel[0]<=50 and pixel[1]>23 and pixel[1]<68:
        return black_pixel
    else:
        return white_pixel

def buildframe(hsvframe):
    i = 0
    z = 0
    for row in hsvframe:
        for pixel in row:
            hsvframe[i][z] = paintpixel(row[z])
            z += 1
        z = 0
        i += 1

    # print(newframe)
    return hsvframe
"""

#make folder to save training data
folder = input("'Letter: '")
if os.path.exists('./'+str(folder)) == False:
    os.mkdir('./'+str(folder))



#video streaming and processing
cap = cv2.VideoCapture(0)
hsvSkinThreshold = [0,48,80]
rgbSkinThreshold = [20, 255, 255]
# cap.set(3,320) # Width
#cap.set(4, 240) # Height
count = 0
name = 0

max_frames = 800
while(True):
        #capture frame-by-frame
        ret, frame = cap.read()
        blur = cv2.blur(frame, (3,3))
        hsvframe = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsvframe, np.array(hsvSkinThreshold), np.array(rgbSkinThreshold))

        # guas = cv2.GaussianBlur(blur, (5,5), 0)
        #edgeBlur = cv2.bilateralFilter(blur, 9,75,75)

        #make frame hsv
    #
        #print(hsvframe)q
        #kapow = buildframe(hsvframe)
        #print(kapow)
        # break;




        #display frame

        cv2.imshow('frame', mask2)


        if count == 5:
            #capture frame
            print(name)
            count = 0;
            cv2.imwrite(str(folder)+'/frame%d.jpg' % name, mask2)     # save frame as JPEG file
            name += 1
            if name == max_frames:
                sys.exit(0)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#when done
cap.release()
cv2.destroyAllWindows()
