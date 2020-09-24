import dlib
import cv2 as cv
import numpy as np
import os

def rect_to_bb(rect):
    y=rect.top()
    x=rect.left()
    w=rect.right()-x
    h=rect.bottom()-y
    
    return x,y,w,h
def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype=float)
    
    for i in range(68):
        
        coords[i]=(shape.part(i).x,shape.part(i).y)
        
    return coords

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

for i in os.listdir("images"):
    image=cv.imread(os.path.join('images',i))
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    rects=detector(gray)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape=predictor(gray,rect)
        coords=shape_to_np(shape)
        x,y,w,h=rect_to_bb(rect)
            
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for j in range(len(coords)):
            x,y=coords[j]
            center = (int(x), int(y))
            cv.circle(image,center,3,(0,0,255),-1)
            if j<67:
                x1,y1=coords[j+1]
                x1,y1=int(x1),int(y1)
                cv.line(image,center,(x1,y1),(0,0,255),1)
    cv.imshow("Output", image)
    key = cv.waitKey(1)
    if key == 27:
        break
        
    
    