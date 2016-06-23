import cv2
import numpy as np

def threshold(img):
    ret,thresh = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
    return thresh

def contour(img):
    im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def rotate(img1, scaleFactor = 1, degreesCCW = 30):
    (oldY,oldX) = img1.shape #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.
    newX,newY = oldX*scaleFactor,oldY*scaleFactor
    r = np.deg2rad(degreesCCW)
    newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx
    M[1,2] += ty
    rotatedImg = cv2.warpAffine(img1, M, dsize=(int(newX),int(newY)))
    return rotatedImg

def rotinv(cnt,img):
    ellipse = cv2.fitEllipse(cnt)
    angle = ellipse[2]
    if(90-angle>-45):
        rot_image = rotate(img,1,90-angle)
    else:
        rot_image = rotate(img,1,90-angle)
    return rot_image

def cropit(img,cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    return crop

def largestcont(contours):
    cnt = contours[0]

    for cont in contours:
        if(cv2.contourArea(cont) > cv2.contourArea(cnt)):
            cnt = cont
    return cnt

def extractBorder(img):
    return cv2.Canny(img,100,200)

def preprocess(img):
    thresh = threshold(img)
    contours = contour(thresh)
    thresh = threshold(img)
    cnt = largestcont(contours)
    crop = cropit(thresh,cnt)
    rot_image = rotinv(cnt,crop)
    border = extractBorder(rot_image)
    resized_image = cv2.resize(border, (512, 512))
    return resized_image
