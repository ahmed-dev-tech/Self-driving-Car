import cv2
import numpy as np
import utlis

if __name__ == '__main__':
    cap = cv2.VideoCapture('vid1.mp4')
    while True:
        success, img = cap.read() # GET THE IMAGE
        img = cv2.resize(img,(480,240)) # RESIZE
        cv2.imshow('Vid',img)
        cv2.waitKey(1)
-----------------------------------------------------------------------------
def getLaneCurve(img): #Lane.py
    imgThres = utlis.thresholding(img)
    cv2.imshow('Thres',imgThres)
    return None

###Thresholding: perform based on color so that we get the white pixels

def thresholding(img): # unction in utils.py
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([80, 0, 0])
    upperWhite = np.array([255, 160, 255])
    maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite)
    return maskedWhite

if __name__ == '__main__':
    cap = cv2.VideoCapture('vid1.mp4')
    while True:
        success, img = cap.read() # GET THE IMAGE
        img = cv2.resize(img,(480,240)) # RESIZE
        getLaneCurve(img)
        cv2.imshow('Vid',img)
        cv2.waitKey(1)
.........................................................................
import cv2
import numpy as np
 
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
 
 
def empty(a):
    pass
 
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)
 
cap = cv2.VideoCapture('vid1.mp4')
frameCounter = 0
 
while True:
    frameCounter +=1
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) ==frameCounter:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter=0
 
    _, img = cap.read()
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
    print(h_min)
 
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
 
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])
    cv2.imshow('Horizontal Stacking', hStack)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()

-------------------------------------------------------------------------
def warpImg (img,points,w,h): # util.spy
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp

def getLaneCurve(img): #Lane.py
    #Step 1
    imgThres = utlis.thresholding(img)
    #Step 2
    h,w,c = img.shape
    points = utils.valTrackbars()
    imgWrap = utils.wrapImg(img,points,w,h)
    cv2.imshow('Thres',imgThres)
    cv2.imshow('warp',imgWarp)
    return None

def nothing(a): #utils.py
    pass

def initializeTrackbars(intialTracbarVals,wT=480, hT=240): #utils.py
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)

def valTrackbars(wT=480, hT=240): #util.py
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points

if __name__ == '__main__':
    cap = cv2.VideoCapture('vid1.mp4')
    initialTrackBarVals = [100,100,100,100] #.... 102,80,20,214
    utils.initializeTrackbars(initialTrackBarVals)
    frameCounter =0
    while True:
        frameCounter +=1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) ==frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            frameCounter=0
        success, img = cap.read() # GET THE IMAGE
        img = cv2.resize(img,(480,240)) # RESIZE
        getLaneCurve(img)
        cv2.imshow('Vid',img)
        cv2.waitKey(1)
 ...............................................................
image will be wraped ac to threshold but we want to get the points

def drawPoints(img,points):
    for x in range(4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)
    return img

def getLaneCurve(img): #Lane.py
    imgCopy = img.copy()
    #Step 1
    imgThres = utlis.thresholding(img)
    #Step 2
    h,w,c = img.shape
    points = utils.valTrackbars()
    imgWrap = utils.wrapImg(img,points,w,h) # use imgThres for black and white
    imgWarpPoints = utlis.drawPoints(imgCopy, points)
    cv2.imshow('Thres',imgThres)
    cv2.imshow('warp',imgWarp)
    cv2.imshow('Warp Points',imgWarpPoints)
    return None

------------------------------------------------------------------------------

def getHistogram(img,minPer=0.1,display=False): #utils.py
 
    histValues = np.sum(img, axis=0)
    #print(histValues)
    maxValue = np.max(histValues)
    #print(maxvalue)
    minValue = minPer*maxValue
    indexArray = np.where(histValues >= minValue)
    basePoint = int(np.average(indexArray))
    #print(basePoint)
    if display:
        imgHist = np.zeros((imgHist.shape[0],img.shape[1],3),np.uint8)
       for x,intensity in enumerate(histValues):
          # print(intensity)
           cv2.line(img,(x,img.shape[0]),(x,img.shape[0]-intensity//255),(255,0,255),1)
           cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
       return basePoint,imgHist
    return basePoint

def getLaneCurve(img): #Lane.py
    imgCopy = img.copy()
    #Step 1
    imgThres = utlis.thresholding(img)
    #Step 2
    h,w,c = img.shape
    points = utils.valTrackbars()
    imgWrap = utils.wrapImg(img,points,w,h) # use imgThres for black and white
    imgWarpPoints = utlis.drawPoints(imgCopy, points)
   #Step 3
    basePoint,imgHist=utlis.getHistogram(imgWrap, display=True) #numpy array of summation of each columns

    cv2.imshow('Thres',imgThres)
    cv2.imshow('warp',imgWarp)
    cv2.imshow('Warp Points',imgWarpPoints)
    cv2.imshow('Histogram',imgHist)
    return None
------------------------------------------------------------------------------------
instead of summing the columns we have to sum the bottom part soi that we get the center of the path
curveList = []
avgVal = 10
def getHistogram(img,minPer=0.1,display=False,region = 1): #utils.py
    if region ==1:
      histValues = np.sum(img, axis=0)
  else :
      histValues = np.sum(img[img.shape[0]//region:,:], axis=0) 

    histValues = np.sum(img, axis=0)
    #print(histValues)
    maxValue = np.max(histValues)
    #print(maxvalue)
    minValue = minPer*maxValue
    indexArray = np.where(histValues >= minValue)
    basePoint = int(np.average(indexArray))
    #print(basePoint)
    if display:
        imgHist = np.zeros((imgHist.shape[0],img.shape[1],3),np.uint8)
       for x,intensity in enumerate(histValues):
          # print(intensity)
           cv2.line(img,(x,img.shape[0]),(x,img.shape[0]-intensity//255//region),(255,0,255),1)
           cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
       return basePoint,imgHist
    return basePoint

def getLaneCurve(img,display = 2): #Lane.py
    imgCopy = img.copy()
    imgResult = img.copy()
    #Step 1
    imgThres = utlis.thresholding(img)
    #Step 2
    hT,wT,c = img.shape
    points = utils.valTrackbars()
    imgWrap = utils.wrapImg(img,points,wT,hT) # use imgThres for black and white
    imgWarpPoints = utlis.drawPoints(imgCopy, points)
   #Step 3
    #basePoint,imgHist=utlis.getHistogram(imgWrap, display=True,minPer=0.5,region=4) #numpy array of summation of each columns
    #midPoint,imgHist=utlis.getHistogram(imgWrap, display=True,minPer=0.9) #numpy array of summation of each columns
    #print(basePoint - midvalue) #curve value
    middlePoint,imgHist=utlis.getHistogram(imgWrap, display=True,minPer=0.5,region=4) #numpy array of summation of each columns
    curveAveragePoint,imgHist=utlis.getHistogram(imgWrap, display=True,minPer=0.9) #numpy array of summation of each columns
    curveRaw = middlePoint - curveAveragePoint
   #Step 4 #Noise reduction by averaging (not values very high and low)
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))
    # Step 5
    if display != 0:
       imgInvWarp = utlis.warpImg(imgWarp, points, wT, hT,inv = True)
       imgInvWarp = cv2.cvtColor(imgInvWarp,cv2.COLOR_GRAY2BGR)
       imgInvWarp[0:hT//3,0:wT] = 0,0,0
       imgLaneColor = np.zeros_like(img)
       imgLaneColor[:] = 0, 255, 0
       imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
       imgResult = cv2.addWeighted(imgResult,1,imgLaneColor,1,0)
       midY = 450
       cv2.putText(imgResult,str(curve),(wT//2-80,85),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
       cv2.line(imgResult,(wT//2,midY),(wT//2+(curve*3),midY),(255,0,255),5)
       cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY-25), (wT // 2 + (curve * 3), midY+25), (0, 255, 0), 5)
       for x in range(-30, 30):
           w = wT // 20
           cv2.line(imgResult, (w * x + int(curve//50 ), midY-10),
                    (w * x + int(curve//50 ), midY+10), (0, 0, 255), 2)
       #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
       #cv2.putText(imgResult, 'FPS '+str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,50,50), 3);
   if display == 2:
       imgStacked = utlis.stackImages(0.7,([img,imgWarpPoints,imgWarp],
                                         [imgHist,imgLaneColor,imgResult]))
       cv2.imshow('ImageStack',imgStacked)
   elif display == 1:
       cv2.imshow('Resutlt',imgResult)

    cv2.imshow('Thres',imgThres)
    cv2.imshow('warp',imgWarp)
    cv2.imshow('Warp Points',imgWarpPoints)
    cv2.imshow('Histogram',imgHist)
    return curve

def warpImg (img,points,w,h, inv=False): # util.spy
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


if __name__ == '__main__':
    cap = cv2.VideoCapture('vid1.mp4')
    initialTrackBarVals = [100,100,100,100] #.... 102,80,20,214
    utils.initializeTrackbars(initialTrackBarVals)
    frameCounter =0
  
    while True:
        frameCounter +=1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) ==frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            frameCounter=0
        success, img = cap.read() # GET THE IMAGE
        img = cv2.resize(img,(480,240)) # RESIZE
        getLaneCurve(img)
     # baad mein chalni hain ye dono line depends on your display 2 for complete pipeline
        #curve = getLaneCurve(img,display=0,2)
        #print(curve)
        cv2.imshow('Vid',img)
        cv2.waitKey(1)
---------------------------------------------------------------------------------
def getLaneCurve(img,display = 2): #Lane.py
    imgCopy = img.copy()
    imgResult = img.copy()
    #Step 1
    imgThres = utlis.thresholding(img)
    #Step 2
    hT,wT,c = img.shape
    points = utils.valTrackbars()
    imgWrap = utils.wrapImg(img,points,wT,hT) # use imgThres for black and white
    imgWarpPoints = utlis.drawPoints(imgCopy, points)
   #Step 3
    #basePoint,imgHist=utlis.getHistogram(imgWrap, display=True,minPer=0.5,region=4) #numpy array of summation of each columns
    #midPoint,imgHist=utlis.getHistogram(imgWrap, display=True,minPer=0.9) #numpy array of summation of each columns
    #print(basePoint - midvalue) #curve value
    middlePoint,imgHist=utlis.getHistogram(imgWrap, display=True,minPer=0.5,region=4) #numpy array of summation of each columns
    curveAveragePoint,imgHist=utlis.getHistogram(imgWrap, display=True,minPer=0.9) #numpy array of summation of each columns
    curveRaw = middlePoint - curveAveragePoint
   #Step 4 #Noise reduction by averaging (not values very high and low)
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))
    # Step 5
    if display != 0:
       imgInvWarp = utlis.warpImg(imgWarp, points, wT, hT,inv = True)
       imgInvWarp = cv2.cvtColor(imgInvWarp,cv2.COLOR_GRAY2BGR)
       imgInvWarp[0:hT//3,0:wT] = 0,0,0
       imgLaneColor = np.zeros_like(img)
       imgLaneColor[:] = 0, 255, 0
       imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
       imgResult = cv2.addWeighted(imgResult,1,imgLaneColor,1,0)
       midY = 450
       cv2.putText(imgResult,str(curve),(wT//2-80,85),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
       cv2.line(imgResult,(wT//2,midY),(wT//2+(curve*3),midY),(255,0,255),5)
       cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY-25), (wT // 2 + (curve * 3), midY+25), (0, 255, 0), 5)
       for x in range(-30, 30):
           w = wT // 20
           cv2.line(imgResult, (w * x + int(curve//50 ), midY-10),
                    (w * x + int(curve//50 ), midY+10), (0, 0, 255), 2)
       #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
       #cv2.putText(imgResult, 'FPS '+str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,50,50), 3);
   if display == 2:
       imgStacked = utlis.stackImages(0.7,([img,imgWarpPoints,imgWarp],
                                         [imgHist,imgLaneColor,imgResult]))
       cv2.imshow('ImageStack',imgStacked)
   elif display == 1:
       cv2.imshow('Resutlt',imgResult)
   curve = curve/100
   if curve >1 : curve == 1
   if curve <-1: curve == -1
    cv2.imshow('Thres',imgThres)
    cv2.imshow('warp',imgWarp)
    cv2.imshow('Warp Points',imgWarpPoints)
    cv2.imshow('Histogram',imgHist)
    return curve
