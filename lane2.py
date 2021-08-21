import cv2
import numpy as np
import utils
curveList = []
avgVal = 10

'''def getLaneCurve(img,display=True): #Lane.py
    imgCopy = img.copy()
    #imgResult = img.copy()
    #Step 1
    imgThres = utils.thresholding(img)
    #Step 2
    hT,wT,c = img.shape
    points = utils.valTrackbars()
    imgWrap = utils.wrapImg(img,points,wT,hT) # use imgThres for black and white
    imgWarpPoints = utils.drawPoints(imgCopy, points)
   #Step 3
    #basePoint,imgHist=utlis.getHistogram(imgWrap, display=True,minPer=0.5,region=4) #numpy array of summation of each columns
    #midPoint,imgHist=utlis.getHistogram(imgWrap, display=True,minPer=0.9) #numpy array of summation of each columns
    #print(basePoint - midvalue) #curve value
    basePoint,imgHist=utils.getHistogram(imgWrap,minPer= 0.5,display=True,region=4) #numpy array of summation of each columns
    #curveAveragePoint,imgHist=utils.getHistogram(imgWrap, display=True,minPer=0.9) #numpy array of summation of each columns
    #curveRaw = midPoint - curveAveragePoint
    #print(curveRaw)
   #Step 4 #Noise reduction by averaging (not values very high and low)
    cv2.imshow('Thres',imgThres)
    cv2.imshow('warp',imgWrap)
    cv2.imshow('Warp Points',imgWarpPoints)
    cv2.imshow('Histogram',imgHist)
    return None'''
def getLaneCurve(img,display = 2): #Lane.py
    imgCopy = img.copy()
    imgResult = img.copy()
    #Step 1
    imgThres = utils.thresholding(img)
    #Step 2
    hT,wT,c = img.shape
    points = utils.valTrackbars()
    imgWrap = utils.wrapImg(img,points,wT,hT) # use imgThres for black and white
    imgWarpPoints = utils.drawPoints(imgCopy, points)
   #Step 3
    basePoint,imgHist=utils.getHistogram(imgWrap, display=True,minPer=0.5,region=4) #numpy array of summation of each columns
    midPoint,imgHist=utils.getHistogram(imgWrap, display=True,minPer=0.9) #numpy array of summation of each columns
    #print(basePoint - midvalue) #curve value
    middlePoint,imgHist=utils.getHistogram(imgWrap,minPer=0.5,display=True,region=4) #numpy array of summation of each columns
    curveAveragePoint,imgHist=utils.getHistogram(imgWrap, display=True,minPer=0.9) #numpy array of summation of each columns
    curveRaw = middlePoint - curveAveragePoint
   #Step 4 #Noise reduction by averaging (not values very high and low)
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))
    # Step 5
    if display != 0:
       imgInvWarp = utils.warpImg(imgWrap, points, wT, hT,inv = True)
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
       imgStacked = utils.stackImages(0.7,([img,imgWarpPoints,imgWrap],
                                         [imgHist,imgLaneColor,imgResult]))
       cv2.imshow('ImageStack',imgStacked)
    elif display == 1:
       cv2.imshow('Resutlt',imgResult)
    curve = curve/100
    if curve >1 : curve == 1
    if curve <-1: curve == -1
    cv2.imshow('Thres',imgThres)
    cv2.imshow('warp',imgWrap)
    cv2.imshow('Warp Points',imgWarpPoints)
    cv2.imshow('Histogram',imgHist)
    return curve

if __name__ == '__main__':
    cap = cv2.VideoCapture('vid1.mp4')
    initialTrackBarVals = [102,80,20,214] #.... 102,80,20,214
    utils.initializeTrackbars(initialTrackBarVals)
    frameCounter =0
  
    while True:
        frameCounter +=1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) ==frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            frameCounter=0
        success, img = cap.read() # GET THE IMAGE
        #img = cv2.resize(img,(480,240)) # RESIZE
        getLaneCurve(img)
     # baad mein chalni hain ye dono line depends on your display 2 for complete pipeline
        #curve = getLaneCurve(img,display=0,2)
        #print(curve)
# if __name__ == '__main__':
#     cap = cv2.VideoCapture('vid1.mp4')
#     initialTrackBarVals = [100,100,100,100] #.... 102,80,20,214
#     utils.initializeTrackbars(initialTrackBarVals)
#     frameCounter =0
  
#     while True:
#         frameCounter +=1
#         if cap.get(cv2.CAP_PROP_FRAME_COUNT) ==frameCounter:
#             cap.set(cv2.CAP_PROP_POS_FRAMES,0)
#             frameCounter=0
#         success, img = cap.read() # GET THE IMAGE
#         img = cv2.resize(img,(480,240)) # RESIZE
#         getLaneCurve(img)
#      # baad mein chalni hain ye dono line depends on your display 2 for complete pipeline
#         #curve = getLaneCurve(img,display=0,2)
#         #print(curve)
        cv2.imshow('Vid',img)
        #cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cap.release() 
   # Closes all the frames 
    cv2.destroyAllWindows()    
            
 
          
'''-----------------------------------------------------------------------------

-------------------------------------------------------------------------
ṇ





 ...............................................................
image will be wraped ac to threshold but we want to get the points



------------------------------------------------------------------------------




------------------------------------------------------------------------------------
instead of summing the columns we have to sum the bottom part soi that we get the center of the path





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
'''