if __name__ == '__main__':
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