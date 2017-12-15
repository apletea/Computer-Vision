import cv2

vc = cv2.VideoCapture(0)

while(True):
    _,_img = vc.read()
    img = _img.copy()
    cv2.cvtColor(img,cv2.COLOR_BGR2HSV,img)
    # cv2.cvtColor()
    # out noise
    cv2.GaussianBlur(img,(7,7),1,img,1)

    cv2.inRange(img,(5,38,51),(17,250,242),dst=img)

    cv2.cvtColor(img,cv2.COLOR_HSV2BGR,img)
    cv2.cvtColor(img,cv2.COLOR_BGR2GRAY,img)

    mean = img.mean()
    _,img =  cv2.threshold(img,mean,255,cv2.THRESH_BINARY,dst=img)
    cv2.erode(img,np.ones((3,3)),img)

    kernel = np.ones((7,7),np.uint8)
    img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel,img)
    kernel = np.ones((9,9),np.uint8)
    img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,img)
    img = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel,img)
    img = cv2.medianBlur(img,15,img)

    cv2.waitKey(2)
    cv2.imshow('cap',img)
    cv2.imshow('org',_img)
