import cv2
import numpy as np
import pandas as pd
import model
import math

vc = cv2.VideoCapture(0)
cv2.namedWindow('img',0)
labels = pd.read_csv('labels.txt')
model = model.get_model_final(len(labels['class']),128)
model.load_weights('weights2.hdf5')

def peer2(img):
    b, g, r = cv2.split(img)
    ret, m1 = cv2.threshold(r, 95, 255, cv2.THRESH_BINARY)
    ret, m2 = cv2.threshold(g, 30, 255, cv2.THRESH_BINARY)
    ret, m3 = cv2.threshold(b, 20, 255, cv2.THRESH_BINARY)
    mmax = cv2.max(r, cv2.max(g, b))
    mmin = cv2.min(r, cv2.min(g, b))

    ret, m4 = cv2.threshold(mmax - mmin, 15, 255, cv2.THRESH_BINARY)
    ret, m5 = cv2.threshold(cv2.absdiff(r, g), 15, 255, cv2.THRESH_BINARY)
    m6 = cv2.compare(r, g, cv2.CMP_GE)
    m7 = cv2.compare(r, b, cv2.CMP_GE)
    mask = m1 & m2 & m3 & m6 & m4 & m5 & m7

    return mask



def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


while True:
    _, img = vc.read()
    # input = np.array([cv2.resize(img,(128,128))],np.float32)/255
    # y = model.predict(input)
    # # print y[0]
    # ans = np.argmax(y[0])
    # print labels['class'][ans]
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img,labels['class'][ans],(10,400), font, 4,(255,255,255),2,cv2.LINE_AA)
    th3 = peer2(img)
    kernel = np.ones((7,7), np.uint8)
    closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    # gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
    _, contours, hiearachy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    cv2.drawContours(img, contours, max_index, (255, 255, 255), 3)
    hull = cv2.convexHull(cnt, returnPoints=False)
    # cv2.drawContours(img,[hull],0,(0,255,0),3)
    defects = cv2.convexityDefects(cnt,hull)
    _,fin = calculateFingers(cnt,img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if (fin == 2):
        cv2.putText(img, 'V', (10, 400), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    if (fin == 3):
        cv2.putText(img, 'A', (10, 400), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    if (fin == 1):
        cv2.putText(img, 'Point', (10, 400), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    if (fin == 0):
        cv2.putText(img, 'B', (10, 400), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    if (fin == 4 or fin == 5):
        cv2.putText(img, 'Five', (10, 400), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    # for i in range(defects.shape[0]):
    #     s, e, f, d = defects[i, 0]
    #     start = tuple(cnt[s][0])
    #     end = tuple(cnt[e][0])
    #     far = tuple(cnt[f][0])
    #     cv2.line(img, start, end, [0, 255, 0], 2)
    #     cv2.circle(img, far, 5, [0, 0, 255], -1)
    cv2.imshow('img',img)
    cv2.waitKey(5)








