import cv2
import numpy as np
import pandas as pd
import model

vc = cv2.VideoCapture(0)
labels = pd.read_csv('labels.txt')
model = model.get_model_final(len(labels['class']),28)
model.load_weights('weights1.hdf5')



while True:
    _, img = vc.read()
    cv2.imshow('img',img)
    cv2.waitKey(5)
    input = np.array([cv2.resize(img,(28,28))],np.float32)/255
    y = model.predict(input)
    # print y[0]
    ans = np.argmax(y[0])
    print labels['class'][ans]