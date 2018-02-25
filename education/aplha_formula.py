    alpha =  255 - alpha
    background_gray = np.array(background_gray,np.float32)/255
    alpha = np.array(alpha,np.float32)/255
    img_gray = np.array(img_gray, np.float32)/255
    res = alpha*background_gray + img_gray*(1-alpha)
