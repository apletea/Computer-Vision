import numpy as np
from skimage.transform import resize


def shift(img, x, y):
    reshaped = np.roll(img, x, 0)
    reshaped = np.roll(reshaped, y, 1)
    return reshaped


def resize_2(img):
    hei = img.shape[0]
    wid = img.shape[1]
    return resize(img, (int(hei / 2), int(wid / 2)), mode='reflect')


def cross_correlation(img1, img2, max_shift):
    img1_p = img1[max_shift:-max_shift]
    img2_p = img2[max_shift:-max_shift]
    return np.sum(img1_p * img2_p) / np.sqrt(np.sum(img1_p**2) * np.sum(img2_p**2))


def mse(img1, img2, max_shift):
    img1_p = img1[max_shift:-max_shift]
    img2_p = img2[max_shift:-max_shift]
    return np.mean((img1_p - img2_p)**2)


def min_shift(channel, gr):
    max_shift = 15
    shifts = {}
    shifts['mse'] = []
    shifts['corr'] = []

    min_accur = 1000
    max_accur = -1000

    for x in range(-max_shift, max_shift + 1):
        for y in range(-max_shift, max_shift + 1):
            shifted = shift(channel, x, y)
            accur_mse = mse(gr, shifted, max_shift)
            accur_corr = cross_correlation(gr, shifted, max_shift)

            if accur_mse < min_accur:
                min_accur = accur_mse
                cur_shift_mse = (x, y)
            if accur_corr > max_accur:
                max_accur = accur_corr
                cur_shift_corr = (x, y)

        shifts['mse'] = cur_shift_mse
        shifts['corr'] = cur_shift_corr
    return shifts


def min_pyramid_shift(channel, gr, prev_shift, max_shift):
    shifts = {}
    shifts['mse'] = []
    shifts['corr'] = []

    max_accur = -100

    for x in range(-max_shift + prev_shift[0], max_shift + 1 + prev_shift[0]):
        for y in range(-max_shift + prev_shift[1], max_shift + 1 + prev_shift[1]):
            shifted = shift(channel, -x, -y)
            accur_corr = cross_correlation(gr, shifted, max_shift)
            if accur_corr > max_accur:
                max_accur = accur_corr
                cur_shift_corr = np.array([x, y])
        shifts['corr'] = cur_shift_corr
    return shifts['corr']


def pyramid(channel, gr):
    pyramid = []
    pyramid.append([channel, gr])

    while pyramid[-1][0].shape[0] > 500 and pyramid[-1][0].shape[1] > 500:
        pyramid.append([resize_2(pyramid[-1][0]), resize_2(pyramid[-1][1])])

    prev_shift = np.array([0, 0])

    max_shift = 15
    for im in reversed(pyramid):
        prev_shift *= 2
        prev_shift = min_pyramid_shift(im[0], im[1], prev_shift, max_shift)
        max_shift = 1
    return prev_shift


def align(img, g_coord):
    try:
        img = np.array(np.split(img, 3))
    except:
        try:
            img = np.array(np.split(img[:-1], 3))
        except:
            img = np.array(np.split(img[:-2], 3))

    shape_of_part = img.shape[1]

    x_fr = int(0.05 * img.shape[1])
    y_fr = int(0.05 * img.shape[2])

    img = img[:, x_fr: -x_fr, y_fr: -y_fr]

    red = img[0]
    gr = img[1]
    blue = img[2]

    shift_r = pyramid(red, gr)
    shift_b = pyramid(blue, gr)

    red = shift(img[0], shift_r[0], shift_r[1])
    blue = shift(img[2], shift_b[0], shift_b[1])

    aligned_image = np.dstack((red, gr, blue))

    result = []
    result.append((g_coord[0] + shift_r[0] - shape_of_part, g_coord[1] + shift_r[1]))
    result.append((g_coord[0] + shift_b[0] + shape_of_part, g_coord[1] + shift_b[1]))

return aligned_image[:, :, ::-1], result[0], result[1]
