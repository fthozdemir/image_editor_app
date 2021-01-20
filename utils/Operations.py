import cv2
import base64
import numpy as np
from scipy import ndimage
from io import BytesIO
from PIL import Image, ImageEnhance
from imageio import imread
import PIL.ImageOps
import math
import sys
from shapely.geometry import Point
from shapely.geometry import Polygon
import time


import utils.b64kit as b64kit


def rotate(baseString, angle):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)

    try:
        alpha_ones = img[:, :, 3]  # have alpha channel or not
    except:
        # Add imitation alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    final = ndimage.rotate(img, angle)

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def lumos(baseString, lumos):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)
    factor = lumos/50.0
    try:
        alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
        bgr = img[:, :, :3]  # Channels 0..2

        new_pil_img = Image.fromarray(bgr)
        enhancer = ImageEnhance.Brightness(new_pil_img)
        output = enhancer.enhance(factor)
        final = np.array(output)
        final = np.dstack([final, alpha])
    except:
        new_pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(new_pil_img)
        output = enhancer.enhance(factor)
        final = np.array(output)

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def flip(baseString, hor, ver):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)

    try:
        final = img
        if(hor and ver):
            final = np.flip(img, (0, 1))
        else:
            if(hor):
                final = np.fliplr(img)
            if(ver):
                final = np.flipud(img)

    except:
        final = img

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def mirror(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)
    try:
        final = np.fliplr(img)
    except:
        final = img

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def inverse(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)
    try:
        alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
        bgr = img[:, :, :3]  # Channels 0..2

        new_pil_img = Image.fromarray(bgr)
        inverted_bgr = PIL.ImageOps.invert(new_pil_img)
        final = np.array(inverted_bgr)
        final = np.dstack([final, alpha])
    except:
        final = cv2.bitwise_not(img)

    img_code = b64kit.image_to_base64(final)

    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def crop(baseString, _p, width):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()

    img = b64kit.base64_to_image(baseString)
    ratio = img.shape[1]/width

    for p in _p:
        temp = p[0]
        p[0] = math.floor(p[1]*ratio)
        p[1] = math.floor(temp*ratio)

    im = Image.fromarray(img).convert('RGBA')
    pixels = np.array(im)
    im_copy = np.array(im)

    region = Polygon(_p)
    for index, pixel in np.ndenumerate(pixels):
      # Unpack the index.
        row, col, channel = index
        # We only need to look at spatial pixel data for one of the four channels.
        if channel != 0:
            continue
        point = Point(row, col)
        if not region.contains(point):
            im_copy[(row, col, 0)] = 255
            im_copy[(row, col, 1)] = 255
            im_copy[(row, col, 2)] = 255
            im_copy[(row, col, 3)] = 0

    final = im_copy
    img_code = b64kit.image_to_base64(final)
    print("{},{} image process time : {}",
          img.shape[1], img.shape[0], time.time()-start)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def histogramEqualizer(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    final = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    try:
        alpha = img[:, :, 3]
        final = np.dstack([final, alpha])
    except:
        print("3-channel image")
    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def contrast(baseString, c_value):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)
    isAlphaExist = False
    if(len(img.shape) == 2):
        grayed = img
    else:
        print(len(img.shape))

        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    factor = c_value/50.0
    final = img

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            try:
                for c in range(img.shape[2]):
                    final[y, x, c] = np.clip(factor*img[y, x, c], 0, 255)
            except:
                final[y, x] = np.clip(factor*img[y, x], 0, 255)
    if(isAlphaExist):
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code
