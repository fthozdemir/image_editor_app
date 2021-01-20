import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from imageio import imread
import math
import scipy
import sys
import utils.b64kit as b64kit
import utils.Operations as op
from scipy.interpolate import UnivariateSpline
import time
import random


def hsv(img, l, u):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([l, 128, 128])  # setting lower HSV value
    upper = np.array([u, 255, 255])  # setting upper HSV value
    mask = cv2.inRange(hsv, lower, upper)  # generating mask
    return mask


def spreadLookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def compute_average_image_color(img):
    width = img.shape[0]
    height = img.shape[1]

    r_total = 0
    g_total = 0
    b_total = 0

    r, g, b = cv2.split(img)
    count = 0
    for x in range(0, width):
        for y in range(0, height):
            r_total += r[x, y]
            g_total += g[x, y]
            b_total += b[x, y]
            count += 1

    return r_total/count, g_total/count, b_total/count


def grayish(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)

    #
    #   eğer bgra image' ı direkt grayscale edersek alpha chanell ı kaybederiz.
    #   Bu sebeple alpha kanalını önce ayıklayıp sonra eklememiz gerekir.

    try:
        alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
        bgr = img[:, :, :3]  # Channels 0..2
        gray_bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        final = np.dstack([gray_bgr, alpha])
    except:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        alpha = img[:, :, 3]
        final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def emboss(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)

    kernel_emboss_1 = np.array([[0, -1, -1],
                                [1, 0, -1],
                                [1, 1, 0]])

    try:
        alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
        bgr = img[:, :, :3]  # Channels 0..2
        output = cv2.filter2D(bgr, -1, kernel_emboss_1) + 12
        final = np.dstack([output, alpha])

    except:
        final = cv2.filter2D(img, -1, kernel_emboss_1) + 12
    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def pixelArt(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    input = b64kit.base64_to_image(baseString)

    factor = 12.5
# Get input size
    height, width = input.shape[:2]

    w, h = (math.floor(width/factor), math.floor(height/factor))

# Resize input to "pixelated" size
    temp = cv2.resize(input, (w, h), interpolation=cv2.INTER_LINEAR)

# Initialize output image
    final = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def popArt(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    # convert base64 string to cv image
    img = b64kit.base64_to_image(baseString)
    isAlphaExist = False  # seperate the alpha channel for not lose transparency
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass
    background_colour = [224, 247, 20]
    dots_colour = (247, 19, 217)
    max_dots = 120

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_image_height = img.shape[0]
    original_image_width = img.shape[1]

    if original_image_height == max(original_image_height, original_image_width):
        downsized_image = cv2.resize(img, (int(
            original_image_height*(max_dots/original_image_width)), max_dots))
    else:
        downsized_image = cv2.resize(img, (max_dots, int(
            original_image_height*(max_dots/original_image_width))))
    downsized_image_height = downsized_image.shape[0]
    downsized_image_width = downsized_image.shape[1]
    multiplier = 100
    blank_img_height = downsized_image_height * multiplier
    blank_img_width = downsized_image_width * multiplier
    padding = int(multiplier/2)
    blank_image = np.full(
        ((blank_img_height), (blank_img_width), 3), background_colour, dtype=np.uint8)

    for y in range(0, downsized_image_height):
        for x in range(0, downsized_image_width):
            cv2.circle(blank_image, (((x*multiplier)+padding), ((y*multiplier)+padding)),
                       int((0.6 * multiplier) * ((255-downsized_image[y][x])/255)), dots_colour, -1)

    width = int(blank_image.shape[1] * 10 / 100)
    height = int(blank_image.shape[0] * 10 / 100)

    # dsize because output is too large
    dsize = (original_image_width, original_image_height)

    # resize image
    final = cv2.resize(blank_image, dsize)

    if(isAlphaExist):  # add aplha channel if it has
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def oldtv(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)
    isAlphaExist = False
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass
    b, g, r = cv2.split(img)
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = 0.8  # creating threshold. This means noise will be added to 80% pixels
    for i in range(height):
        for j in range(width):
            if np.random.rand() <= thresh:
                if np.random.randint(2) == 0:
                    # adding random value between 0 to 64. Anything above 255 is set to 255.
                    point = min(gray[i, j] + np.random.randint(0, 64), 255)
                    b[i, j] = point
                    g[i, j] = point
                    r[i, j] = point

                else:
                    # subtracting random values between 0 to 64. Anything below 0 is set to 0.
                    point = min(gray[i, j] + np.random.randint(0, 64), 255)
                    b[i, j] = point
                    g[i, j] = point
                    r[i, j] = point

    final = np.dstack([b, g, r])
    if(isAlphaExist):
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def sketch(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    # convert base64 string to cv image
    img = b64kit.base64_to_image(baseString)
    isAlphaExist = False  # seperate the alpha channel for not lose transparency
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray_image = 255 - gray_image
    k_size = (21, 21)
    blurred_img = cv2.GaussianBlur(inverted_gray_image, k_size, 0)
    inverted_blurred_img = 255 - blurred_img

    final = cv2.divide(gray_image, inverted_blurred_img, scale=256.0)

    if(isAlphaExist):  # add aplha channel if it has
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def splash(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    # convert base64 string to cv image
    img = b64kit.base64_to_image(baseString)

    isAlphaExist = False  # seperate the alpha channel for not lose transparency
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    res = np.zeros(img.shape, np.uint8)  # creating blank mask for result
    l = 15  # the lower range of Hue we want
    u = 255  # the upper range of Hue we want
    mask = hsv(img, l, u)
    inv_mask = cv2.bitwise_not(mask)  # inverting mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # region which has to be in color
    res1 = cv2.bitwise_and(img, img, mask=mask)
    # region which has to be in grayscale
    print("2")

    res2 = cv2.bitwise_and(gray, gray, mask=inv_mask)
    for i in range(3):
        res[:, :, i] = res2  # storing grayscale mask to all three slices
    final = cv2.bitwise_or(res1, res)  # joining grayscale and color region

    if(isAlphaExist):  # add aplha channel if it has
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def sepya(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    # convert base64 string to cv image
    img = b64kit.base64_to_image(baseString)

    isAlphaExist = False  # seperate the alpha channel for not lose transparency
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    # converting to float to prevent loss
    img = np.array(img, dtype=np.float64)
    img = cv2.transform(img, np.matrix([[0.393, 0.769, 0.189],
                                        [0.349, 0.686, 0.168],
                                        [0.272, 0.534, 0.131]]))  # multipying image with special sepia matrix
    # normalizing values greater than 255 to 255
    img[np.where(img > 255)] = 255
    final = np.array(img, dtype=np.uint8)  # converting back to int

    if(isAlphaExist):  # add aplha channel if it has
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def cartoon(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    # convert base64 string to cv image
    img = b64kit.base64_to_image(baseString)
    isAlphaExist = False  # seperate the alpha channel for not lose transparency
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
            print(isAlphaExist)
        except:
            pass

    img_small = cv2.pyrDown(img)
    num_iter = 5
    for _ in range(num_iter):
        img_small = cv2.bilateralFilter(
            img_small, d=9, sigmaColor=9, sigmaSpace=7)

    img_rgb = cv2.pyrUp(img_small)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edge = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    final = cv2.bitwise_and(img, img_edge)

    if(isAlphaExist):  # add aplha channel if it has
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def oily(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    # convert base64 string to cv image
    img = b64kit.base64_to_image(baseString)

    isAlphaExist = False  # seperate the alpha channel for not lose transparency
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    height, width, channels = img.shape

    # Grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Divide the image into 8x8 small blocks, and count the gray value of each pixel in the small block
    # Divide the gray level, such as 0-255: 0-63, 64-127, ...
    # Find the number of each level in each small block and find the most
    # Replace the original pixel with the most average

    final = np.zeros((height, width, channels), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            max_level_arr = []
            levelMap = {}  # k: level, v: list of original pixel values
            for m in range(-3, 3):
                for n in range(-3, 3):
                    # Handling out of bounds
                    if i + m >= height or i + m < 0:
                        m = -m
                    if j + n >= width or j + n < 0:
                        n = -n

                    # Classification 0-31, 32-63, ...
                    level = gray_img[i + m, j + n] // 32
                    if not level in levelMap.keys():
                        levelMap[level] = [img[i + m, j + n], ]
                    else:
                        levelMap[level].append(img[i + m, j + n])
                    # The highest number of levels
                    if len(levelMap[level]) > len(max_level_arr):
                        max_level_arr = levelMap[level]
            # Take the mean
            size = len(max_level_arr)
            b_sum, g_sum, r_sum = 0, 0, 0
            for b, g, r in max_level_arr:
                b_sum += b
                g_sum += g
                r_sum += r
            final[i, j] = [b_sum // size, g_sum // size, r_sum // size]

    if(isAlphaExist):  # add aplha channel if it has
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def abstractify(baseString):
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
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k_size = (51, 51)
    blurred = cv2.GaussianBlur(grayed, k_size, 0)

    level = 7

    indices = np.arange(0, 256)
    divider = np.linspace(0, 255, level+1)[1]
    quantiz = np.int0(np.linspace(0, 255, level))
    color_levels = np.clip(np.int0(indices/divider), 0, level-1)
    palette = quantiz[color_levels]
    img2 = palette[blurred]
    img2 = cv2.convertScaleAbs(img2)

    final = cv2.applyColorMap(img2, cv2.COLORMAP_RAINBOW)
    if(isAlphaExist):
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def warm(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    # convert base64 string to cv image
    img = b64kit.base64_to_image(baseString)

    isAlphaExist = False  # seperate the alpha channel for not lose transparency
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    increaseLookupTable = spreadLookupTable(
        [0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable(
        [0, 64, 128, 256], [0, 50, 100, 256])

    red_channel, green_channel, blue_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    final = cv2.merge((red_channel, green_channel, blue_channel))

    if(isAlphaExist):  # add aplha channel if it has
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def cold(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    # convert base64 string to cv image
    img = b64kit.base64_to_image(baseString)

    isAlphaExist = False  # seperate the alpha channel for not lose transparency
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    increaseLookupTable = spreadLookupTable(
        [0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable(
        [0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    final = cv2.merge((red_channel, green_channel, blue_channel))

    if(isAlphaExist):  # add aplha channel if it has
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def lines(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    # convert base64 string to cv image
    img = b64kit.base64_to_image(baseString)
    isAlphaExist = False  # seperate the alpha channel for not lose transparency
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass
    final = cv2.Canny(img, 100, 200)

    if(isAlphaExist):  # add aplha channel if it has
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def blush(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)
    isAlphaExist = False
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    _r, _g, _b = compute_average_image_color(img)
    img[:, :, 2] = _b
    img[:, :, 1] = _g

    final = img
    if(isAlphaExist):
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def glass(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)
    isAlphaExist = False
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    height, width, channels = img.shape

    # Randomly select a point in the 10x10 range of the lower right for each point
    # mm is the value range, the bigger the wider
    mm = 10
    for i in range(0, height):
        for j in range(0, width):
            # Random points
            r = int(random.random() * mm)
            r_i, r_j = i + r, j + r
            # Handling out of bounds
            if r_i >= height:
                r_i = height - 1
            elif r_i < 0:
                r_i = 0
            if r_j >= width:
                r_j = width - 1
            elif r_j < 0:
                r_j = 0
            # Modify pixels
            img[i, j] = img[r_i, r_j]

    final = img
    if(isAlphaExist):
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def xpro(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)
    isAlphaExist = False
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    B, G, R = cv2.split(img)

    # define vignette scale
    vignetteScale = 6

    # calculate the kernel size
    k = np.min([img.shape[1], img.shape[0]])/vignetteScale

    # create kernel to get the Halo effect
    kernelX = cv2.getGaussianKernel(img.shape[1], k)
    kernelY = cv2.getGaussianKernel(img.shape[0], k)
    kernel = kernelY * kernelX.T

    # normalize the kernel
    mask = cv2.normalize(kernel, None, alpha=0, beta=1,
                         norm_type=cv2.NORM_MINMAX)

    # apply halo effect to all the three channels of the image
    B = B + B*mask
    G = G + G*mask
    R = R + R*mask

    # merge back the channels
    output = cv2.merge([B, G, R])

    output = output / 2

    # limit the values between 0 and 255
    output = np.clip(output, 0, 255)

    # convert back to uint8
    output = np.uint8(output)

    # split the channels
    B, G, R = cv2.split(output)

    # Interpolation values
    redValuesOriginal = np.array([0, 42, 105, 148, 185, 255])
    redValues = np.array([0, 28, 100, 165, 215, 255])
    greenValuesOriginal = np.array([0, 40, 85, 125, 165, 212, 255])
    greenValues = np.array([0, 25, 75, 135, 185, 230, 255])
    blueValuesOriginal = np.array([0, 40, 82, 125, 170, 225, 255])
    blueValues = np.array([0, 38, 90, 125, 160, 210, 222])

    # create lookuptable
    allValues = np.arange(0, 256)

    # create lookup table for red channel
    redLookuptable = np.interp(allValues, redValuesOriginal, redValues)
    # apply the mapping for red channel
    R = cv2.LUT(R, redLookuptable)

    # create lookup table for green channel
    greenLookuptable = np.interp(allValues, greenValuesOriginal, greenValues)
    # apply the mapping for red channel
    G = cv2.LUT(G, greenLookuptable)

    # create lookup table for blue channel
    blueLookuptable = np.interp(allValues, blueValuesOriginal, blueValues)
    # apply the mapping for red channel
    B = cv2.LUT(B, blueLookuptable)

    # merge back the channels
    output = cv2.merge([B, G, R])

    # convert back to uint8
    output = np.uint8(output)

    # adjust contrast
    # convert to YCrCb color space
    output = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)

    # convert to float32
    output = np.float32(output)

    # split the channels
    Y, Cr, Cb = cv2.split(output)

    # scale the Y channel
    Y = Y * 1.2

    # limit the values between 0 and 255
    Y = np.clip(Y, 0, 255)

    # merge back the channels
    output = cv2.merge([Y, Cr, Cb])

    # convert back to uint8
    output = np.uint8(output)

    # convert back to BGR color space
    final = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)

    if(isAlphaExist):
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def daylight(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)
    original = img.copy()

    isAlphaExist = False
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    # convert image to HSV color space
    image_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  # Conversion to HLS
    image_HLS = np.array(image_HLS, dtype=np.float64)
    daylight = 1.15
    # scale pixel values up for channel 1(Lightness)
    image_HLS[:, :, 1] = image_HLS[:, :, 1]*daylight
    # Sets all values above 255 to 255
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    final = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR)  # Conversion to RGB

    if(isAlphaExist):
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def moon(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)

    isAlphaExist = False
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    # convert to LAB color space
    output = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # split into channels
    L, A, B = cv2.split(output)
    # Interpolation values
    originalValues = np.array([0, 15, 30, 50, 70, 90, 120, 160, 180, 210, 255])
    values = np.array([0, 0, 5, 15, 60, 110, 150, 190, 210, 230, 255])

    # create lookup table
    allValues = np.arange(0, 256)

    # Creating the lookuptable
    lookuptable = np.interp(allValues, originalValues, values)

    # apply mapping for L channels
    L = cv2.LUT(L, lookuptable)

    # convert to uint8
    L = np.uint8(L)

    # merge back the channels
    output = cv2.merge([L, A, B])

    # convert back to BGR color space
    output = cv2.cvtColor(output, cv2.COLOR_LAB2BGR)

    # desaturate the image
    # convert to HSV color space
    output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

    # split into channels
    H, S, V = cv2.split(output)

    # Multiply S channel by saturation scale value
    S = S * 0.01

    # convert to uint8
    S = np.uint8(S)

    # limit the values between 0 and 256
    S = np.clip(S, 0, 255)

    # merge back the channels
    output = cv2.merge([H, S, V])

    # convert back to BGR color space
    final = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    if(isAlphaExist):
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code


def blueish(baseString):
    print(sys._getframe().f_code.co_name + " running")
    start = time.time()
    img = b64kit.base64_to_image(baseString)
    original = img.copy()

    isAlphaExist = False
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        try:
            alpha = img[:, :, 3]  # Channel 3 //have alpha channel or not
            img = img[:, :, :3]  # Channels 0..2
            isAlphaExist = True
        except:
            pass

    for i in range(3):
        if i == 2:
            # creating table for exponent
            table = np.array([min((i**1.05), 255)
                              for i in np.arange(0, 256)]).astype("uint8")
            img[:, :, i] = cv2.LUT(img[:, :, i], table)
        else:
            img[:, :, i] = 0  # setting values of all other slices to 0

    final = img
    if(isAlphaExist):
        final = np.dstack([final, alpha])

    img_code = b64kit.image_to_base64(final)
    print(sys._getframe().f_code.co_name +
          " finished in " + "{:.2f}".format(time.time()-start)+" s")
    return img_code
