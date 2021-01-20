
import cv2
import base64
import numpy as np
from scipy import ndimage
from imageio import imread
import io
from PIL import Image
import PIL.ImageOps
import math
import scipy


def grayish(baseString):

    img = base64_to_image(baseString)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_code = image_to_base64(gray_image)

    cv2.imshow("gray", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_code


def rotate(baseString):

    img = base64_to_image(baseString)
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, 22, 1.0)
    final = cv2.warpAffine(
        img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    return final


def base64_to_image(b64String):
    # b64String = "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"
    img = imread(io.BytesIO(base64.b64decode(b64String)))
   # img = cv2.cvtColor(b_img, cv2.COLOR_RGB2BGR)

    return img


def image_to_base64(img):
    pil_img = Image.fromarray(img)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    b64string = base64.b64encode(buff.getvalue()).decode("utf-8")

    return b64string


def crop():
    img = cv2.imread("C:\\Users\\fatih\\Desktop\\bat.jpg")
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    p = [[100, 350], [120, 400], [310, 350], [360, 200], [350, 20], [25, 120]]
    points = np.array(p)
    # method 1 smooth region
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # method 2 not so smooth region
    # cv2.fillPoly(mask, points, (255))
    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    # crate the white background of the same size of original image
    wbg = np.ones_like(img, np.uint8)*255
    cv2.bitwise_not(wbg, wbg, mask=mask)
    # overlap the resulted cropped image on the white background
    dst = wbg+res
    cv2.imshow('Original', img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Cropped", cropped)
    cv2.imshow("Samed Size Black Image", res)
    cv2.imshow("Samed Size White Image", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imshow('Output', crop)
    cv2.imwrite("C:\\Users\\fatih\\Desktop\\out.png", cropped)
    cv2.waitKey(0)


def exponential_function(channel, exp):
    table = np.array([min((i**exp), 255) for i in np.arange(0, 256)]
                     ).astype("uint8")  # creating table for exponent
    channel = cv2.LUT(channel, table)
    return channel


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


def test():
    img = cv2.imread("C:\\Users\\fatih\\Desktop\\two.png")
    # convert to YCrCb color space
    imageYcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # convert to float32
    imageYcb = np.float32(imageYcb)

    # split into channels
    Y, C, B = cv2.split(imageYcb)

    # define scale factor
    alpha = 1.5

    # scale the Y channel
    Y = Y * alpha

    # clip the values betweeen 0 and 255
    Y = np.clip(Y, 0, 255)

    # merge the channels
    imageYcb = cv2.merge([Y, C, B])

    # convert back from float32
    imageYcb = np.uint8(imageYcb)

    # convert back to BGR color space
    result = cv2.cvtColor(imageYcb, cv2.COLOR_YCrCb2BGR)

    # display image
    cv2.imshow("image", img)
    cv2.imshow("result", result)

    # press esc to exit the program
    cv2.waitKey(0)

    # close all the opended windows
    cv2.destroyAllWindows()


for i in range(10):
    for j in range(10):
        print(i)
        print(j)
        j += 1
    i += 1

# baseString = image_to_base64(test_img)
# out = rotate(baseString)
# cv2.imwrite("C:\\Users\\fatih\\Desktop\\out.png", out)
#
#  new_im = Image.fromarray(img)
#    vert_img = new_im.transpose(method=Image.FLIP_TOP_BOTTOM)
#    final = np.array(vert_img)
