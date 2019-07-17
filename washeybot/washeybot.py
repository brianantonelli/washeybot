#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
import argparse
import locale
import imutils
from PIL import Image

import cv2
import numpy as np

locale.setlocale(locale.LC_ALL, "C")  # needed for tesserocr

from tesserocr import PyTessBaseAPI, PSM, OEM


parser = argparse.ArgumentParser(description="")
parser.add_argument("--image", help="Path to image.")
args = parser.parse_args()

level = logging.INFO
logging.basicConfig(level=level, format="%(asctime)s %(levelname)-2s [%(filename)s] > %(message)s",
                    datefmt="%m-%d %H:%M")


def ocr(img):
    with PyTessBaseAPI(psm=PSM.SINGLE_LINE, oem=OEM.LSTM_ONLY) as api:
        api.SetImage(Image.fromarray(img))
        api.Recognize()
        text = api.GetUTF8Text().strip()

        return text


def process_image(image):
    logging.info("Processing image..")

    img = cv2.imread(image)
    orig = img.copy()

    # convert the image to grayscale, blur it, and find edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    cropped = gray[20:110, 34:165]
    blurred = cv2.GaussianBlur(cropped, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    cv2.imshow("Edged", edged)

    thresh = cv2.threshold(cropped, 0, 255, cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Threshold", thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []

    img_bb = img[20:110, 34:165]
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        logging.info("contour w,h: {}, {}".format(w, h))
        # TODO: merge matching X+W with lowers into one contour (to collapse segments) ?
        if h > 7:
            digitCnts.append(c)
            cv2.rectangle(img_bb, (x, y), (x+w, y+h), (38, 222, 42), 1)

    logging.info("found {} digit contours".format(len(digitCnts)))
    cv2.imshow("digits", img_bb)
    cv2.waitKey(0)


def main():
    if not args.image:
        logging.info("Please provide an image with --image")
        sys.exit(1)

    process_image(args.image)

if __name__ == "__main__":
    main()
