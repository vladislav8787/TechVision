import cv2 as cv
import numpy as np
import my_functions as mfs

features = mfs.Features() #инициализация характеристик обработки видеопотока и создание окон

while True:
    success, frame = mfs.Features.cap.read()
    frame = cv.bilateralFilter(frame, 9, 75, 75)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #получение значений в HSV
    h = cv.getTrackbarPos("H", "track")
    s = cv.getTrackbarPos("S", "track")
    v = cv.getTrackbarPos("V", "track")
    hl = cv.getTrackbarPos("HL", "track")
    sl = cv.getTrackbarPos("SL", "track")
    vl = cv.getTrackbarPos("VL", "track")
    thresh1 = cv.getTrackbarPos("T1", "track")
    thresh2 = cv.getTrackbarPos("T2", "track")

    lower = np.array([hl, sl, vl])
    upper = np.array([h, s, v])

    mask = cv.inRange(hsv, lower, upper)
    only_object = cv.bitwise_and(frame, frame, mask=mask)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, mfs.Features.kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, mfs.Features.kernel)

    contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    mfs.Features.get_contours(only_object, thresh1, thresh2, mfs.Features.kernel, frame, mfs.Features.depth)

    if cv.waitKey(1) & 0xFF == ord('q'):
        if mfs.Features.count_to_exit >= mfs.Features.amount:
            break
        mfs.Detected_object.info_analysis(features.info, lower, upper)
        mfs.Features.count_to_exit += 1

    cv.imshow("mask", closing)
    cv.imshow("frame", frame)
