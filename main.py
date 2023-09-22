import cv2 as cv
import numpy as np
import my_functions as mfs
import sqlite3 as sql

features = mfs.Features() #инициализация характеристик обработки видеопотока и создание окон

#цикл обработки кадров видеопотока
while True:
    success, frame = features.cap.read()
    frame = cv.bilateralFilter(frame, 9, 75, 75)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #получение значений в hsv
    h = cv.getTrackbarPos("H", "track")
    s = cv.getTrackbarPos("S", "track")
    v = cv.getTrackbarPos("V", "track")
    hl = cv.getTrackbarPos("HL", "track")
    sl = cv.getTrackbarPos("SL", "track")
    vl = cv.getTrackbarPos("VL", "track")
    thresh1 = cv.getTrackbarPos("T1", "track")
    thresh2 = cv.getTrackbarPos("T2", "track")
    #верхняя и нижняя граница hsv
    lower = np.array([hl, sl, vl])
    upper = np.array([h, s, v])
    #получение маски, изолированного объекта и проведение морфологических операций
    mask = cv.inRange(hsv, lower, upper)
    onlyObject = cv.bitwise_and(frame, frame, mask=mask)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, features.kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, features.kernel)
    #обнаружение объекта, при нахождении в кадре замкнутых контуров
    detectedObject = features.getContours(onlyObject, thresh1, thresh2, features.kernel, frame, features.depth)
    #отслеживание нажатия клавиши
    if cv.waitKey(1) & 0xFF == ord('q'):
        if features.countToExit >= features.amount:
            break
        detectedObject.infoAnalysis(lower, upper)
        features.countToExit += 1
    #отображение окон
    cv.imshow("mask", closing)
    cv.imshow("frame", frame)

cv.destroyWindow("frame")
cv.destroyWindow("mask")

#перебор объектов из БД для распознавания
with sql.connect('D:\pythonProject\TechVision\objects.db') as con:
    cur = con.cursor()
    cur.execute("""SELECT * FROM objects""")
    allObjects = cur.fetchall()
    #задание параметров hsv для объекта
    for object in allObjects:
        detectedObject = mfs.DetectedObject()

        cv.setTrackbarPos("H", "track", object[6])
        cv.setTrackbarPos("HL", "track", object[9])
        cv.setTrackbarPos("S", "track", object[7])
        cv.setTrackbarPos("SL", "track", object[10])
        cv.setTrackbarPos("V", "track", object[8])
        cv.setTrackbarPos("VL", "track", object[11])
        #цикл обработки кадров видеопотока
        while True:
            success, frame = features.cap.read()
            frame = cv.bilateralFilter(frame, 9, 75, 75)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            h = cv.getTrackbarPos("H", "track")
            s = cv.getTrackbarPos("S", "track")
            v = cv.getTrackbarPos("V", "track")
            hl = cv.getTrackbarPos("HL", "track")
            sl = cv.getTrackbarPos("SL", "track")
            vl = cv.getTrackbarPos("VL", "track")

            lower = np.array([hl, sl, vl])
            upper = np.array([h, s, v])

            mask = cv.inRange(hsv, lower, upper)
            onlyObject = cv.bitwise_and(frame, frame, mask=mask)
            opening = cv.morphologyEx(mask, cv.MORPH_OPEN, features.kernel)
            closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, features.kernel)
            frame = cv.putText(frame, f"Searching \"{object[0]}\"", (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            #распознавание объекта
            mfs.DetectedObject.recognizeObject(onlyObject, thresh1, thresh2, features.kernel, frame, object, features)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            cv.imshow("frame", frame)
            cv.imshow("mask", closing)