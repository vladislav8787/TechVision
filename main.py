import cv2 as cv
import numpy as np
import sqlite3 as sql
import detected_object
import features

videoFeatures = features.Features() #инициализация характеристик обработки видеопотока и создание окон

#цикл обработки кадров видеопотока
while True:
    success, frame = videoFeatures.cap.read()
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
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, videoFeatures.kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, videoFeatures.kernel)
    #обнаружение объекта, при нахождении в кадре замкнутых контуров
    detectedObject = videoFeatures.getContours(onlyObject, thresh1, thresh2, videoFeatures.kernel, frame, videoFeatures.depth)
    #отслеживание нажатия клавиши
    if cv.waitKey(1) & 0xFF == ord('q'):
        if videoFeatures.countToExit >= videoFeatures.amount:
            break
        detectedObject.infoAnalysis(lower, upper)
        videoFeatures.countToExit += 1
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
    index = detected_object.Indexes()
    #задание параметров hsv для объекта
    for object in allObjects:
        cv.setTrackbarPos("H", "track", object[index.H])
        cv.setTrackbarPos("HL", "track", object[index.HL])
        cv.setTrackbarPos("S", "track", object[index.S])
        cv.setTrackbarPos("SL", "track", object[index.SL])
        cv.setTrackbarPos("V", "track", object[index.V])
        cv.setTrackbarPos("VL", "track", object[index.VL])
        #цикл обработки кадров видеопотока
        while True:
            success, frame = videoFeatures.cap.read()
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
            opening = cv.morphologyEx(mask, cv.MORPH_OPEN, videoFeatures.kernel)
            closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, videoFeatures.kernel)
            frame = cv.putText(frame, f"Searching \"{object[0]}\"", (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            #распознавание объекта
            detected_object.DetectedObject.recognizeObject(onlyObject, thresh1, thresh2, videoFeatures.kernel,
                                frame, object, videoFeatures, index)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            cv.imshow("frame", frame)
            cv.imshow("mask", closing)