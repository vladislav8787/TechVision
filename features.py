import cv2 as cv
import numpy as np
import detected_object

class Features():
    def __init__(self):
        #ядро, кол-во объектов, расстояние для определения параметров, счетчик,
        #минимальная площадь контура, объект захвата видеопотока с камеры 1
        self.kernel = np.ones((5, 5), np.uint8)
        self.amount = int(input("Enter the number of objects: "))
        self.depth = 40
        self.countToExit = 0
        self.minAreaContour = 2500 #значение в пикселях
        self.detectingOffset = 0.3 #погрешность для коэффициентов
        self.cap = cv.VideoCapture(1, cv.CAP_DSHOW)
        #окна
        cv.namedWindow("frame")
        cv.namedWindow("track", cv.WINDOW_NORMAL)
        #ползунки
        cv.createTrackbar("H", "track", 0, 179, self.nothing)
        cv.createTrackbar("S", "track", 0, 255, self.nothing)
        cv.createTrackbar("V", "track", 0, 255, self.nothing)
        cv.createTrackbar("HL", "track", 0, 179, self.nothing)
        cv.createTrackbar("SL", "track", 0, 255, self.nothing)
        cv.createTrackbar("VL", "track", 0, 255, self.nothing)
        cv.createTrackbar("T1", "track", 0, 255, self.nothing)
        cv.createTrackbar("T2", "track", 0, 255, self.nothing)

    #обработка замкнутых контуров
    def getContours(self, onlyObject, thresh1, thresh2, kernel, frame, depth):
        gray = cv.cvtColor(onlyObject, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, thresh1, thresh2)
        dil = cv.dilate(canny, kernel, iterations=1)
        frame = cv.putText(frame, f"Distance: {depth} cm", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        contours, hierarchy = cv.findContours(dil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        counter = 0
        for contour in contours:
            area = cv.contourArea(contour)
            if area > self.minAreaContour:
                counter += 1
                return detected_object.DetectedObject(onlyObject, frame, depth, counter, hsv, contour)

    #пустой метод для заполнения аргуметна в createTrackbar
    def nothing(self, x):
        pass



