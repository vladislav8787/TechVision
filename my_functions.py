import cv2 as cv
import numpy as np
import sqlite3 as sql
from datetime import datetime

class Features():

    def __init__(self):
        #ядро, кол-во объектов, расстояние для определения параметров, счетчик,
        #объект захвата видеопотока с камеры 1
        self.kernel = np.ones((5, 5), np.uint8)
        self.amount = int(input("Enter the number of objects: "))
        self.depth = 40
        self.count_to_exit = 0
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
    def get_contours(self, only_object, thresh1, thresh2, kernel, frame, depth):
        gray = cv.cvtColor(only_object, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, thresh1, thresh2)
        dil = cv.dilate(canny, kernel, iterations=1)
        frame = cv.putText(frame, f"Distance: {depth} cm", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        contours, hierarchy = cv.findContours(dil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        counter = 0
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 2500:
                counter += 1
                return Detected_object(only_object, frame, depth, counter, hsv, contour)

    #пустой метод для заполнения аргуметна в createTrackbar
    def nothing(self, x):
        pass


class Detected_object():
    def __init__(self, only_object, frame, depth, counter, hsv, contour):
        #данные об объекте
        self.info = []
        self.counter = counter + 1
        self.p = cv.arcLength(contour, True)
        self.approx = cv.approxPolyDP(contour, 0.02 * self.p, True)
        self.x, self.y, self.w, self.h = cv.boundingRect(self.approx)
        #отображение контуров и доп информации на кадрах
        cv.drawContours(only_object, contour, -1, (200, 200, 0), 3)
        cv.drawContours(frame, contour, -1, (200, 200, 0), 3)
        frame = cv.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
        frame = cv.circle(frame, (self.x + (self.w // 2), self.y + (self.h // 2)), 5, (0, 255, 0), -1)
        frame = cv.arrowedLine(frame, (self.x, self.y + self.h + 25), (self.x + self.w, self.y + self.h + 25), (0, 255, 0), 2)
        frame = cv.arrowedLine(frame, (self.x + self.w, self.y + self.h + 25), (self.x, self.y + self.h + 25), (0, 255, 0), 2)
        frame = cv.arrowedLine(frame, (self.x - 25, self.y), (self.x - 25, self.y + self.h), (0, 255, 0), 2)
        frame = cv.arrowedLine(frame, (self.x - 25, self.y + self.h), (self.x - 25, self.y), (0, 255, 0), 2)
        frame = cv.putText(frame, f"{round(self.w / 25)} cm", (self.x + self.w // 3, self.y + self.h + 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        frame = cv.putText(frame, f"{round(self.h / 25)} cm", (self.x - 120, self.y + self.h // 2), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        #параметры объекта
        try:
            self.real_width = round(self.w / 25)
            self.real_height = round(self.h / 25)
            self.f = (self.w * depth) // self.real_width
            self.K = round(self.real_width / self.real_height, 1)
            self.cx = int(self.x + self.w // 2)
            self.cy = int(self.y + self.h // self.w)
        except ZeroDivisionError:
            print("divided by 0")
        self.color_point = hsv[self.cy, self.cx]
        self.color = self.check_color(self.color_point[0])
        cv.drawContours(frame, self.approx, -1, (0, 0, 255), 3)
        cv.putText(frame, f"x:{self.x + (self.w // 2)}, y: {self.y + (self.h // 2)}", (self.x, self.y - 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv.putText(frame, f"Points: {len(self.approx)} | Color: {self.color}", (self.x, self.y - 5), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        self.info.append((len(self.approx), round(self.real_width, 1), round(self.real_height, 1), self.K, self.color, self.f))
        if len(self.info) > 100:
            del self.info[0]

        if self.counter != 0:
            cv.putText(frame, f"Objects: {counter}", (10, 65), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    #поиск наиболее встречающихся значений
    def info_analysis(self, lower, upper):
        name = input("Enter the name of the object: ")

        approx = [self.info[i][0] for i in range(len(self.info))]
        real_width = [self.info[i][1] for i in range(len(self.info))]
        real_height = [self.info[i][2] for i in range(len(self.info))]
        K = [self.info[i][3] for i in range(len(self.info))]
        color = [self.info[i][4] for i in range(len(self.info))]
        f = [self.info[i][5] for i in range(len(self.info))]

        main_list = [approx, real_width, real_height, K, color, f]
        result = []
        for parameter in main_list:
            max = index = 0
            for i in range(len(parameter)):
                param_counter = parameter.count(parameter[i])
                if param_counter > max:
                    max = param_counter
                    index = i
            try:
                if parameter[index] == parameter[0]:
                    parameter[index] = parameter[len(parameter) // 2]
            except Exception:
                print("don't have any values")
            result.append(parameter[index])

        current_time = str(datetime.now().hour) + ":" + str(datetime.now().minute) + ":" + str(datetime.now().second)
        #отправить в бд
        with sql.connect('objects.db') as con:
            cur = con.cursor()
            cur.execute("""INSERT INTO objects(name, color, width, height, K,
            angles, h, s, v, hl, sl, vl, f, time)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (name, result[4], result[1], result[2],
            result[3], result[0], f"{upper[0]}", f"{upper[1]}", f"{upper[2]}",
            f"{lower[0]}", f"{lower[1]}", f"{lower[2]}", result[5], current_time))

    @classmethod
    def recognize_object(cls, only_object, thresh1, thresh2, kernel, frame, object):
        gray = cv.cvtColor(only_object, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, thresh1, thresh2)
        dil = cv.dilate(canny, kernel, iterations=1)
        contours, hierarchy = cv.findContours(dil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        counter = 0

        for contour in contours:
            area = cv.contourArea(contour)
            if area > 2500:
                p = cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, 0.02 * p, True)
                x, y, w, h = cv.boundingRect(approx)

                try:
                    real_width = round(w / 25)
                    real_height = round(h / 25)
                    K = round(real_width / real_height, 1)
                    K2 = round(real_height / real_width, 1)
                except ZeroDivisionError:
                    print("divided by 0")

                if (object[4] - 0.2 <= K <= object[4] + 0.2 or object[4] - 0.2 <= K2 <= object[4] + 0.2) and len(approx) == object[5]:
                    cv.drawContours(only_object, contour, -1, (200, 200, 0), 3)
                    cv.drawContours(frame, contour, -1, (200, 200, 0), 3)
                    counter += 1
                    frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    frame = cv.circle(frame, (x + (w // 2), y + (h // 2)), 5, (0, 255, 0), -1)
                    #сравнение значений коэффициентов для придачи инвариантности
                    if (object[4] - 0.2 <= K <= object[4] + 0.2) and len(approx) == object[5]:
                        Detected_object.display(frame, object[2], object[3], x, y, w, h, object[12], object[0])

                    elif (object[4] - 0.2 <= K2 <= object[4] + 0.2) and len(approx) == object[5]:
                        Detected_object.display(frame, object[3], object[2], x, y, w, h, object[12], object[0])

        if counter != 0:
            cv.putText(frame, f"Objects: {counter}", (10, 65), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    #отображение данных в кадре
    @classmethod
    def display(cls, frame, hight, width, x, y, w, h, f, amount):
        distance = (hight * f) / w
        frame = cv.arrowedLine(frame, (x, y + h + 25), (x + w, y + h + 25), (0, 255, 0), 2)
        frame = cv.arrowedLine(frame, (x + w, y + h + 25), (x, y + h + 25), (0, 255, 0), 2)
        frame = cv.arrowedLine(frame, (x - 25, y), (x - 25, y + h), (0, 255, 0), 2)
        frame = cv.arrowedLine(frame, (x - 25, y + h), (x - 25, y), (0, 255, 0), 2)
        frame = cv.putText(frame, f"{hight} cm", ((x + w // 3, y + h + 50)), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        frame = cv.putText(frame, f"{width} cm", ((x - 120, y + h // 2)), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        frame = cv.putText(frame, f"{amount}", (x, y - 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        frame = cv.putText(frame, f"Distance {round(distance)} cm", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

   #определение цвета по значению hue
    def check_color(self, hue):

        if hue <= 6:
            color = "Red"
        elif hue <= 15:
            color = "Orange"
        elif hue <= 33:
            color = "Yellow"
        elif hue <= 78:
            color = "Green"
        elif hue <= 131:
            color = "Blue"
        elif hue <= 167:
            color = "Purple"
        else:
            color = "Undefined"

        return color

