import cv2 as cv
import numpy as np
import sqlite3 as sql
from datetime import datetime

class Features():
    kernel = np.ones((5, 5), np.uint8)
    amount = int(input("Enter the number of objects: "))
    depth = 40
    count_to_exit = 0
    info = []
    cap = cv.VideoCapture(1, cv.CAP_DSHOW)
    def __init__(self):

        cv.namedWindow("frame")
        cv.namedWindow("track", cv.WINDOW_NORMAL)

        cv.createTrackbar("H", "track", 0, 179, Features.nothing)
        cv.createTrackbar("S", "track", 0, 255, Features.nothing)
        cv.createTrackbar("V", "track", 0, 255, Features.nothing)
        cv.createTrackbar("HL", "track", 0, 179, Features.nothing)
        cv.createTrackbar("SL", "track", 0, 255, Features.nothing)
        cv.createTrackbar("VL", "track", 0, 255, Features.nothing)
        cv.createTrackbar("T1", "track", 0, 255, Features.nothing)
        cv.createTrackbar("T2", "track", 0, 255, Features.nothing)

    @classmethod
    def get_contours(cls, only_object, thresh1, thresh2, kernel, frame, depth):
        gray = cv.cvtColor(only_object, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, thresh1, thresh2)
        dil = cv.dilate(canny, kernel, iterations=1)
        frame = cv.putText(frame, f"Distance: {depth} cm", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        contours, hierarchy = cv.findContours(dil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        counter = 0
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 5000:
                detected_object = Detected_object(only_object, frame, depth, counter, hsv, contour)

    @classmethod
    def nothing(cls, x):
        pass


class Detected_object(Features):
    def __init__(self, only_object, frame, depth, counter, hsv, contour):
        cv.drawContours(only_object, contour, -1, (200, 200, 0), 3)
        cv.drawContours(frame, contour, -1, (200, 200, 0), 3)
        counter += 1
        p = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * p, True)
        x, y, w, h = cv.boundingRect(approx)
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame = cv.circle(frame, (x + (w // 2), y + (h // 2)), 5, (0, 255, 0), -1)
        frame = cv.arrowedLine(frame, (x, y + h + 25), (x + w, y + h + 25), (0, 255, 0), 2)
        frame = cv.arrowedLine(frame, (x + w, y + h + 25), (x, y + h + 25), (0, 255, 0), 2)
        frame = cv.arrowedLine(frame, (x - 25, y), (x - 25, y + h), (0, 255, 0), 2)
        frame = cv.arrowedLine(frame, (x - 25, y + h), (x - 25, y), (0, 255, 0), 2)
        frame = cv.putText(frame, f"{round(w / 25)} cm", ((x + w // 3, y + h + 50)), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        frame = cv.putText(frame, f"{round(h / 25)} cm", ((x - 120, y + h // 2)), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        try:
            real_width = round(w / 25)
            real_height = round(h / 25)
            f = (w * depth) // real_width
            K = round(real_width / real_height, 1)
            cx = int(x + w // 2)
            cy = int(y + h // w)
        except ZeroDivisionError:
            print("divided by 0")
        color_point = hsv[cy, cx]
        color = self.check_color(color_point[0])
        cv.drawContours(frame, approx, -1, (0, 0, 255), 3)
        cv.putText(frame, f"x:{x + (w // 2)}, y: {y + (h // 2)}", (x, y - 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv.putText(frame, f"Points: {len(approx)} | Color: {color}", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        Features.info.append((len(approx), round(real_width, 1), round(real_height, 1), K, color, f))
        if len(Features.info) > 100:
            del Features.info[0]

        if counter != 0:
            cv.putText(frame, f"Objects: {counter}", (10, 65), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        print(Features.info)

    def info_analysis(self, info, lower, upper):
        name = input("Enter the name of the object: ")

        approx = []
        real_width = []
        real_height = []
        K = []
        color = []
        f = []
        result = []

        for i in range(len(info)):
            approx.append(info[i][0])
            real_width.append(info[i][1])
            real_height.append(info[i][2])
            K.append(info[i][3])
            color.append(info[i][4])
            f.append(info[i][5])

        main_list = [approx, real_width, real_height, K, color, f]
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

        with sql.connect('objects.db') as con:
            cur = con.cursor()
            cur.execute("""INSERT INTO objects(name, color, width, height, K,
            angles, h, s, v, hl, sl, vl, f, time)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (name, result[4], result[1], result[2],
            result[3], result[0], f"{upper[0]}", f"{upper[1]}", f"{upper[2]}",
            f"{lower[0]}", f"{lower[1]}", f"{lower[2]}", result[5], current_time))

    def check_color(self, hue):
        color = "Undeafined"
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
            color = "Red"

        return color

