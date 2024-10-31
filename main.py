# 调用笔记本内置摄像头，参数为0
import cv2
import math
import serial.tools.list_ports
import mediapipe as mp
import numpy as np

hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils
ser = serial.Serial('COM3', 9600)


# 调用笔记本内置摄像头，参数为0


def findHind(img_0, hands_0, draw_0):
    imgRGB = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)  # 转换为RGB

    handleable = draw_0.DrawingSpec(color=(0, 0, 255), thickness=5)
    sandcastle = draw_0.DrawingSpec(color=(0, 255, 0), thickness=5)

    results = hands_0.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            draw_0.draw_landmarks(img_0, handLms, mp.solutions.hands.HAND_CONNECTIONS, handleable, sandcastle)

    return results.multi_hand_landmarks


def get_angleError(point_4, point_3, point_2, point_1):
    h, w, c = img.shape

    point_4_cx, point_4_cy = int(point_4.x * w), int(point_4.y * h)
    point_3_cx, point_3_cy = int(point_3.x * w), int(point_3.y * h)
    point_2_cx, point_2_cy = int(point_2.x * w), int(point_2.y * h)
    point_1_cx, point_1_cy = int(point_1.x * w), int(point_1.y * h)

    a = np.array([(point_4_cx - point_3_cx), (point_4_cy - point_3_cy)])
    b = np.array([(point_1_cx - point_2_cx), (point_1_cy - point_2_cy)])

    angle = math.degrees(math.acos((np.dot(a, b)) / (np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))))

    if angle > 150:
        isStraight = 1
    else:
        isStraight = 0

    return angle, isStraight


def detectNumber(hand_landmarks):
    myhand = hand_landmarks[0]
    isStraight_list = []

    point_4 = myhand.landmark[4]
    point_3 = myhand.landmark[3]
    point_2 = myhand.landmark[2]
    point_1 = myhand.landmark[1]
    angle_error_1, isStraight_1 = get_angleError(point_4, point_3, point_2, point_1)
    isStraight_list.append(isStraight_1)

    point_4 = myhand.landmark[8]
    point_3 = myhand.landmark[7]
    point_2 = myhand.landmark[6]
    point_1 = myhand.landmark[5]
    angle_error_2, isStraight_2 = get_angleError(point_4, point_3, point_2, point_1)
    isStraight_list.append(isStraight_2)

    point_4 = myhand.landmark[12]
    point_3 = myhand.landmark[11]
    point_2 = myhand.landmark[10]
    point_1 = myhand.landmark[9]
    angle_error_3, isStraight_3 = get_angleError(point_4, point_3, point_2, point_1)
    isStraight_list.append(isStraight_3)

    point_4 = myhand.landmark[16]
    point_3 = myhand.landmark[15]
    point_2 = myhand.landmark[14]
    point_1 = myhand.landmark[13]
    angle_error_4, isStraight_4 = get_angleError(point_4, point_3, point_2, point_1)
    isStraight_list.append(isStraight_4)

    point_4 = myhand.landmark[20]
    point_3 = myhand.landmark[19]
    point_2 = myhand.landmark[18]
    point_1 = myhand.landmark[17]
    angle_error_5, isStraight_5 = get_angleError(point_4, point_3, point_2, point_1)
    isStraight_list.append(isStraight_5)

    if isStraight_list[0] == 0 and isStraight_list[1] == 1 and isStraight_list[2] == 0 and isStraight_list[3] == 0 and \
            isStraight_list[4] == 0:
        return 1
    elif isStraight_list[0] == 0 and isStraight_list[1] == 1 and isStraight_list[2] == 1 and isStraight_list[3] == 0 \
            and isStraight_list[4] == 0:
        return 2
    elif isStraight_list[0] == 0 and isStraight_list[1] == 1 and isStraight_list[2] == 1 and isStraight_list[3] == 1 \
            and isStraight_list[4] == 0:
        return 3
    elif isStraight_list[0] == 0 and isStraight_list[1] == 1 and isStraight_list[2] == 1 and isStraight_list[3] == 1 \
            and isStraight_list[4] == 1:
        return 4
    elif isStraight_list[0] == 1 and isStraight_list[1] == 1 and isStraight_list[2] == 1 and isStraight_list[3] == 1 \
            and isStraight_list[4] == 1:
        return 5
    elif isStraight_list[0] == 1 and isStraight_list[1] == 0 and isStraight_list[2] == 0 and isStraight_list[3] == 0 \
            and isStraight_list[4] == 1:
        return 6
    elif isStraight_list[0] == 0 and isStraight_list[1] == 0 and isStraight_list[2] == 0 and isStraight_list[3] == 0 \
            and isStraight_list[4] == 0:
        return 0
    else:
        return -1


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # 从摄像头读取图片
    success, img = cap.read()
    if success:
        # 调用hands_landmarks函数找到手的特征
        hands_landmarks = findHind(img, hands, draw)
        if hands_landmarks:
            # 调用detectNumber函数
            resultNumber = detectNumber(hands_landmarks)
            if resultNumber >= 0:
                # 显示数字到视频窗口
                cv2.putText(img, str(resultNumber), (150, 150), 19, 5, (255, 0, 255), 5, cv2.LINE_AA)
                # 参数：视频窗口，显示内容，开始位置，字体，字体大小，字体颜色，画笔笔粗细，线条种类
                if resultNumber == 0:
                    ser.write(b'0')
                elif resultNumber == 1:
                    ser.write(b'1')
                elif resultNumber == 2:
                    ser.write(b'2')
                elif resultNumber == 3:
                    ser.write(b'3')
                elif resultNumber == 4:
                    ser.write(b'4')
                elif resultNumber == 5:
                    ser.write(b'5')
                elif resultNumber == 6:
                    ser.write(b'6')
            else:
                cv2.putText(img, "NO NUMBER", (150, 150), 20, 1, (0, 0, 255))
            # 显示摄像头
            cv2.imshow("Gesture Recognition", img)
    # 通过 esc 键退出摄像
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
# 关闭摄像头
cap.release()
