"""
面部表情识别-视频
"""
import cv2
from model import FacialExpressionModel
import dlib
import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class face_emotion():
    def __init__(self):
        # 使用特征提取器get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # 加载模型
        self.model = FacialExpressionModel("../weights/model.json", "../weights/model_weights.h5")

        # 建cv2摄像头对象，这里使用电脑自带摄像头，如果接了外部摄像头，则自动切换到外部摄像头 视频路径则调用视频
        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture("../sample/dai.mp4")
        # 设置视频参数，propId设置的视频参数，value设置的参数值
        self.cap.set(3, 480)
        # 截图screenshoot的计数器
        self.cnt = 0

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    def learning_face(self):
        # cap.isOpened（） 返回true/false 检查初始化是否成功
        while (self.cap.isOpened()):
            # cap.read()
            # 返回两个值：
            # 一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
            # 图像对象，图像的三维矩阵
            flag, im_rd = self.cap.read()

            # 每帧数据延时1ms，延时为0读取的是静态帧
            k = cv2.waitKey(1)

            # 取灰度
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

            # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数dets
            dets = self.detector(img_gray, 0)

            # 待会要显示在屏幕上的字体
            font = cv2.FONT_HERSHEY_SIMPLEX

            for d in dets:
                face_area = img_gray[d.top():d.bottom(), d.left():d.right()]
                roi = cv2.resize(face_area, (48, 48))
                pred = self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])  # 得到标签
                # print(pred)
                cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 3)
                # x_put = d.left() + (d.right()-d.left())/2
                cv2.putText(im_rd, pred, (d.left(), d.top()), font, 1, (255, 255, 255), 2)

            # 按下s键截图保存
            if (k == ord('s')):
                self.cnt += 1
                cv2.imwrite("screenshoot" + str(self.cnt) + ".jpg", im_rd)

            # 按下q键退出
            if (k == ord('q')):
                break

            # 窗口显示
            cv2.imshow("camera", im_rd)

        # 释放摄像头
        self.cap.release()
        # 删除建立的窗口
        cv2.destroyAllWindows()


if __name__ == "__main__":
    my_face = face_emotion()
    my_face.learning_face()
