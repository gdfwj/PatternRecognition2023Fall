import os
import cv2
import numpy as np
from mtcnn import MTCNN
from dataset import get_all

# 读取图像文件夹路径
input_dir = '../data'
# 保存截取人脸的图像文件夹路径
output_dir = '../cropped_data'

# 创建保存截取人脸的图像文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 初始化MTCNN模型
detector = MTCNN()

dataset = get_all()

# 遍历图像文件夹中的所有图像
for path in dataset.getpath():
    # 读取图像
    img = cv2.imread(path)
    # 检测人脸
    faces = detector.detect_faces(img)
    # 遍历检测到的人脸
    for face in faces:
        # 获取人脸坐标
        x1, y1, width, height = face['box']
        # 截取对齐人脸
        cropped = img[y1:y1+height, x1:x1+width]

        out_path = "..\\cropped_data\\"+path[8:]
        print(out_path)
        outdir = out_path[:out_path.rfind('\\')]
        # print(outdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        cv2.imwrite(out_path, cropped)
        # exit(1)
