# -*- coding: utf-8 -*-
import random
import time 
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from turtle import *
from time import ctime,sleep
#import threading


delay(delay=None)
color('red', 'blue')
cap = cv2.VideoCapture(0)  # 或传入0，使用摄像头

 
def getpicture():
    # 读取一帧
    _, frame = cap.read()
    # 把 BGR 转为 HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV中蓝色范围
    lower_blue = np.array([100,90,99])
    upper_blue = np.array([124,255,255])

    # 获得黑色区域的mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.blur(mask,(10,10))
    # 和原始图片进行and操作，获得黑色区域
    res = cv2.bitwise_and(frame,frame, mask= mask)
    res = cv2.blur(res,(10,10))
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.imshow('mask',mask)
    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.imshow('res',res)
    '''k = cv2.waitKey(5) & 0xFF
    if k == 27:
        sign = 1
    else:
        sign = 0'''
    return res
def getlocation(mask):
    location = np.where(mask>150)
    if len(location[0])!= 0:
      x = location[0][0]
      y = location[1][0]
      goto(-y+100,-x+100)
while True:
    mask = getpicture()
    getlocation(mask)
    time.sleep(0.01)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
done()
