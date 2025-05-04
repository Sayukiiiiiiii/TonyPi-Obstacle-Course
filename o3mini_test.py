#!/usr/bin/python3
# coding=utf8
"""
仿真测试版 障碍跑方案示例代码 (无实机, 仅终端输出控制指令)
功能说明：
  1. 机器人通过摄像头识别红、蓝、黄三色障碍物（视为圆柱形障碍物, 在图像中呈矩形）。
  2. 根据障碍物在画面中的位置, 产生前进、转向等指令（只在终端输出对应信息）。
  3. 检测到3个障碍物后, 进入终点检测, 当检测到符合条件的白色终点线时, 输出相应指令。
  
注意：本代码仅进行视觉及状态机仿真, 调用“控制指令”时, 将直接在终端打印指令（模拟AGC.runActionGroup()逻辑）, 
可在无实机环境下进行测试；调试时请根据实际环境调整颜色阈值和其他参数。
"""

import sys
import cv2
import time
import math
import threading
import numpy as np

# 为仿真测试, 仅打印动作指令, 定义打印动作控制函数
def print_action(action_name):
    print("动作指令:", action_name)

#——————————————————————————————————
# 全局参数及状态变量设置

IMG_WIDTH = 640
IMG_HEIGHT = 480

# 设定两条用于判断的X轴阈值：
threshold_offset_in = 300   # 参考线偏移（右侧区域） 例如：300像素
threshold_offset_out = 100  # 外侧限制, 障碍物过于靠边时需转弯 例如：100像素

x_threshold_in = IMG_WIDTH - threshold_offset_in   # 内边界线
x_threshold_out = IMG_WIDTH - threshold_offset_out  # 外边界线

reverse = 1 # 边界反转标志（-1表示反向即从右边绕过障碍物）

tolerance = 15  # 容差

turn_limit = 1  # 每个障碍物转向计数阈值
reset_turn_interval = 20  # 定时重置转向计数的时间间隔（秒）

# 终点检测相关参数
threshold_k_judge = 0.6 # 终点直线斜率阈值（检测）
threshold_k_turn  = 0.1  # 对齐终点直线时的斜率调整阈值
end_t = 2  # 终点通过后继续前进时间（秒）

# 状态变量
isRunning = False           # 整体状态开关（仿真测试时通过键盘退出）
current_mode = "obstacle"   # "obstacle" 障碍避让模式； "finish" 终点检测模式
obstacle_count = 0          # 已绕过障碍数量
turn_cnt = 0                # 当前障碍转向累计次数
last_turn_time = time.time()  # 最近一次转向计数重置时间
squating = False #正在蹲下
squated = False #蹲下过
detected_end = False #检测到过终点线

undetected_count = 0 #未检测到终点线计数器
max_undetected_count = 10 #最大未检测到终点线计数器

target_colors = ['red', 'blue', 'yellow'] # 目标颜色列表

# 假设颜色检测相关阈值数据已经通过yaml加载, 此处构造模拟数据（LAB空间）
lab_data = {
    'red':   {'min': [0, 166, 21], 'max': [255, 255, 255]},
    'blue':  {'min': [0, 0, 0],  'max': [255, 182, 96]},
    'yellow':{'min': [154, 0, 148],'max': [255, 255, 255]},
    'white': {'min': [193, 0, 0],'max': [255, 255, 255]},
}

#——————————————————————————————————
# 初始化与复位函数（仅打印）
def initMove():
    print("初始化机器人姿态 -- 模拟设置初始关节角度")

def reset():
    global current_mode, obstacle_count, turn_cnt, last_turn_time
    current_mode = "obstacle"
    obstacle_count = 0
    turn_cnt = 0
    last_turn_time = time.time()

def init():
    print("障碍跑方案仿真测试 -- 初始化")
    initMove()

def start():
    global isRunning
    reset()
    isRunning = True
    print("障碍跑开始仿真测试")

def stop():
    global isRunning
    isRunning = False
    print("障碍跑仿真测试已停止")
    
def exit_app():
    global isRunning
    isRunning = False
    print("退出仿真测试, 机器人恢复站立状态")
    print_action('stand')


# 切换边界方向
def reverse_x_threshold():
    global  threshold_offset_in,threshold_offset_out,x_threshold_in,x_threshold_out, reverse
    reverse = -reverse
    if reverse is -1:
        x_threshold_in = threshold_offset_in   # 内边界线
        x_threshold_out = threshold_offset_out  # 外边界线
    else:   
        x_threshold_in = IMG_WIDTH - threshold_offset_in   # 内边界线
        x_threshold_out = IMG_WIDTH - threshold_offset_out  # 外边界线

#——————————————————————————————————
# 图像处理辅助函数

def getAreaMaxContour(contours):
    contour_area_max = 0
    area_max_contour = None
    for cnt in contours:
        area = math.fabs(cv2.contourArea(cnt))
        if area > contour_area_max and area >= 300:
            contour_area_max = area
            area_max_contour = cnt
    return area_max_contour, contour_area_max

# def detect_obstacle(frame):
#     """
#     检测红、蓝、黄三色障碍物, 选取轮廓面积最大的一个
#     返回 (color, center_x, center_y, area), 未检测到返回 (None, None, None, 0)
#     """
#     obstacle_info = (None, None, None, 0)
#     max_area = 0
#     detected_color = None
#     center_x = None
#     center_y = None

#     frame_resize = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
#     frame_gb = cv2.GaussianBlur(frame_resize, (3, 3), 3)
#     frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)

#     target_colors = ('red', 'blue', 'yellow')
#     for color in target_colors:
#         if color not in lab_data:
#             continue
        
#         lower = np.array(lab_data[color]['min'])
#         upper = np.array(lab_data[color]['max'])
#         mask = cv2.inRange(frame_lab, lower, upper)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         mask = cv2.erode(mask, kernel, iterations=1)
#         mask = cv2.dilate(mask, kernel, iterations=2)
#         # 限制ROI区域, 去除左右边缘噪声
#         # mask[:, 0:160] = 0
#         # mask[:, 480:640] = 0
        
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnt, area = getAreaMaxContour(contours)
#         if cnt is not None and area > max_area:
#             print(f"识别到颜色:{color}")
#             max_area = area
#             detected_color = color
#             rect = cv2.minAreaRect(cnt)
#             box = np.int0(cv2.boxPoints(rect))
#             # 计算矩形对角中心点
#             pt1, pt3 = box[0], box[2]
#             center_x = int((pt1[0] + pt3[0]) / 2)
#             center_y = int((pt1[1] + pt3[1]) / 2)
#             obstacle_info = (detected_color, center_x, center_y, area)
#     return obstacle_info

def detect_obstacle(frame):
    """
    检测红、蓝、黄三色障碍物, 选取轮廓面积最大的一个，同时要求轮廓必须满足矩形近似条件。
    返回 (color, center_x, center_y, area)；未检测到返回 (None, None, None, 0)
    """
    global target_colors

    obstacle_info = (None, None, None, 0)
    max_area = 0
    detected_color = None
    center_x = None
    center_y = None

    frame_resize = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    frame_gb = cv2.GaussianBlur(frame_resize, (3, 3), 3)
    frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)

    #target_colors = ('red', 'blue', 'yellow')
    for color in target_colors:
        if color not in lab_data:
            continue

        lower = np.array(lab_data[color]['min'])
        upper = np.array(lab_data[color]['max'])
        mask = cv2.inRange(frame_lab, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt, area = getAreaMaxContour(contours)
        if cnt is not None and area > max_area:
            # 计算轮廓周长并近似多边形
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # 只有当近似多边形有4个顶点时，认为检测到矩形
            if len(approx) >= 4 and len(approx) <= 6: 
                print(f"识别到颜色: {color}，且为矩形")
                max_area = area
                detected_color = color
                # 计算最小外接矩形，并取对角线中点作为中心
                rect = cv2.minAreaRect(cnt)
                box = np.int0(cv2.boxPoints(rect))
                pt1, pt3 = box[0], box[2]
                center_x = int((pt1[0] + pt3[0]) / 2)
                center_y = int((pt1[1] + pt3[1]) / 2)
                obstacle_info = (detected_color, center_x, center_y, area)
    return obstacle_info


def detect_finish_line(frame):
    """
    通过二值化提取亮区(白色)并拟合直线, 判断终点线
    若直线斜率 |k| < threshold_k_judge, 则认为检测到终点线, 返回 (True, k)
    否则返回 (False, k)
    """
    frame_resize = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (False, None, None, None)
    cnt, _ = getAreaMaxContour(contours)
    if cnt is None:
        return (False, None, None, None)
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    if vx == 0:
        k = float('inf')
    else:
        k = vy / vx
    lefty = int((-x0 * vy/vx) + y0)
    righty = int(((IMG_WIDTH - x0) * vy/vx) + y0)
    
    if abs(k) < threshold_k_judge:
        return (True, k, lefty, righty)
    else:
        return (False, k, lefty, righty)

#——————————————————————————————————
# 主控制（状态机）函数
def process_frame(frame):
    """
    根据当前状态(避障/终点检测)处理图像, 并输出对应动作指令（打印到终端）
    同时在画面上绘制辅助信息, 便于调试
    """
    global current_mode, turn_cnt, last_turn_time, obstacle_count, isRunning, squated, squating,\
            detected_end, x_threshold_in, x_threshold_out, reverse, undetected_count, max_undetected_count
    disp_frame = frame.copy()
    current_time = time.time()

    if not isRunning:
        return disp_frame
    
    color = None

    #print(f"当前状态： {current_mode}")
    if current_mode == "obstacle":
        # 检测障碍物并获取信息（color, center_x, center_y, area）
        cv2.line(disp_frame, (x_threshold_in, 0), (x_threshold_in, IMG_HEIGHT), (0,255,0), 6)
        cv2.line(disp_frame, (x_threshold_out, 0), (x_threshold_out, IMG_HEIGHT), (0,255,0), 6)
        obst_color, center_x, center_y, area = detect_obstacle(frame)
        if obst_color is not None:
            color = obst_color
            # 绘制障碍物中心和参考线
            cv2.circle(disp_frame, (center_x, center_y), 6, (255,255,255), -1)
            # 计算误差：障碍物中心与参照线的横向距离
            error = center_x - x_threshold_in
            if reverse * x_threshold_in < reverse * center_x < reverse * x_threshold_out:
                print_action("go_forward")
            elif - reverse * error >= tolerance:
                if reverse is 1:
                    print_action("turn_right")
                else:
                    print_action("turn_left")
            # 当障碍物中心过于靠近内侧边缘时, 切换到转弯模式
            if reverse * (center_x - x_threshold_out) >= tolerance:
                current_mode = "turn"
                print("进入转弯模式")
        else:
            # 未检测到障碍物时, 继续前进
            print("没找到障碍物噢！")
            print_action("stop")
        
    #—————————————————— 转弯状态处理
    elif current_mode == "turn":
        # 检测障碍物并获取信息（color, center_x, center_y, area）
        cv2.line(disp_frame, (x_threshold_in, 0), (x_threshold_in, IMG_HEIGHT), (0,255,0), 6)
        cv2.line(disp_frame, (x_threshold_out, 0), (x_threshold_out, IMG_HEIGHT), (0,255,0), 6)
        obst_color, center_x, center_y, area = detect_obstacle(frame)
        if obst_color is not None:
            color = obst_color
            # 绘制障碍物中心和参考线
            cv2.circle(disp_frame, (center_x, center_y), 6, (255,255,255), -1)
        if obst_color is not None and abs(center_x - x_threshold_in) <= tolerance:
            turn_cnt += 1
            print("完成一次转向, turn count:", turn_cnt)
            current_mode = "obstacle"
            last_turn_time = current_time
        else:
            error = center_x - x_threshold_in if center_x is not None else 0
            if error is 0 or reverse * error >= tolerance:
                if reverse is 1:
                    print_action("turn_left")
                else:
                    print_action("turn_right")               

    
    elif current_mode == "finish":
        detected, k, left, right = detect_finish_line(frame)
        if not detected:
            undetected_count += 1
        else:
            undetected_count = 0  # 重置计数器

        if undetected_count >= max_undetected_count:
            if not detected_end:
                if not squated and not squating:
                    print("终点线未检测到, 执行下蹲动作")
                    print_action("squat")
                    squated = True
                    squating = True
                elif squated and squating:
                    print("即使蹲下了也找不到")
            else:
                print("走过了可见终点线范围, 继续直行")
                print_action("go_forward")
                time.sleep(end_t)
                stop()
                print_action("chest")  # 模拟庆祝
               
        else:
            if detected:
                detected_end = True
                cv2.line(disp_frame, (IMG_WIDTH-1, right), (0, left), (255,0,0), 10)
                cv2.putText(disp_frame, "Finish line detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if abs(k) > threshold_k_turn:
                    print("检测到了终点线，调整方向")
                    if k > 0:
                        print_action("turn_left")
                    else:
                        print_action("turn_right")
                else:
                    print("检测到了终点线，直行中")
                    if squating is True:
                        print_action("stand")
                        squating = False
                    print_action("go_forward")
                # elif abs(k) < threshold_k_judge:
                #     if squating is True:
                #         print_action("stand")
                #         squating = False
                #     print_action("go_forward")
            

    # 定时重置转向计数
    if (current_time - last_turn_time) > reset_turn_interval:
        turn_cnt = 0

    # 判断是否绕过当前障碍, 若转向次数超过turn_limit, 则认为完成绕行
    print("已绕过障碍数量:", obstacle_count)
    if turn_cnt > turn_limit:
        obstacle_count += 1
        reverse_x_threshold()
        turn_cnt = 0
        current_mode = "obstacle"
        last_turn_time = current_time
        if color is not None:
            print(f"绕过障碍物: {color}, 已绕过障碍数量: {obstacle_count}")
            target_colors.remove(color)  # 移除已绕过的颜色

    # 当累计三个障碍后, 切换至终点检测模式
    if obstacle_count >= 3:
        current_mode = "finish"
        print("进入终点检测模式")

    return disp_frame

#——————————————————————————————————
# 主函数
if __name__ == '__main__':
    # 导入相机标定参数（如果有标定文件, 在此修改路径；否则直接使用摄像头原始图像）
    # 模拟用代码, 直接读取摄像头图像
    cap = cv2.VideoCapture(0)  # 根据实际情况选择摄像头, 若使用其他设备请修改
    
    if not cap.isOpened():
        print("摄像头未打开")
        sys.exit(0)
    
    init()
    start()
    print_action("stand")  # 初始站立动作（打印输出）
    
    while True:
        ret, frame = cap.read()
        if ret:
            disp = process_frame(frame)
            cv2.imshow("Obstacle Run Simulation", disp)
            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break
        else:
            time.sleep(0.01)
    
    cap.release()
    cv2.destroyAllWindows()
    exit_app()
