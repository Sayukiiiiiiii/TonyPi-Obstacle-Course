#!/usr/bin/python3
# coding=utf8
"""
人形机器人障碍跑方案示例代码
功能说明：
  1. 机器人通过摄像头识别红、蓝、黄三种颜色障碍物（假设为圆柱体, 在视野中表现为一个矩形）。
  2. 控制策略：机器人先检测进入视野内的第一个障碍（若有多个则选择面积最大或者画面中最靠前的）, 
     然后以画面右侧一条 "阈值线" 为参考, 调整朝向使障碍物中心点保持在该线上；
  3. 当障碍物中心接近图像边缘时（即超过内侧阈值）, 机器人停止直行进入转弯阶段, 转向直至障碍物中心回到参考线上, 
     每次转向计数, 若在一定时间内累计转向次数超过预设 "turn_limit" , 则当前障碍物视为已绕过, 
     如果视野中出现下一个障碍, 则更换参考线重复；累计三次障碍后, 开始终点检测——
     当发现白色终点线且其斜率满足设定条件, 则调整朝向, 直行通过终点。
────────────────────────────
注意：本代码基于 OpenCV 及机器人动作库进行实现, 需在相应环境下运行。
"""

import sys
import cv2
import time
import math
import threading
import numpy as np

# 导入Hiwonder相关模块（需根据实际情况调整）
import hiwonder.Misc as Misc
import hiwonder.Board as Board
import hiwonder.Camera as Camera
import hiwonder.ActionGroupControl as AGC
import hiwonder.yaml_handle as yaml_handle

print('add-filter')
#——————————————————————————————————

# 调试模式(True时不会执行实际动作)
DEBUG = False

# 打印动作指令，定义打印动作控制函数
def print_action(action_name):
    print("动作指令:", action_name)

# 全局参数及状态变量设置
# 图像尺寸（一般摄像头输出尺寸）
IMG_WIDTH = 640
IMG_HEIGHT = 480

# 设定两条用于判断的X轴阈值：
threshold_offset_in = 300  # 可调参数
threshold_offset_out = 100   # 可调参数
x_threshold_in = IMG_WIDTH - threshold_offset_in   # 内边界线
x_threshold_out = IMG_WIDTH - threshold_offset_out  # 外边界线

reverse = 1 # 边界反转标志（-1表示反向即从右边绕过障碍物）

# 控制误差容差（像素）, 如果障碍物中心在参考线上 ±tolerance, 我们认为对齐
tolerance = 15

turn_limit = 3  # 每个障碍物转向计数阈值
reset_turn_interval = 20  # 定时重置转向计数的时间间隔（秒）

# 终点检测相关
# 检测终点线时, 赛道背景为绿色, 终点为白色, 另外赛道边缘也有白线, 
# 故需要检测出直线后判断其斜率是否平缓。两参数：
threshold_k_judge = 0.6  # 检测终点线时要求直线斜率 |k| < threshold_k_judge
threshold_k_turn  = 0.1  # 机器人转向微调结束条件： |k| < threshold_k_turn
# 终点通过后继续前进时间（秒）
end_t = 2

# 状态变量
isRunning = False           # 总开关
current_mode = "obstacle"   # 两种模式： "obstacle"（障碍绕行）和 "finish"（终点检测通过）
obstacle_count = 0          # 已绕过障碍数量
turn_cnt = 0                # 本次绕障转向计数
last_turn_time = time.time()  # 用于定时重置转向计数

squating = False            #正在蹲下
squated = False             #蹲下过
detected_end = False        #检测到过终点线

undetected_count = 0 #未检测到终点线计数器
max_undetected_count = 10 #最大未检测到终点线计数器

target_colors = ['red', 'blue', 'yellow'] # 目标颜色列表

# 用于颜色数据（通过yaml加载, 一般包含LAB空间的min/max）；
# lab_data中应包含"red", "blue", "yellow", "white"等设定；
lab_data = None
servo_data = None

# 加载颜色参数
def load_config():
    global lab_data, servo_data
    lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)
    servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)

load_config()

#——————————————————————————————————
# 初始化与复位函数

def initMove():
    # 机器人初始位置
    Board.setPWMServoPulse(1, servo_data['servo1'], 500)
    Board.setPWMServoPulse(2, servo_data['servo2'], 500)

def reset():
    global current_mode, obstacle_count, turn_cnt, last_turn_time
    current_mode = "obstacle"
    obstacle_count = 0
    turn_cnt = 0
    last_turn_time = time.time()

def init():
    print("障碍跑方案初始化")
    load_config()
    initMove()

def start():
    global isRunning
    reset()
    isRunning = True
    print("障碍跑开始")

def stop():
    global isRunning
    isRunning = False
    print("障碍跑停止")
    
def exit_app():
    global isRunning
    isRunning = False
    if DEBUG is False: AGC.runActionGroup('stand')
    print_action('stand')
    print("障碍跑退出")

# 切换边界方向（绕行方向
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
    """
    给定轮廓列表, 返回面积最大的有效轮廓及其面积
    """
    contour_area_max = 0
    area_max_contour = None
    for cnt in contours:
        area = math.fabs(cv2.contourArea(cnt))
        # 根据面积大小过滤噪声, 面积阈值可根据环境调整
        if area > contour_area_max and area >= 300:
            contour_area_max = area
            area_max_contour = cnt
    return area_max_contour, contour_area_max

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
    检测终点线：终点线为白色, 与绿色跑道对比明显, 但赛道边缘也有白线, 
    故需借助形状（直线）的斜率进行判断：
      若检测到白色区域, 通过轮廓拟合或Hough直线获得直线斜率k, 若 |k| < threshold_k_judge,
      则认为检测到终点线。
    返回：(detected, k) , 如果未检测到, 则 detected==False
    """
    # 将图像预处理：这里直接工作在BGR空间中寻找较亮的区域
    frame_resize = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
    # 二值化：根据场上光线可调, 阈值可根据实际现场调试
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (False, None, None, None)
    # 选取面积最大的白色区域
    cnt, area = getAreaMaxContour(contours)
    if cnt is None:
        return (False, None, None, None)
    # 拟合直线
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    # 斜率 k = vy/vx
    if vx == 0:
        k = float('inf')
    else:
        k = vy/vx
    # 画出拟合直线用于调试
    lefty = int((-x0*vy/vx) + y0)
    righty = int(((IMG_WIDTH - x0)*vy/vx)+ y0)
    #cv2.line(frame_resize,(IMG_WIDTH-1,righty),(0,lefty),(255,0,0),2)
    # 根据斜率判断
    if abs(k) < threshold_k_judge:
        return (True, k, lefty, righty)
    else:
        return (True, k, lefty, righty)

#——————————————————————————————————
# 主控制函数（状态机逻辑）
#
# 根据当前模式（障碍避免/终点检测）, 结合摄像头传入的图像, 决定调用相应运动指令。

def process_frame(frame):
    """
    主处理函数, 每帧图像处理后返回画面（便于在显示窗口中调试）
    并根据状态机调用动作（例如前进、调整角度、转弯等）
    """
    global current_mode, turn_cnt, last_turn_time, obstacle_count, isRunning, squated, squating,\
            detected_end, x_threshold_in, x_threshold_out, reverse, undetected_count, max_undetected_count
    
    disp_frame = frame.copy()
    current_time = time.time()

    # 若未启动则不处理
    if not isRunning:
        return disp_frame
    
    color = None

    # 先判断模式：障碍物避让模式或终点检测模式
    if current_mode == "obstacle":
        # 绘制目标矩形参考线
        cv2.line(disp_frame, (x_threshold_in, 0), (x_threshold_in, IMG_HEIGHT), (0,255,0), 2)
        cv2.line(disp_frame, (x_threshold_out, 0), (x_threshold_out, IMG_HEIGHT), (0,255,0), 2)
        # 检测障碍物（红/蓝/黄三色）
        obst_color, center_x, center_y, area = detect_obstacle(frame)
        if obst_color is not None:
            color = obst_color
            # 绘制检测到的障碍物中心
            cv2.circle(disp_frame, (center_x, center_y), 6, (0,0,255), -1)
            # 如果处于 "直行保持" 状态（当前模式 "obstacle" 中）
            # 计算与参考线的误差
            error = center_x - x_threshold_in

            # 如果误差较小（障碍物在参考线上）, 继续直行
            if reverse * x_threshold_in < reverse * center_x < reverse * x_threshold_out:
                if DEBUG is False: AGC.runActionGroup('go_forward')
                print_action("go_forward")
            elif - reverse * error >= tolerance:
                if reverse is 1:
                    if DEBUG is False: AGC.runActionGroup('turn_right')
                    print_action("turn_right")
                else:
                    if DEBUG is False: AGC.runActionGroup('turn_left')
                    print_action("turn_left")
            # 当障碍物中心 "过早" 接近画面外侧边缘，则认为需要回转
            if (center_x - x_threshold_out) >= tolerance:
                current_mode = "turn"
                print("切换到回转模式")
        else:
            # 若没有检测到障碍物
            # AGC.runActionGroup('go_forward') # 前进
            print("没找到障碍物噢！")
            if DEBUG is False: AGC.runActionGroup("stop")
            print_action("stop")
        
    #—————— 处理回转状态 ——————
    elif current_mode == "turn":
         # 绘制目标矩形参考线
        cv2.line(disp_frame, (x_threshold_in, 0), (x_threshold_in, IMG_HEIGHT), (0,255,0), 2)
        cv2.line(disp_frame, (x_threshold_out, 0), (x_threshold_out, IMG_HEIGHT), (0,255,0), 2)
        obst_color, center_x, center_y, area = detect_obstacle(frame)
        # 在转弯阶段, 机器人转向直到障碍的中心再次到达参考线上
        if obst_color is not None and abs(center_x - x_threshold_in) <= tolerance:
            color = obst_color
            # 绘制障碍物中心和参考线
            cv2.circle(disp_frame, (center_x, center_y), 6, (255,255,255), -1)
            # 转弯完成一次, 计数
            turn_cnt += 1
            print("完成一次回转, 计数：", turn_cnt)
            current_mode = "obstacle"  # 切回直行模式
            last_turn_time = current_time  # 记录转弯完成时间
        else:
            # 继续调整转向方向。选择与上面类似的策略：
            error = center_x - x_threshold_in if center_x is not None else 0
            if error is 0 or reverse * error >= tolerance:
                if reverse is 1:
                    if DEBUG is False: AGC.runActionGroup('turn_left')
                    print_action("turn_left")
                else:
                    if DEBUG is False: AGC.runActionGroup('turn_right')
                    print_action("turn_right")  


    #—————— 处理检测终点状态 ——————
    elif current_mode == "finish":
        # 终点检测流程：检测终点白线, 并判断其平缓性
        detected, k, left, right = detect_finish_line(frame)

        if not detected:
            undetected_count += 1
        else:
            undetected_count = 0  # 重置计数器
        
        # 未检测到终点线的帧数过多
        if undetected_count >= max_undetected_count:
            if not detected_end: # 没有检测到过终点线
                if not squated and not squating:
                    print("终点线未检测到, 执行下蹲动作")
                    if DEBUG is False: AGC.runActionGroup('squat')
                    print_action("squat")
                    squated = True
                    squating = True
                elif squated and squating:
                    print("即使蹲下了也找不到") 
            else: # 检测到过终点线
                print("走过了可见终点线范围, 继续直行")
                if DEBUG is False: AGC.runActionGroup('go_forward')
                print_action("go_forward")
                time.sleep(end_t) # 继续前进一段时间
                stop()
                if DEBUG is False: AGC.runActionGroup('chest')
                print_action("chest")  # 结束模拟庆祝
               
        else:
            if detected:
                detected_end = True
                cv2.line(disp_frame, (IMG_WIDTH-1, right), (0, left), (255,0,0), 10)
                cv2.putText(disp_frame, "Finish line detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if abs(k) > threshold_k_turn:
                    print("检测到了终点线，调整方向")
                    if k > 0:
                        if DEBUG is False: AGC.runActionGroup('turn_left')
                        print_action("turn_left")
                    else:
                        if DEBUG is False: AGC.runActionGroup('turn_right')
                        print_action("turn_right")
                else:
                    print("检测到了终点线，直行中")
                    if squating is True:
                        if DEBUG is False: AGC.runActionGroup('stand')
                        print_action("stand")
                        squating = False
                    if DEBUG is False: AGC.runActionGroup('go_forward')
                    print_action("go_forward")


     #—————— 定时清除转向计数（防止局部抖动导致计数累加） ——————
    if (current_time - last_turn_time) > reset_turn_interval:
        turn_cnt = 0

    #—————— 判断是否已绕过当前障碍 ——————
    if turn_cnt > turn_limit:
        print("障碍物完成绕行：", obstacle_count+1)
        turn_cnt = 0
        obstacle_count += 1
        reverse_x_threshold() # 反转绕行方向
        current_mode = "obstacle"
        last_turn_time = current_time
        # 移除已绕过的颜色
        if color is not None:
            print(f"绕过障碍物: {color}, 已绕过障碍数量: {obstacle_count}")
            target_colors.remove(color)  


    #—————— 如果累计三个障碍, 则切换为终点检测模式 ——————
    if obstacle_count >= 3:
        current_mode = "finish"
        #print("进入终点检测模式")

    return disp_frame

#——————————————————————————————————
# 辅助子线程（简单示例）：可以将控制直接嵌入图像处理循环中, 
# 或以另外线程方式调用动作, 这里采用主循环调用

#——————————————————————————————————
# Main主函数

if __name__ == '__main__':
    # 调试模式 不执行实际动作
    if DEBUG:
        # 导入相机标定参数（如果有标定文件，在此修改路径；否则直接使用摄像头原始图像）
        # 模拟用代码，直接读取摄像头图像
        cap = cv2.VideoCapture(0)  # 根据实际情况选择摄像头，若使用其他设备请修改
        
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

    else:
        from CameraCalibration.CalibrationConfig import *
        # 加载相机标定参数
        param_data = np.load(calibration_param_path + '.npz')
        mtx = param_data['mtx_array']
        dist = param_data['dist_array']
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (IMG_WIDTH, IMG_HEIGHT), 0, (IMG_WIDTH, IMG_HEIGHT))
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (IMG_WIDTH, IMG_HEIGHT), 5)
        
        init()
        start()
        
        # 若需要检测多色障碍, 可直接使用目标颜色集合, 此处默认lab_data中已有red, blue, yellow
        # 选择不同模式时可改变 __target_color; 此示例在detect_obstacle中遍历颜色
        
        # 初始化摄像头
        open_once = yaml_handle.get_yaml_data('/boot/camera_setting.yaml').get('open_once', False)
        if open_once:
            my_camera = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream?dummy=param.mjpg')
        else:
            my_camera = Camera.Camera()
            my_camera.camera_open()
        
        # 初始站立动作
        AGC.runActionGroup('stand')
        
        while True:
            ret, frame = my_camera.read()
            if ret:
                # 畸变矫正
                frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
                disp = process_frame(frame)
                cv2.imshow('Obstacle Run', disp)
                key = cv2.waitKey(1)
                if key == 27:  # ESC退出
                    break
            else:
                time.sleep(0.01)
        # 清理资源
        my_camera.camera_close()
        cv2.destroyAllWindows()
