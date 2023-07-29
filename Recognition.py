# -*- coding: UTF-8 -*-
import argparse
import time
import os
import shutil
import math
import cv2
import torch
import pymysql
import base64
from numpy import random
import open3d as o3d
import copy
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords, increment_path
from utils.torch_utils import time_synchronized
from utils.cv_puttext import cv2ImgAddText
from plate_recognition.plate_rec import get_plate_result,allFilePath, allFilePath_img, allFilePath_txt,init_model,cv_imread
# from plate_recognition.plate_cls import cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
from plate_recognition.color_rec import plate_color_rec,init_color_model
from car_recognition.car_rec import init_car_rec_model,get_color_and_score
from database import db, get_info, get_data, lidar_coord, distance, post_event, is_point_inside_polygon
from coord_trans import car_to_geo, car_geo, car_frd, car_enu
from pic_lighten import PSShadowHighlight, ps_shadow_highlight_adjust_and_save_img

colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
danger = ['危', '险']
object_color = [(0, 255, 255), (0, 255, 0), (255, 255, 0)]
class_type = ['单层车牌', '双层车牌', '汽车']



def order_points(pts):                   #四个点安好左上 右上 右下 左下排列
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):                       #透视变换得到车牌小图
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):  #返回到原图坐标
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    # coords[:, 8].clamp_(0, img0_shape[1])  # x5
    # coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num,device,plate_rec_model,car_rec_model):
    h,w,c = img.shape
    result_dict = {}
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    landmarks_np = np.zeros((4,2))
    rect = [x1, y1, x2, y2]
    
    if int(class_num) == 2:
        # 汽车
        car_roi_img = img[y1:y2,x1:x2]
        car_color, color_conf=get_color_and_score(car_rec_model,car_roi_img,device)
        result_dict['class_type']=class_type[int(class_num)]
        result_dict['rect']=rect                      #车辆roi
        result_dict['score']=conf                     #车牌区域检测得分
        result_dict['object_no']=int(class_num)
        result_dict['car_color']=car_color
        result_dict['color_conf']=color_conf
        return result_dict
    else:
        for i in range(4):
            point_x = int(landmarks[2 * i])
            point_y = int(landmarks[2 * i + 1])
            landmarks_np[i]=np.array([point_x, point_y])

        class_label = int(class_num)  #车牌的的类型0代表单牌，1代表双层车牌
        roi_img = four_point_transform(img,landmarks_np)   #透视变换得到车牌小图
        if class_label:        #判断是否是双层车牌，是双牌的话进行分割后然后拼接
            roi_img = get_split_merge(roi_img)
        plate_number, plate_color = get_plate_result(roi_img,device,plate_rec_model)                 #对车牌小图进行识别,得到颜色和车牌号
        for dan in danger:                                                           #只要出现‘危’或者‘险’就是危险品车牌
            if dan in plate_number:
                plate_number='危险品'
        # cv2.imwrite("roi.jpg",roi_img)
        result_dict['class_type']=class_type[class_label]
        result_dict['rect']=rect                      #车牌roi区域
        result_dict['landmarks']=landmarks_np.tolist() #车牌角点坐标
        result_dict['plate_no']=plate_number   #车牌号
        result_dict['roi_height']=roi_img.shape[0]  #车牌高度
        result_dict['plate_color']=plate_color   #车牌颜色
        result_dict['object_no']=class_label   #单双层 0单层 1双层
        result_dict['score'] = conf           #车牌区域检测得分
        return result_dict


def detect_Recognition_plate(model, orgimg, device,plate_rec_model,img_size,car_rec_model=None):
    # Load model
    # img_size = opt_img_size
    conf_thres = 0.3
    iou_thres = 0.5
    car_list = [] # 只含有车的框
    plate_list = [] # 只含有车牌的框
    # orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' 
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # img =process_data(img0)
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]
    t2=time_synchronized()
    # print(f"infer time is {(t2-t1)*1000} ms")

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # print('img.shape: ', img.shape)
    # print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:13].view(-1).tolist()
                class_num = det[j, 13].cpu().numpy()
                result_dict = get_plate_rec_landmark(orgimg, xyxy, conf, landmarks, class_num,device,plate_rec_model,car_rec_model)
                if result_dict['class_type'] == '汽车':
                    car_list.append(result_dict)
                else:
                    plate_list.append(result_dict)
    for plt in plate_list[:]:  # 使用[:]类似plate_dict的副本 实现在副本上遍历，在原来的上面操作
        v = 0
        for car in car_list:
            if plt['rect'][0] > car['rect'][0] and plt['rect'][1] > car['rect'][1] \
                    and plt['rect'][2] < car['rect'][2] and plt['rect'][3] < car['rect'][3]:
                v = 1
                break
            else:
                continue
        if v == 0:  # 如果车牌框不在汽车框的内部，则删除这个车牌框
            plate_list.remove(plt)

    # for car in car_dict[:]:  # 使用[:]类似car_dict的副本 实现在副本上遍历，在原来的上面操作
    #     v = 0
    #     for plt in plate_dict:
    #         if plt['rect'][0] > car['rect'][0] and plt['rect'][1] > car['rect'][1] \
    #                 and plt['rect'][2] < car['rect'][2] and plt['rect'][3] < car['rect'][3]:
    #             v = 1
    #             break
    #         else:
    #             continue
    #     if v == 0:  # 如果车牌框不在汽车框的内部，则删除这个汽车框
    #         car_dict.remove(car)
    dict_list = car_list + plate_list  # 车+车牌的框
    return dict_list, car_list, plate_list
    # cv2.imwrite('result.jpg', orgimg)


def draw_result(orgimg, dict_list):
    result_str = ""
    for result in dict_list:
        rect_area = result['rect']
        object_no = result['object_no']
        if not object_no == 2:  # 车牌
            x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
            padding_w = 0.05 * w
            padding_h = 0.11 * h
            rect_area[0] = max(0, int(x - padding_w))
            rect_area[1] = max(0, int(y - padding_h))
            rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
            rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

            height_area = int(result['roi_height'] / 2)
            landmarks = result['landmarks']
            result_p = result['plate_no']
            if result['object_no'] == 0:  # 单层
                result_p += " " + result['plate_color']
            else:  # 双层
                result_p += " " + result['plate_color'] + "双层"
            result_str += result_p + " "
            for i in range(4):  # 关键点
                cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, colors[i], -1)

            if len(result) >= 1:
                if "危险品" in result_p:  # 如果是危险品车牌，文字就画在下面
                    orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0], rect_area[3], (0, 255, 0), height_area)
                else:
                    orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0] - height_area,
                                           rect_area[1] - height_area - 10, (0, 255, 0), height_area)
        else:
            height_area = int((rect_area[3] - rect_area[1]) / 20)
            car_color = result['car_color']
            car_color_str = "车辆颜色:"
            car_color_str += car_color
            orgimg = cv2ImgAddText(orgimg, car_color_str, rect_area[0], rect_area[1], (0, 255, 0), height_area)

        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), object_color[object_no],
                      2)  # 画框
    print(result_str)
    return orgimg


def get_second(capture):
    if capture.isOpened():
        rate = capture.get(5)   # 帧速率
        FrameNumber = capture.get(7)  # 视频文件的帧数
        duration = FrameNumber/rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
        return int(rate),int(FrameNumber),int(duration)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default='/home/ubuntu/Car_recognition-master/weights/detect.pt', help='model.pt path(s)')  #检测模型
    parser.add_argument('--rec_model', type=str, default='/home/ubuntu/Car_recognition-master/weights/plate_rec_color.pth', help='model.pt path(s)')#车牌识别+车牌颜色识别模型
    parser.add_argument('--car_rec_model',type=str,default='/home/ubuntu/Car_recognition-master/weights/car_rec_color.pth',help='car_rec_model') #车辆识别模型
    parser.add_argument('--image_path', type=str, default='/home/ubuntu/Car_recognition-master/img_txt', help='source')
    parser.add_argument('--lighten_image_path', type=str, default='/home/ubuntu/Car_recognition-master/img_txt/lighten', help='source')
    parser.add_argument('--img_size', type=int, default=384, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='/home/ubuntu/Car_recognition-master/plate_imgs', help='source')
    parser.add_argument('--video', type=str, default='', help='source')
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    opt = parser.parse_args()
    print(opt)
    park_area = []
    p1 = [(114.61567824896055, 30.461089619612636), (114.61567678411323, 30.46105027905674),
          (114.61590238251921, 30.461062969702066), (114.61589798775309, 30.461104848356083),
          (114.61567824896055, 30.461089619612636)]
    p2 = [(114.61536664422414, 30.460133750452176), (114.61537103336276, 30.460017146849832),
          (114.61554074596415, 30.460024751320695), (114.61552757879478, 30.46014008752297),
          (114.61536664422414, 30.460133750452176)]
    p3 = [(114.61452050339922, 30.46009526711145), (114.61470482635444, 30.46010287116218),
          (114.61471920513591, 30.4599828885636), (114.61453342603686, 30.45997021588016),
          (114.61452050339922, 30.46009526711145)]
    p4 = [(114.61446033915055, 30.461870759408075), (114.61446358814757, 30.46183886255628),
          (114.61478414186162, 30.461869821760853), (114.61477872707061, 30.461897027888174),
          (114.61446033915055, 30.461870759408075)]
    p5 = [(104.048574, 30.752506), (104.050062, 30.751254), (104.050801, 30.750952), (104.048574, 30.752506),
          (104.048574, 30.752506)]
    p6 = [(114.61589329270005, 30.46010134938339), (114.61589329274565, 30.46004186360928),
                  (114.61627818665417, 30.46006064836139), (114.61627276575489, 30.460124830382213), (114.61589329270005, 30.46010134938339)]
    park_area.append(p1)
    park_area.append(p2)
    park_area.append(p3)
    park_area.append(p4)
    park_area.append(p5)
    park_area.append(p6)

    count = 0
    detect_model = load_model(opt.detect_model, device)  # 初始化检测模型
    plate_rec_model = init_model(device, opt.rec_model)  # 初始化识别模型
    car_rec_model = init_car_rec_model(opt.car_rec_model, device)  # 初始化车辆识别模型
    # 算参数量
    total = sum(p.numel() for p in detect_model.parameters())
    total_1 = sum(p.numel() for p in plate_rec_model.parameters())
    print("detect params: %.2fM,rec params: %.2fM" % (total / 1e6, total_1 / 1e6))
    time_all = 0
    time_begin = time.time()
    save_path = increment_path(Path(opt.output) / 'exp')
    pic_path = increment_path(Path(opt.image_path) / 'exp')  # 图片路径
    lighten_path = increment_path(Path(opt.lighten_image_path) / 'exp')  # 图片路径


       # 全局变量 因为不同帧的图片可能识别出相同的车牌号
    cars_info = [] # 形如plates_infor:[{'id':3412,'plate':鄂AK4345,'coord':[20,30],'dist':0.2,'whe_in':False},{'plate':湘AK4345,'coord':[24,30],'dist':0.1},'whe_in':False]
    lidar_list = []
    pic_list = []
    event_time_list = []
    car_time_list = []
    event_id_list = []
    plate_num = 0  # 判断有多少个不同的车牌 即编号
    # 访问数据库
    cursor = db.cursor()   # 创建游标对象
    infor_event, infor_car, start_time, end_time = get_info(cursor)  # 获取信息 event: id编号，time时间,pic照片编码,status雷达文件,car:time,llh,head_angle

    k = 0
    for i, (data1, data2) in enumerate(zip(infor_event, infor_car)):  # 对每张图进行处理 zip将event car信息打包
        # data1(time时间,pic照片编码,status雷达文件) data2(time,lon,lat,h,head_angle)
        print("第几张图,data2:", i, data2)
        if i > 6:
            break
        event_id = data1[0]
        event_id_list.append(event_id)
        # 处理event_lidar
        lidar_txt = data1[3]  # /testLidar/2023_07_01/2023_07_01_17_38_05.txt
        shutil.copy(lidar_txt, pic_path)
        lidar_name = os.path.basename(lidar_txt)  # 2023_07_01_17_38_05.txt
        new_lidar = os.path.join(pic_path, lidar_name)  # 新的lidar路径
        lidar_list.append(new_lidar)
        # 处理event图片
        pic_encode = data1[2]
        #print("图片编码为:%s, 雷达文件为:%s" % (new_lidar, pic_encode))
        pic_name = str(i) + '.jpg'
        # 将 Base64 编码的字符串解码为图像数据
        image_data = base64.b64decode(pic_encode)
        pic = os.path.join(pic_path, pic_name)
        #print("图片路径 ：", pic)
        #cv2.imwrite(pic, image_data)
        with open(pic, "wb") as image:
            image.write(image_data)
        mod_img = ps_shadow_highlight_adjust_and_save_img(pic)
        mod_pic_path = os.path.join(lighten_path, pic_name)
        pic_list.append(mod_pic_path)
        # 将图像数据写入文件
        # with open(mod_pic_path, "wb") as image:
        #     image.write(mod_img)
        cv2.imwrite(mod_pic_path, mod_img)
        # 处理event时间
        event_time = data1[1]
        event_time_list.append(event_time)
        # 处理car的信息
        car_time = data2[0]
        lon_ref = data2[1]
        lat_ref = data2[2]
        h_ref = data2[3]
        head_angle = data2[4]
        im0 = mod_img.copy()  # 复制img img用来做前处理 im0用来做车牌识别和画框
        #print(count, pic_path, end=" ")
        time_b = time.time()

        if mod_img is None:
            continue
        if im0.shape[-1] == 4:
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGRA2BGR)

        # detect_one(model,img_path,device)
        dict_list, car_list, plt_list = detect_Recognition_plate(detect_model, im0, device, plate_rec_model, opt.img_size,
                                             car_rec_model) # 所有框，车框，车牌框
        car_boxes = len(car_list)
        plt_boxes = len(plt_list)
        print("car_plt_list:", car_list, plt_list)
        ori_img = draw_result(im0, dict_list)
        point_list = lidar_coord(new_lidar, mod_img, car_list)   # 车辆中心点的雷达坐标系坐标

        # 将每个车牌号对应的信息存起来
        # 形如cars_infor:[{'id':3412,'num':0, plate':鄂AK4345,'lon':20,'lat':30,'dist':0.2,'whe_in':True},{'num':1,'plate':湘AK4345,'lon':20,'lat':30,'dist':0.1,'whe_in':False}]
        for j, point in enumerate(point_list):  # point是图中每辆车的点云坐标
            x = point[0]
            y = point[1]
            z = point[2]
            #print("xyz:", x, y, z)
            #llh = car_to_geo(x, y, z, 0, 0, head_angle, lat_ref, lon_ref, h_ref)
            #llh = car_to_geo(5.88746977, 1.02342606, -0.34505868, 0, 0, -87.830902, 30.460104929, 114.615962345, 12.126)
            #llh = car_to_geo(x, y, z, 0, 0, -87.830902, 30.460104929, 114.615962345, 12.126)
            #llh = car_geo(x, y, z, ex0, ey0, ez0, -87.830902, 30.460104929, 114.615962345) ex0 ey0 ez0表示车的ecef坐标
            llh = car_geo(x, y, z, head_angle, lat_ref, lon_ref, h_ref)
            #print("llh:", llh)
            lat = llh[0]  # 纬度
            lon = llh[1]  # 经度
            # 判断违停 车辆的经纬度在违停区域内且dist<0则表示违停
            point_coords = lon, lat
            in_result = False
            for area in park_area:
                in_result = is_point_inside_polygon(point_coords, area)  # 输出 True 或 False，表示点是否在多边形内部
                if in_result:
                    break
            print("in_result:", in_result)
            # 车牌号临时变量plate_t
            if plt_boxes:
                if j <= plt_boxes-1:
                    plate_t = plt_list[j]['plate_no']
                else:
                    plate_t = "未检测到车牌号" + str(k)
            else:
                plate_t = "未检测到车牌号" + str(k)
            print("plate:", plate_t)
            if not len(cars_info):  # 第一次写入车辆信息
                car_info ={}
                car_info['id'] = event_id
                car_info['num'] = plate_num   # 识别出的车牌数量
                car_info['plate'] = plate_t
                car_amount = 0
                lonlist = []
                latlist = []
                distances = []
                whe_in = []
                lonlist.append(lon)
                latlist.append(lat)
                car_info['lon'] = lon
                car_info['lat'] = lat
                car_info['amount'] = 1   # 表示同一车牌的车共出现几次
                car_info['dist'] = 0  # 表示同一车牌的车前后距离差在阈值内的次数
                car_info['whe_in'] = 1 if in_result else 0  # whe_in 表示同一车牌的车在违停区域有几次出现
                cars_info.append(car_info)
            else:  #
                plate_num_t = plate_num
                for info in cars_info:  # 判断是否已录入该车牌
                    if plate_t in info.values():  # 有 则更新该车牌的信息
                        plate_num_t += 1  # 如果存在，将车牌数量临时变量+1
                        info['id'] = event_id
                        lon_0 = lonlist[0]  # 第一次识别出该车牌时的lon
                        lat_0 = latlist[0]  # 第一次识别出该车牌时的lat
                        pic_dist = distance(lon, lon_0, lat, lat_0)
                        print("pic_dist:", pic_dist)
                        lonlist.append(lon)
                        latlist.append(lat)
                        info['lon'] = lon
                        info['lat'] = lat
                        car_info['amount'] += 1
                        if pic_dist < 1.23e-8:
                            info['dist'] += 1
                        if in_result:
                            car_info['whe_in'] += 1
                        break
                    else:
                        continue
                if plate_num == plate_num_t:  # 如果没有录入该车牌 plate_num_t不会改变 就插入新的车牌信息
                    plate_num += 1
                    car_info = {}
                    car_info['id'] = event_id
                    car_info['num'] = plate_num
                    car_info['plate'] = plate_t
                    car_amount = 0
                    lonlist = []
                    latlist = []
                    distances = []
                    lonlist.append(lon)
                    latlist.append(lat)
                    car_info['lon'] = lon
                    car_info['lat'] = lat
                    car_info['amount'] = 1
                    car_info['dist'] = 0  # 录入新的车牌dist都是0
                    car_info['whe_in'] = 1 if in_result else 0
                    cars_info.append(car_info)
            k += 1


        save_img_path = os.path.join(save_path, pic_name)
        time_e = time.time()
        time_gap = time_e - time_b
        if count:
            time_all += time_gap
        cv2.imwrite(save_img_path, ori_img)
        # with open(save_img_path, "wb") as image:
        #     image.write(ori_img)
        count += 1
    print("车辆信息：", cars_info)
    # 判断违停并回传

    vio_parking = []
    for info in cars_info:
        if info['amount'] == 1:  # 该车牌只有一张
            if info['whe_in'] > 0:
                vio_parking.append(info['plate'])
        else:   # 该车牌有多张，需要结合dist判断
            if info['dist'] > 0 and info['whe_in'] > 0:
                vio_parking.append(info['plate'])

    sql3 = "UPDATE event set eventdescription = '{}' WHERE time = '{}'".format(vio_parking, end_time)
    #sql4 = "DELETE from event WHERE time >= '%s' and time < '%s'" % (start_time, end_time)
    cursor.execute(sql3)
    #cursor.execute(sql4)
    db.commit()
    cursor.close()
    db.close()

    # # 将违停信息上传数据库
    # for info in cars_info:
    #     sign = "违停车牌号为：" + info['plate']
    #     pic_id = info['id']
    #     cursor = db.cursor()  # 创建游标对象
    #     sql = "UPDATE event set eventdescription = sign WHERE id = pic_id"
    #     cursor.execute(sql)














