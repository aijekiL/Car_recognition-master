import pandas as pd
import pymysql
import base64
import os
import numpy as np
import cv2
import datetime
import shutil
import open3d as o3d
import math
from shapely.geometry import Point, Polygon

def is_point_inside_polygon(pot, pol):
    point = Point(pot)
    polygon = Polygon(pol)
    return polygon.contains(point)

# 示例用法
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






db = pymysql.connect(
        host="frp-add.top",
        port=17609,
        database='sensor',
        user='root',    #在这里输入用户名
        password='sensorweb',     #在这里输入密码
        charset='utf8mb4'
        ) #连接数据库

def get_info(cursor):
    sql = "SELECT task_id from event order by time desc LIMIT 1"  # SQL语句
    # sql2 = "SHOW COLUMNS FROM event"
    cursor.execute(sql)  # 执行SQL语句
    task_id = cursor.fetchall()
    # column_name = cursor.fetchall()
    # print(column_name)
    date = task_id[0][0]  # 通过fetchall方法获得时间
    start_time = date.split(',')[0]
    end_time = date.split(',')[1]
    print(start_time, end_time)
    # 执行多条SQL语句
    # sql_statements = [
    #     "SELECT time,eventpicture,eventstatus from event where time BETWEEN '%s' and '%s'" % (start_time, end_time),
    #     "SELECT longtitude,latitude,high,headingangle from car_data where time BETWEEN '%s' and '%s'" % (start_time, end_time)
    # ]
    #
    # for statement in sql_statements:
    #     cursor.execute(statement)
    #     results = cursor.fetchall()
    #     for row in results:
    #         print(row)
    sql1 = "SELECT id,time,eventpicture,eventstatus from event where time BETWEEN '%s' and '%s'" % (start_time, end_time)
    sql2 = "SELECT time,longtitude,latitude,high,headingangle from car_data where time BETWEEN '%s' and '%s'" % (start_time, end_time)
    cursor.execute(sql1)
    infor_event = cursor.fetchall()
    cursor.execute(sql2)
    infor_car = cursor.fetchall()

    return infor_event, infor_car, start_time, end_time

def post_event(cursor,sign,pic_id):
    sql = "UPDATE event set car_id = sign WHERE id = pic_id"
    cursor.execute(sql)

def get_data(infor,save_path):  # 通过数据库得到的信息获取（雷达txt,图片路径）的文件路径
    lidar_list = []
    pic_list = []
    time_list = []
    k = 0
    for data in infor:
        lidar_txt = data[-3]  # /testLidar/2023_07_01/2023_07_01_17_38_05.txt
        lidar_name = os.path.basename(lidar_txt)  # 2023_07_01_17_38_05.txt
        new_lidar = os.path.join(save_path, lidar_name)
        shutil.copy(lidar_txt, save_path)
        lidar_list.append(new_lidar)

        pic_encode = data[-4]
        print("图片编码为:%s, 雷达文件为:%s" % (new_lidar, pic_encode))
        pic_name = str(k) + '.jpg'
        # 将 Base64 编码的字符串解码为图像数据
        image_data = base64.b64decode(pic_encode)
        pic_path = os.path.join(save_path, pic_name)
        pic_list.append(pic_path)
        # 将图像数据写入文件
        with open(pic_path, "wb") as image_file:
            image_file.write(image_data)
        time = data[1]
        time_list.append(time)
        k += 1
        if k >= 5:
            break
    return lidar_list, pic_list, time_list


def lidar_coord(new_lidar,img, car_list):  # 投影坐标
    point_list = []
    pointcloud_data = np.loadtxt(new_lidar, delimiter=" ")
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pointcloud_data)
    # 定义相机内参矩阵
    intrinsic_matrix = np.array(
        [[587.1242998410094, 0, 328.747030982911], [0, 589.5352850380269, 243.2224945558319], [0, 0, 1]],
        dtype=np.float64)
    # 定义雷达到相机的转换矩阵
    transformation_matrix = np.array([[0.000279395, -0.998832, -0.0483229, -0.0245034],
                                      [0.163632, 0.0476259, -0.985371, -0.576563],
                                      [0.986521, -0.00818246, 0.163427, -0.558098],
                                      [0, 0, 0, 1]], dtype=np.float64)
    # 将点云转换到相机坐标系
    pointcloud_camera = o3d.geometry.PointCloud()
    pointcloud_camera.points = o3d.utility.Vector3dVector(
        np.dot(pointcloud.points, transformation_matrix[:3, :3].T)
        + transformation_matrix[:3, 3])
    # 投影点云到图像上
    pixel_coordinates = np.dot(pointcloud_camera.points, intrinsic_matrix.T)
    pixel_coordinates[:, :2] /= pixel_coordinates[:, 2:]
    # 在图像上绘制点云投影
    for pixel_coordinate in pixel_coordinates:
        x = int(pixel_coordinate[0])
        y = int(pixel_coordinate[1])
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    # 定义车辆检测框像素坐标范围
    for result in car_list:
        rect_area = result['rect']
        min_x, max_x = rect_area[0], rect_area[2]
        min_y, max_y = rect_area[1], rect_area[3]
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        selected_point = None
        min_distance = float('inf')
        for j, pixel_coordinate in enumerate(pixel_coordinates):
            x = pixel_coordinate[0]
            y = pixel_coordinate[1]
            # 计算点到中心点的距离
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            # 如果距离比之前的最小距离小，则更新选择的点
            if distance < min_distance and min_x <= x <= max_x and min_y <= y <= max_y:
                selected_point = pointcloud_camera.points[j]
                min_distance = distance
        selected_point_original = np.dot(selected_point - transformation_matrix[:3, 3],
                                         np.linalg.inv(transformation_matrix[:3, :3].T))
        point_list.append(selected_point_original)
    return point_list


def distance(a, b, c, d):
    dist = math.sqrt((a-b)**2 + (c-d)**2)
    return dist

if __name__ == '__main__':
    # data = get_data(db)
    # get_info(data)
    #update_data(db)
    dist = distance(3, 1, 4, 2)
    print(dist)





