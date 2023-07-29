import math
import numpy as np
import pymysql

sin = math.sin
cos = math.cos


a = 6378137
b = 6356752.3142
f = (a - b) / a
e_sq = f * (2 - f)
pi = 3.14159265359

def car_enu(x, y, z, a, b, c): # 载体坐标系--enu  a b c 代表圆滚角，俯仰角，航向角 xyz代表点云坐标
    arr_car = np.array([x, y, z])
    l1 = [cos(a)*cos(c)+sin(a)*sin(b)*sin(c), sin(c)*cos(b), sin(a)*cos(c)+cos(a)*sin(b)*sin(c)]
    l2 = [-cos(a)*sin(c)+sin(a)*sin(b)*sin(c), cos(c)*cos(b), cos(c)*cos(b)-sin(a)*sin(c)-cos(a)*sin(b)*cos(c)]
    l3 = [-sin(a)*cos(b), sin(b), cos(a)*cos(b)]
    Cnb = np.array([l1, l2, l3])
    arr_enu = np.dot(Cnb, arr_car)
    xEast = arr_enu[0]
    yNorth = arr_enu[1]
    zUp = arr_enu[2]
    return xEast, yNorth, zUp

def enu_ecef(xEast, yNorth, zUp, b, l, x0, y0, z0):
    arr_enu = np.array([xEast, yNorth, zUp])
    l1 = [-sin(b) * cos(l), -sin(l), -cos(b) * cos(l)] # 旋转矩阵 北东地cen
    l2 = [-sin(b) * sin(l), cos(l), -cos(b) * sin(l)]
    l3 = [cos(b), 0, -sin(b)]
    Cen = np.array(l1, l2, l3)
    arr_ecef = np.dot(Cen, arr_enu)
    x = x0 + arr_ecef[0]
    y = y0 + arr_ecef[1]
    z = z0 + arr_ecef[2]



