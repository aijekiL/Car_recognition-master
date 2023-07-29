# -*- coding: utf-8 -*-

# 从雷达坐标转经纬度要经过几次坐标转换
# 雷达坐标（载体坐标系）--导航坐标系（ENU 东北天）--地心地固坐标系（ECEF XYZ）--大地经纬度坐标系（WGS BLH）
import math
import numpy as np
import pymysql

sin = math.sin
cos = math.cos

x,y,z = 5.88746977, 1.02342606, -0.3450868
a = 6378137
b = 6356752.3142
f = (a - b) / a
e_sq = f * (2 - f)
pi = 3.14159265359

# def get_ori_llh(): # 从数据库获得小车的经纬度

def geodetic_to_ecef(lat, lon, h):
    # (lat, lon) in degrees
    # h in meters
    lamb = math.radians(lat)
    phi = math.radians(lon)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e_sq) * N) * sin_lambda

    return x, y, z


def ecef_to_enu(x, y, z, lat0, lon0, h0):
    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    xd = x - x0
    yd = y - y0
    zd = z - z0

    t = -cos_phi * xd - sin_phi * yd

    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = t * sin_lambda + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

    return xEast, yNorth, zUp

def car_frd(x, y, z, l, k, j): # 载体坐标系--frd 北东地  a b c 代表圆滚角，俯仰角，航向角 xyz代表点云坐标
    arr_car = np.array([[x], [y], [z]])
    # l1 = [cos(k)*cos(j), -cos(l)*sin(j)+sin(l)*cos(j)*sin(k), sin(l)*sin(j)+cos(l)*cos(j)*sin(k)]
    # l2 = [cos(k)*sin(j), cos(l)*sin(j)+sin(l)*sin(j)*sin(k), sin(a)*cos(j)+cos(l)*sin(j)*sin(k)]
    # l3 = [-sin(k), cos(k)*sin(l), cos(k)*cos(l)]
    l10 = [cos(j), sin(j), 0]
    l20 = [-sin(j), cos(j), 0]
    l30 = [0, 0, 1]
    Cnb = np.array([l10, l20, l30])
    arr_enu = np.dot(Cnb, arr_car)
    xEast = arr_enu[0]
    yNorth = arr_enu[1]
    zUp = arr_enu[2]
    return xEast, yNorth, zUp

def car_enu(x, y, z, a, b, c): # 载体坐标系--enu  a b c 代表圆滚角，俯仰角，航向角 xyz代表点云坐标
    a = math.radians(a)
    b = math.radians(b)
    c = math.radians(c)
    arr_car = np.array([[x], [y], [z]])
    l1 = [cos(a)*cos(c)+sin(a)*sin(b)*sin(c), sin(c)*cos(b), sin(a)*cos(c)+cos(a)*sin(b)*sin(c)]
    l2 = [-cos(a)*sin(c)+sin(a)*sin(b)*sin(c), cos(c)*cos(b), cos(c)*cos(b)-sin(a)*sin(c)-cos(a)*sin(b)*cos(c)]
    l3 = [-sin(a)*cos(b), sin(b), cos(a)*cos(b)]
    Cnb = np.array([l1, l2, l3])
    arr_enu = np.matmul(Cnb, arr_car)
    xEast = arr_enu[0]
    yNorth = arr_enu[1]
    zUp = arr_enu[2]
    return xEast, yNorth, zUp

def frd_ecef(x,y,z,j,l,b):  # j航向角 l经度 b纬度 xyz雷达的坐标 ss = cbm*(x,)
    j = math.radians(j)
    l = math.radians(l)
    b = math.radians(b)
    l1 = [-sin(b) * cos(l), -sin(l), -cos(b) * cos(l)]  # 旋转矩阵 北东地cen
    l2 = [-sin(b) * sin(l), cos(l), -cos(b) * sin(l)]
    l3 = [cos(b), 0, -sin(b)]
    l10 = [cos(j), sin(j), 0]
    l20 = [-sin(j), cos(j), 0]
    l30 = [0, 0, 1]
    arr_car = np.array([x, -y, -z])
    cen = np.array([l1, l2, l3])
    cbm = np.array([l10, l20, l30])
    ss = np.matmul(cbm, arr_car)
    x1, y1, z1 = np.dot(cen, ss)
    return x1, y1, z1


def enu_to_ecef(xEast, yNorth, zUp, lat0, lon0, h0):   # lat0, lon0, h0代表原点（小车的llh）
    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    t = cos_lambda * zUp - sin_lambda * yNorth

    zd = sin_lambda * zUp + cos_lambda * yNorth
    xd = cos_phi * t - sin_phi * xEast
    yd = sin_phi * t + cos_phi * xEast

    x = xd + x0
    y = yd + y0
    z = zd + z0

    return x, y, z


def ecef_to_geodetic(x, y, z):   # 这个公式没问题
    # Convert from ECEF cartesian coordinates to
    # latitude, longitude and height.  WGS-84
    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2

    a = 6378137.0000  # earth radius in meters
    b = 6356752.3142  # earth semiminor in meters
    e = math.sqrt(1 - (b / a) ** 2)
    b2 = b * b
    e2 = e ** 2
    ep = e * (a / b)
    r = math.sqrt(x2 + y2)
    r2 = r * r
    E2 = a ** 2 - b ** 2
    F = 54 * b2 * z2
    G = r2 + (1 - e2) * z2 - e2 * E2
    c = (e2 * e2 * F * r2) / (G * G * G)
    s = (1 + c + math.sqrt(c * c + 2 * c)) ** (1 / 3)
    P = F / (3 * (s + 1 / s + 1) ** 2 * G * G)
    Q = math.sqrt(1 + 2 * e2 * e2 * P)
    ro = -(P * e2 * r) / (1 + Q) + math.sqrt(
        (a * a / 2) * (1 + 1 / Q) - (P * (1 - e2) * z2) / (Q * (1 + Q)) - P * r2 / 2)
    tmp = (r - e2 * ro) ** 2
    U = math.sqrt(tmp + z2)
    V = math.sqrt(tmp + (1 - e2) * z2)
    zo = (b2 * z) / (a * V)

    height = U * (1 - b2 / (a * V))

    lat = math.atan((z + ep * ep * zo) / r)

    temp = math.atan(y / x)
    if x >= 0:
        long = temp
    elif (x < 0) & (y >= 0):
        long = pi + temp
    else:
        long = temp - pi

    lat0 = lat / (pi / 180)
    lon0 = long / (pi / 180)
    h0 = height

    return lat0, lon0, h0


def geodetic_to_enu(lat, lon, h, lat_ref, lon_ref, h_ref):
    x, y, z = geodetic_to_ecef(lat, lon, h)

    return ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)


def enu_to_geodetic(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref):
    x, y, z = enu_to_ecef(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref)

    return ecef_to_geodetic(x, y, z)


def car_to_geo(x, y, z, a, b, c, lat_ref, lon_ref, h_ref):  # 从载体坐标--大地经纬度
    # xyz代表雷达坐标 abc代表圆滚角，俯仰角，航向角 llh代表车的经纬度
    xEast, yNorth, zUp = car_enu(x, y, z, a, b, c)   # enu坐标系下坐标
    ex, ey, ez = enu_to_ecef(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref) # ecef下的xyz坐标
    return ecef_to_geodetic(ex, ey, ez)

def car_geo(x, y, z, c, lat, lon, h):
    ex1, ey1, ez1 = frd_ecef(x, y, z, c, lon, lat)
    ex0, ey0, ez0 = geodetic_to_ecef(lat, lon, h)
    ex = ex1 + ex0
    ey = ey1 + ey0
    ez = ez1 + ez0
    return ecef_to_geodetic(ex, ey, ez)


if __name__ == '__main__':
    #frd_ecef(26.71636390686035, 1.4836910963058472, -1.009831547737122, -87.830902, 114.615962345, 30.460104929)
    frd_ecef(5.88746977, 1.02342606, -0.34505868, -87.830902, 114.615962345, 30.460104929)


    # x,y,z = ecef_to_geodetic(-2292038.470266606,5002499.617620880, 3214450.744199568)
    # print(x,y,z)



