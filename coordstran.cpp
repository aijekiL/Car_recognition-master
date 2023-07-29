#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <string>

#include<iomanip>
using namespace std;

int main(){
    Eigen::Vector3d t;
    // t << 0.21, 0, 0;
    t << 0, 0, 0;
    Eigen::Vector3d preCoord;
    Eigen::Vector3d curCoord;
    preCoord << 0.859902, -3.176400, -0.820807;  # 雷达坐标
    curCoord << 30.461606227, 114.614849085, 0;

    Eigen::Matrix<double, 3, 3> p;

        Eigen::Matrix<double, 3, 3> cbm;
        double h = -123.974770 * M_PI / 180;     # 车的航向角cbm

            cbm << cos(h), sin(h), 0,
        -sin(h), cos(h), 0,
        0, 0, 1;


        double b = 30.46167845 *  M_PI / 180;        # 车的经纬度
        double l = 114.614849085 *  M_PI / 180;


    Eigen::Matrix<double, 3, 3> cen;
  
    cen << -sin(b)*cos(l), -sin(l), -cos(b) * cos(l),      # 旋转矩阵 北东地cen
    -sin(b) * sin(l), cos(l), -cos(b) * sin(l),
    cos(b), 0, -sin(b);

    Eigen::Vector3d mid = preCoord;      # 雷达坐标 cen*ss表示目标点与车在笛卡尔坐标系下的变化分量
    Eigen::Vector3d mids;
    mids << mid(0), -mid(1), -mid(2);
    Eigen::Vector3d ss =  cbm * mids;   # cbm 载体转东北地 mids雷达坐标  ss=cbm*mids 表示car_enu
    cout << "ss:" << ss << endl;                   cen是enu_wgs
                                       ss =

    Eigen :: Vector3d resss(-2292015.1395, 5002512.6552, 3214449.1065);   # 车的笛卡尔坐标
    cout << "resss + ss:\n" << setiosflags(ios::fixed) << setprecision(9) << std::fixed << resss + ss<< endl;
    cout << "cen * ss:\n" << setiosflags(ios::fixed) << setprecision(9) << std::fixed << cen * ss << endl;
    cout << "resss + cen * ss:\n" << setiosflags(ios::fixed) << setprecision(9) << std::fixed <<  resss + cen * ss << endl;

}

