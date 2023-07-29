import pymysql


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
    sql1 = "SELECT time,eventpicture,eventstatus from event where time BETWEEN '%s' and '%s'" % (start_time, end_time)
    sql2 = "SELECT longtitude,latitude,high,headingangle from car_data where time BETWEEN '%s' and '%s'" % (start_time, end_time)
    # cursor.execute(sql1)
    # infor_event = cursor.fetchall()
    cursor.execute(sql2)
    infor_car = cursor.fetchall()

    return infor_car

if __name__ == '__main__':
    db = pymysql.connect(
        host="frp-add.top",
        port=17609,
        database='sensor',
        user='root',  # 在这里输入用户名
        password='sensorweb',  # 在这里输入密码
        charset='utf8mb4'
    )  # 连接数据库

    cursor = db.cursor()  # 创建游标对象
    infor_car = get_info(cursor)  # 获取信息 event:time时间,pic照片编码,status雷达文件,car:llh,head_angle
    print("car:", infor_car)
    cursor.close()
    db.close()