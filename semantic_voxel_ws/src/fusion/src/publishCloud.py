#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from fusion import matrix2D_msg
import numpy as np
# 点の座標を定義するフレームの名前
HEADER = Header(frame_id='/velo_link')

# PointCloud2のフィールドの一覧
FIELDS = [
    # 点の座標(x, y, z)
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),

    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='label', offset=16, datatype=PointField.FLOAT32, count=1),

]

# publishする点
POINTS = [
    # FIELDSで定義したフィールドを列挙
    # [x, y, z, rgb, my_field1, my_field2]
    [0.3, 0.0, 0.0, 0xff0000, 0.5, 1.2],
    [0.0, 0.3, 0.0, 0x00ff00, 1.8, 0.0],
    [0.0, 0.0, 0.3, 0x0000ff, 0.9, 0.4],
]


class CustomPointCloud(object):
    def __init__(self):
        rospy.init_node('publish_custom_point_cloud')
        self.publisher = rospy.Publisher('/custom_point_cloud', PointCloud2, queue_size=1)

    def publish_points(self):
        point_cloud = pc2.create_cloud(HEADER, FIELDS, POINTS)
        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.publisher.publish(point_cloud)
            r.sleep()

    def callback(msg):
        arr = np.array(msg.data)
        arr = arr.reshape(msg.x_size,msg.y_size)
        print(arr)
        point_cloud = pc2.create_cloud(msg.header, FIELDS, arr)
def main():
    try:
        custom_point_cloud = CustomPointCloud()
        rospy.Subscriber("fusion/fusion_points", String, callback)
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()