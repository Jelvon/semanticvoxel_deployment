#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from fusion.msg import matrix2D_msg
import numpy as np

HEADER = Header(frame_id='/velo_link')

FIELDS = [

    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),

    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='label', offset=16, datatype=PointField.UINT32, count=1),

]

POINTS = [

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

    def callback(self,msg):
        arr = np.array(msg.data)
        arr = arr.reshape(msg.x_size,msg.y_size)
        #print(arr)
	#print(msg.header)
        point_cloud = pc2.create_cloud(HEADER,FIELDS, arr)
	print(11)
	print(msg.x_size,msg.y_size)
	print(np.max(arr[:,4]))
	self.publisher.publish(point_cloud)
def main():
    try:
        custom_point_cloud = CustomPointCloud()
        rospy.Subscriber("fusion/fusion_points", matrix2D_msg,custom_point_cloud.callback)
	rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
