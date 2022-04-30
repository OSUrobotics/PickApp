#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """

    # bag = rosbag.Bag(args.bag_file, "r")

    location = "/home/avl/ur_ws/src/apple_proxy/bag_files/"
    bagfile = "apple_proxy_pick15-0.bag"
    bag = rosbag.Bag(location + bagfile, "r")

    # bag = location + bagfile
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=["/camera/image_raw"]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        cv2.imwrite(os.path.join("/home/avl/ur_ws/src/apple_proxy/bag_files/", "frame%06i.png" % count), cv_img)
        print( "Wrote image %i" % count)

        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()