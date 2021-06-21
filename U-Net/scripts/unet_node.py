#!/usr/bin/python3
import cv2
import torch
from torchvision import datasets, transforms
from src.model import UNet
from src.loss import FocalLoss
from config import config
from src.utils import *
from src.IoU import *
from src.eval import *
# ROS module
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import time
class MyUnet:
    def __init__(self):
        self.pub = rospy.Publisher('segmentated_image', Image, queue_size=1)

        segmentation_hz = rospy.get_param("u_net/segmentation_hz", 1)
        camera_fps = rospy.get_param("camera_fps", 30)
        self.cnt = 0
        self.border = camera_fps // segmentation_hz
        self.image_height = rospy.get_param("u_net/image_height", 128)
        self.image_width = rospy.get_param("u_net/image_width", 128)
        self.load_model_pth = rospy.get_param('u_net/load_model_pth', '/')
        class_color_dict_pth = rospy.get_param('u_net/class_color_dict_pth', '/')
        self.code2id, self.id2code, self.name2id, self.id2name = Color_map(class_color_dict_pth)
        if(torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_height, self.image_width), 0),
            transforms.ToTensor(),
        ])
        self.model = UNet(3, 32, True).to(self.device)
        self.model.load_state_dict(torch.load(self.load_model_pth))
        self.model.eval()

        self.bridge = CvBridge()
        rospy.loginfo("complete loading U-Net")


    def callback(self, msg):
        self.cnt += 1
        self.cnt %= self.border
        if self.cnt == 0:
            try:
                image_orig = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except CvBridgeError as e:
                print(e)
            image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
            _, height, width = image.shape
            image = image.view(1, 3, height, width)

            pred = myTest_eval(self.model, image, self.load_model_pth, self.device, self.id2code)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            try:
                imageMsg = self.bridge.cv2_to_imgmsg(pred, "bgr8")
                self.pub.publish(imageMsg)
            except Exception as e:
                print(e)
        

if __name__ == "__main__":
    rospy.init_node('UNet')
    myUnet = MyUnet()
    rospy.Subscriber("image_raw", Image, myUnet.callback, queue_size=1)
    rospy.spin()
