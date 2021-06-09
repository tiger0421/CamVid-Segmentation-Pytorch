import cv2
import torch
from torchvision import datasets, transforms
from torchinfo import summary
from src.model import UNet
from src.loss import FocalLoss
from config import config
from src.utils import *
from src.Camvid import *
from src.IoU import *
from src.eval import *
# ROS module
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class MyUnet:
    def __init__(self):
        self.pub = rospy.Publisher('segmentated_image', Image, queue_size=10)
        self.CONFIG = config()
        self.path = CONFIG.path
        self.batch = CONFIG.batch
        self.input_size = CONFIG.input_size
        self.load_model_pth = CONFIG.load_model
        self.device = CONFIG.device

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size, 0),
            transforms.ToTensor(),
        ])
        self.model = UNet(3, 32, True).to(self.device)


    def callback(self, msg):
        try:
            image_orig = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            print(e)
        image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        _, height, width = image.shape
        image = image.view(1, 3, height, width)

        pred = myTest_eval(self.model, image, self.load_model_pth, self.device)
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        try:
            imageMsg = bridge.cv2_to_imgmsg(pred, "bgr8")
            self.pub.publish(imageMsg)
        except Exception as e:
            print(e)
        

if __name__ == "__main__":
    rospy.init_node('UNet')
    myUnet = MyUnet()
    rospy.Subscriber("image_raw", Image, myUnet.callback)
    rospy.spin()
