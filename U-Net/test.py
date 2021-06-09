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

CONFIG = config()
path = CONFIG.path
batch = CONFIG.batch
input_size = CONFIG.input_size
load_model_pth = CONFIG.load_model
device = CONFIG.device

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size, 0),
        transforms.ToTensor(),
    ])

    image_orig = cv2.imread("/root/share/CamVid-Segmentation-Pytorch/images/real_line2.jpg", cv2.IMREAD_COLOR)
    image_orig = image_orig[:1024, :1024, :]
    image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    image = transform(image)
    _, height, width = image.shape
    image = image.view(1, 3, height, width)

    model = UNet(3, 32, True).to(device)

    pred = myTest_eval(model, image, load_model_pth, device)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    cv2.imwrite("input.png", image_orig)
    cv2.imwrite("result.png", pred)
