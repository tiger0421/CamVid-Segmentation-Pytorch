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
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image = cv2.imread("/root/share/CamVid-Segmentation-Pytorch/images/frankfurt.png", cv2.IMREAD_COLOR)
    image = image[:1024, :1024, :]
    cv2.imwrite("input_image.png", image)
    print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)

    model = UNet(3, 32, True).to(device)
    model.load_state_dict(torch.load(load_model_pth))
    #summary(model, (1, 3, 256, 256))
    model.eval()
    _, height, width = image.shape
    image = image.view(1, 3, height, width)
    input_image = image.to(device)
    output = model(input_image.float())
    print(output.shape)
    output = mask_to_rgb(np.array(output.cpu().detach().numpy()), CONFIG.id2code)[0]
    print(output.shape)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite("result.png", output)
    print(output.shape)

    

#    imgs, masks, pred = Test_eval(model, testloader, criterion, load_model_pth, device)
#    print(imgs.shape, masks.shape, pred.shape)

#    Visualize(imgs, 'Original Image', 6, 1, change_dim=True)
#    Visualize(masks, 'Original Mask', 6, 1, change_dim=True)
#    Visualize(pred, 'Predicted mask', 6, 1, change_dim=False)
