import cv2
from ultralytics import YOLO
import torchvision
from PIL import Image
import torch
import matplotlib.pyplot as plt


path = 'best.pt'
test_img = 'test.jpg'


def load_model(model_path):
    model = YOLO(model_path)
    return model


def detect(img,model_path):
    model = load_model(model_path)

    img = torchvision.transforms.ToPILImage()(img)
    img = img.convert('RGB')
    res = model(img)
    res_plotted = res[0].plot()
    return res_plotted


def test(image_path):
    model = load_model()

    img = Image.open(image_path).convert('RGB')
    img_tensor = torchvision.transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(axis=0)
    res = model(img)
    res_plotted = res[0].plot()

    plt.imshow(res_plotted)
    plt.show()

if __name__ == '__main__':
    #model = load_model()
    test('test.png')
