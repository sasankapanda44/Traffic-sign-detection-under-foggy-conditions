import torch
import torchvision
from PIL import Image
from .FFA import *
import matplotlib.pyplot as plt

model_path = './ots_train_ffa_3_19.pk'
device  = 'cuda' if torch.cuda.is_available() else 'cpu'

gps = 3
blocks = 19

def load_model(model_path):
    checkpoints = torch.load(model_path, map_location=device)
    net = FFA(gps, blocks)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(checkpoints['model'])
    net.eval()
    return net

def pre_processing(data):
    data = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(800),
        torchvision.transforms.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(data)[None, ::]
    return data


def inference(data, model_path):
    model = load_model(model_path)
    with torch.no_grad():
        pred = model(data)
    return pred


def post_processing(data):
    ts = torch.squeeze(data.clamp(0, 1).cpu())
    return ts

def dehaze(img_path,model_path):
    img = Image.open(img_path).convert('RGB')
    pre_img = pre_processing(img)
    pred = inference(pre_img,model_path)
    post_img = post_processing(pred)
    return post_img

def test():
    img = Image.open('test4.jpg').convert('RGB')
    pre_img = pre_processing(img)
    pred = inference(pre_img,model_path=model_path)
    post_img = post_processing(pred)
    post_img = torch.tensor(post_img).permute(1,2,0)
    print(post_img.shape)
    plt.imshow(post_img)
    plt.show()

if __name__ == '__main__':
    test()
    

