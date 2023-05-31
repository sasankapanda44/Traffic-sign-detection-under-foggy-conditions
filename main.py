from dehazing.dehaze import dehaze
from  yolo.detection import detect
import matplotlib.pyplot as plt
from  PIL import Image
import numpy as np
import cv2
img_path = 'test8.jpeg'
model_path = 'dehazing/ots_train_ffa_3_19.pk'
yolo_path = 'yolo/best.pt'


def main():
    dehazed_img = dehaze(img_path,model_path)
    detection_res  = detect(dehazed_img,yolo_path)
    img = Image.fromarray(detection_res)
    img = img.convert('RGB')
    img = np.array(img)
    cv2.imwrite('output_8.png',img)


if __name__ == '__main__':
    main()
