import os

from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

currdir = os.getcwd()
father_dir=os.path.abspath(os.path.dirname(os.getcwd()))
patch_number=200
i=0
def image_crop(img):
    hight = 200
    width = 200
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for patch_i in range(patch_number):
        patch = transforms.RandomCrop((int(hight), int(width)))(img)
        patch.save(currdir+"\\patches\\patchmaa{}.jpg".format(patch_i))

if __name__ == '__main__':
    image=cv2.imread("./frames/pourconcrete1806.jpg")
    image_crop(image)
