import os

from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

currdir = os.getcwd()

def image_crop(img,hight,width,patch_number,crops_filpath,imagename,i):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for patch_i in range(patch_number):
        patch = transforms.RandomCrop((int(hight), int(width)))(img)
        imagename=imagename.split(".")[0]
        save_name=imagename+"_"+str(i)+"a_"+str(patch_i)+".jpg"
        patch.save(crops_filpath+"/"+save_name)

if __name__ == '__main__':
    patch_number = 100

    "1concrete,2formwork,3rebar,4con_bar,5form_crete,6rebar_form"
    original_filpath="../patches/samples/"
    filname=os.listdir(original_filpath)
    print(filname)
    for imagename in filname:
        crops_filpath = "../patches/construction_crops/"+imagename
        if not os.path.exists(crops_filpath):
            os.makedirs(crops_filpath)

        print(original_filpath+imagename)
        image=cv2.imread(original_filpath+imagename)
        # G:\Projects\PatchAnalyse-materials\construction_original\1concrete
        sp = image.shape
        height = sp[0]#height(rows) of image
        width = sp[1]#width(colums) of image
        dimension = sp[2]#the pixels value is made up of three primary colors
        print(height,width,dimension)

        patch_percentage=[0.05,0.08,0.1,0.15, 0.2,0.25, 0.3,0.4,0.5,0.8]
        i=1
        for hw_percent in patch_percentage:
            patch_height = int(height * hw_percent)
            patch_width = int(width * hw_percent)

            image_crop(image,patch_height,patch_width,patch_number,crops_filpath,imagename,i)
            i+=1
        # cv2.imshow("image",image)
        # cv2.waitKey(0)