import os
from PIL import Image
import pandas as pd

dir_name_list = []
width_list = []
height_list = []

class GetAvgsize:
    def avg_size(self,Path):
        for root_pat, dirnames, files in os.walk(Path):
            for file_name in files:
                file_path = os.path.join(root_pat, file_name)
                if file_name.endswith('.jpg'):
                    img = Image.open(file_path)
                    dir_name_list.append(file_name)
                    width_list.append(img.size[0])
                    height_list.append(img.size[1])

        avg_width=sum(width_list)/len(width_list)
        avg_hight=sum(height_list)/len(height_list)
        total_number=len(height_list)
        # print('total_number',total_number)
        return avg_width,avg_hight

    def video_number(self,Path):
        videos = []
        for root_pat, dirnames, files in os.walk(Path):
            print(root_pat)
            print(dirnames)
            print(files)
            for file_name in files:
                file_path = os.path.join(root_pat, file_name)
                if file_name.endswith('.mp4'):
                    videos.append(file_name)
        videos_num = len(videos)
        print(videos_num)
GetAvgsize=GetAvgsize()


#
#
# if __name__ == '__main__':
#     # father_dir=os.path.dirname(os.getcwd())
#     # originalimage_dir=os.path.join(father_dir,'originalimage')
#     # avg_size(originalimage_dir)
#     # totaldir=r'C:\\01数据\\程序测试用2'
#     # avg_size(totaldir)
#
#     # totaldir = r'E:\001涵管检测项目\01数据\管道视频-原始视频'
#     # avg_size.video_number(totaldir)
#
