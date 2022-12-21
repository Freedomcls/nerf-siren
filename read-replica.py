import os
import json
import cv2
import math

# input_dir = "/media/secondDisk/scannet_frames_25k/scene0001_00/color/"
input_dir = "/home/chenlinsheng/eg3d-main/Sequence_1/rgb/"
output_dir = "replica"
output_json = 'dataset.json'
img_size=128

# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)

# for file in os.listdir(input_dir):
#     file_path = os.path.join(input_dir, file)
#     img = cv2.imread(file_path)
#     img = cv2.resize(img,(img_size,img_size))
#     out_path = os.path.join(output_dir, file)
#     # img = 255 - img
#     # img = img/255
#     cv2.imwrite(out_path, img)

labels = []
i = 0
# path = "/home/chenlinsheng/eg3d-main/Sequence_1/traj_w_c.txt" #文件夹目录
path = "/home/chenlinsheng/hezijian/3D-nerf-da/room_0/Sequence_1/traj_w_c.txt"
# # files= os.listdir(path) #得到文件夹下的所有文件名称
# #get intri
H = img_size
W = img_size

n_pix = H * W
aspect_ratio = W/H

hfov = 90
# the pin-hole camera has the same value for fx and fy
fx = W / 2.0 / math.tan(math.radians(hfov / 2.0))
# self.fy = self.H / 2.0 / math.tan(math.radians(self.yhov / 2.0))
fy = fx
focal = fx
cx = (W - 1.0) / 2.0
cy = (H - 1.0) / 2.0
intri = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
# print(intri)

# if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
with open(path, 'r') as f:
    data = f.readlines()  # 将txt中所有字符串读入data
    for line in data:
        rgb_name = output_dir + '/' + 'rgb_' + str(i) + '.png'
        i=i+1
        pose = line.split()
        print('pose',pose)
        pose_float = map(float, pose)
        # print('floatpose',pose)
        label = pose + intri
        label=list(map(float,label))
        print('label',label)
        label = [rgb_name, label]
        labels.append(label)
        break
    # print(labels)
# json.dump({'labels':labels}, open('dataset.json','w'),indent=2)
