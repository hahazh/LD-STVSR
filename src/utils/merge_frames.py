#coding:utf-8
 
import os
import cv2
import re

from tqdm import tqdm
 
def get_images(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        if not files:
            continue
        for file in files:
            if file.endswith('.png'):
                file_list.append(os.path.join(root, file))
                #file_list.append(file)
    file_list.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
    return file_list
 

def main():
	#图片路径
    
    path = "/home/zhangyuantong/code/ST-SR/ThreeBArbitrary/adobe_multi/"
    for video in os.listdir(path):

	#将图片存入列表
        file_list = get_images(path+'/'+video)
        print(video)
        #按照图片名称排序

        #一秒25帧，代表1秒视频由25张图片组成
        fps = 25
        #视频分辨率
        img_size = (1280, 720) 
        #保存视频的路径
        save_path = "/home/zhangyuantong/code/ST-SR/ThreeBArbitrary/out_video/adobe"+'/'+video+'.avi'
        # if not os.path.exists("/home/zhangyuantong/code/ST-SR/ThreeBArbitrary/out_video/adobe"+'/'+video):
        #     os.makedirs("/home/zhangyuantong/code/ST-SR/ThreeBArbitrary/out_video/adobe"+'/'+video)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, img_size)
        for file_name in tqdm(file_list) :
            # print (file_name)
            img = cv2.imread(file_name)
            video_writer.write(img)
    
        video_writer.release()
 
if __name__ == "__main__":
    main()