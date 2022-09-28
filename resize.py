import os
import cv2

datadir = "/home/jas0n/PycharmProjects/covid_ct_torch/COVID-CT/all_image_resized"

'''设置目标像素大小，此处设为240'''
IMG_SIZE = 224

'''使用os.path模块的join方法生成路径'''
path = os.path.join(datadir)

'''使用os.listdir(path)函数，返回path路径下所有文件的名字，以及文件夹的名字'''
img_list = os.listdir(path)

for i in img_list:
    img_array = cv2.imread(os.path.join(path, i), cv2.IMREAD_COLOR)
    '''调用cv2.resize函数resize图片'''
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_name = str(i)  # 保存的图片与处理前图片同名
    '''生成图片存储的目标路径'''
    save_path = "/home/jas0n/PycharmProjects/covid_ct_torch/COVID-CT/all_image_resized/"+str(i)
    '''调用cv.2的imwrite函数保存图片'''
    cv2.imwrite(save_path, new_array)
