 # test phase

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import time
from vit import Model
from glob import glob
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
from thop import profile

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_model(path):
    fuse_net = Model()
    fuse_net.load_state_dict(torch.load(path))
    fuse_net.eval()
    return fuse_net.to(device)


def run_demo(model, vi_path, ir_path, output_path_root, index):
    vi_img = Image.open(vi_path)
    vi_img = TF.to_tensor(vi_img).unsqueeze(0).to(device)
    ir_img = Image.open(ir_path)
    ir_img = TF.to_tensor(ir_img).unsqueeze(0).to(device)

    # flops, params = profile(model, inputs=(vi_img,ir_img,True))
    # print(f"模型的参数量: {params}")
    # print(f"模型的FLOPs: {flops}")

    img_fusion = model(vis=vi_img, ir=ir_img, is_test=True)

    file_name =  ir_path.split('\\')[-1]
    # output_path = output_path_root +'2750/'+ file_name
    output_path = output_path_root + file_name

    img_fusion = np.array(img_fusion.squeeze().cpu()*255)
    img = img_fusion.astype('uint8')

    cv2.imwrite(output_path, img)

    # utils.save_images(output_path, img)
    print(output_path)


def test(model_path = None):
    output_path = './result/'
    # if os.path.exists(output_path) is False:
    #     os.mkdir(output_path)

    # 读取每个文件夹里面的图片名称
    train_ir_data_names = glob('./testimgs/ir/*')  # 实际训练使用
    train_vi_data_names = glob('./testimgs/vi/*')  # 实际训练使用

    train_vi_data_names.sort()
    train_ir_data_names.sort()
    print(train_vi_data_names)
    # model_path = "./model/" + model_path
    model_path = "./model/" + 'Final_epoch_27050.model'



    with torch.no_grad():
        # 加载模型
        model_test = load_model(model_path)
        # 每张图片生成
        for i in range(len(train_ir_data_names)):
            start = time.time()
            run_demo(model_test, train_vi_data_names[i], train_ir_data_names[i], output_path, i)
            end = time.time()
            print('time:', end - start, 'S')
    print('Done......')


if __name__ == "__main__":
    test()