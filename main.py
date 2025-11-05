import torch
from torch.utils.data import DataLoader
from dataset import *
import os
from torch.optim import Adam,RMSprop
import kornia.filters as KF
import torch.nn.functional as F
from vit import Model
from test import test
from loss import intLoss, Fusion_loss,final_ssim
from discriminator import Discriminator



'''
1.分类先训练，然后注意力图和融合一起训练
'''

batchsize = 16
lr = 0.001
epochs = 50
device = ('cuda' if torch.cuda.is_available() else 'cpu')
ModelPath = './model/'
# dataPath = './dataset/'
dataPath = './dataset/'

# 权重初始化
def gaussian_weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            print('卷积权重初始化失败')
    elif isinstance(m, torch.nn.BatchNorm2d):
        try:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        except:
            print('BN初始化失败')

def dataLoader(dataset):
    return DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=2,
            drop_last=True,
         )


def train(dataset):
    # 加载数据
    dataloader = dataLoader(dataset)
    # 训练轮数
    train_num = len(dataset)
    # model
    print('\n--- load model ---')
    print(f'------ 训练轮数为{epochs} --------')

    model = Model()
    dis = Discriminator()
    # 初始化权重，放在cuda前面
    model.apply(gaussian_weights_init)
    dis.apply(gaussian_weights_init)
    # model 放入设备
    model.to(device)
    dis.to(device)

    # 开始计时
    from datetime import datetime
    start_time = datetime.now()
    # count 用于打印数据，count%10==0打印损失函数，count%50=0保存模型一次
    count = 0
    batch_num = len(dataloader)
    # 训练的轮数， opt.epoch = 1
    for ep in range(epochs):
        print('~~~Main 训练开始！~~~~')
        # 模型设置为train模式
        model.train()
        dis.train()

        # 设置网络优化器
        optim_G = Adam(model.parameters(), lr)
        optim_D = RMSprop(dis.parameters(), lr)

        # 没50轮降低一下学习率
        scheduler = torch.optim.lr_scheduler.StepLR(optim_G, step_size=15, gamma=0.1)

        # 每个batchsize的训练
        for it, (img_vi, img_ir) in enumerate(dataloader): # img1 是 vis ,img2 是ir
            count += 1
            print(f'--第{ep}轮---{it} / {batch_num}----  ')

            # 优化器梯度清零
            optim_G.zero_grad()
            optim_D.zero_grad()

            # 数据放入设备
            img_vi = img_vi.to(device)
            img_ir = img_ir.to(device)

            # 生成的图片名命为 gen_image
            gen_iamge = model(vis=img_vi, ir=img_ir)


            # 使用鉴别器
            real, fake  = dis(vis=img_vi, ir=img_ir, fu=gen_iamge)

            loss_dis = torch.mean((real-1)**2) + torch.mean((fake.detach()-0)**2)
            loss_gen = torch.mean((fake-1)**2)


            loss_total, loss_intensity, loss_grad = Fusion_loss(vi=img_vi,ir=img_ir,fu=gen_iamge)
            loss_total = loss_total + loss_gen
            # 总损失
            loss_total.backward()
            loss_dis.backward()
            optim_G.step()
            optim_D.step()

# ----------------------------------------------------------------------------------------------------------

            # 打印损失函数
            if count % 10 == 0:
                elapsed_time = datetime.now() - start_time
                print('loss_grad: %s, loss_int: %s,  loss_gen: %s, loss_dis: %s, loss_total: %s, selapsed_time: %s'
                      % (loss_grad.item(), loss_intensity.item(), loss_gen.item(), loss_dis.item(), loss_total.item(), elapsed_time))


            if count % 250 == 0:
                # save model
                model.eval()
                model.cpu()

                save_model_filename = "Epoch_" + str(count) + "_iters_" + str(count) + ".model"
                model_path = os.path.join(ModelPath, save_model_filename)
                torch.save(model.state_dict(), model_path)

                model.train()
                model.to(device)


            #  每个300轮验证一次结果
            if count % 500 == 0:
                # 因为计算机是并行的cpu的写入不能满足代码的调用速度，所以使用本轮前面的保存的一个model
                modelPath = "Epoch_" + str(count-250) + "_iters_" + str(count-250) + ".model"
                test(modelPath)
        scheduler.step()

    # 训练结束，保存最后一个模型
    model.eval()
    model.cpu()

    model_filename = ModelPath + "Final_epoch_" + str(count) + ".model"
    # args.save_model_dir = models
    torch.save(model.state_dict(), model_filename)
    print("\nDone, trained model saved at", model_filename)


if __name__ == '__main__':
    dataset = dataset(path=dataPath)
    train(dataset)

