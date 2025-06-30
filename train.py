import os
import argparse

import torch
import torch.optim as optim
import torchvision.transforms
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model8 import swin_base_patch4_window12_384_in22k as create_model
from utils8 import read_split_data, train_one_epoch, evaluate
import torchvision.transforms.functional as TF
import random
from typing import Sequence
import numpy as np
import os
from  PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] ='0'


def main(args,jiaocha):

    device = torch.device('cuda')

    if os.path.exists("./weights8/"+str(jiaocha)) is False:
        os.makedirs("./weights8/"+str(jiaocha))

    # tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label= read_split_data(jiaocha=jiaocha)
    lr= 0.00004


    img_size = 480
    img_size_h=768
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomRotation([-15, 15], expand=True,
                                       interpolation=torchvision.transforms.InterpolationMode.BICUBIC),

            transforms.RandomResizedCrop(size=(int(img_size_h),int(img_size)),ratio=(0.25,1),scale=(0.8,1.0)),
            #                          transforms.RandomCrop([img_size_h,img_size]),

                                     transforms.RandomHorizontalFlip(),
                                     # MyRotateTransform( [0,90,180,270]),
                                     # transforms.ColorJitter(brightness=0.4),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.277], [0.219])]),
        "val1": transforms.Compose([transforms.Resize([int(img_size_h * 1.04),int(img_size * 1.04)]),
                                   transforms.CenterCrop([img_size_h,img_size]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.273], [0.218])]),


    }

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset1 = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val1"])


    batch_size = 8
    # nw = min([os.cpu_count()*2, batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 8

    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn,
                                               worker_init_fn=np.random.seed(3407))

    val_loader1 = torch.utils.data.DataLoader(val_dataset1,
                                             batch_size=8,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset1.collate_fn)



    # model = create_model(num_classes=args.num_classes).to(device)
    #
    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)["model"]
    #
    #     # 删除有关分类类别的权重
    #     for k in list(weights_dict.keys()):
    #         if "head" in k:
    #             del weights_dict[k]
    #
    #
    #     # weights_dict['patch_embed.proj.weight']=torch.cat([weights_dict['patch_embed.proj.weight'][:,0,:,:].unsqueeze(1),weights_dict['patch_embed.proj.weight'][:,1,:,:].unsqueeze(1),weights_dict['patch_embed.proj.weight'][:,2,:,:].unsqueeze(1),weights_dict['patch_embed.proj.weight'][:,2,:,:].unsqueeze(1)],dim=1)
    #     weights_dict['patch_embed.proj.weight']=weights_dict['patch_embed.proj.weight'][:,0,:,:].unsqueeze(1)+weights_dict['patch_embed.proj.weight'][:,1,:,:].unsqueeze(1)+weights_dict['patch_embed.proj.weight'][:,2,:,:].unsqueeze(1)
    #     # weights_dict['norm1.weight']=weights_dict['norm.weight'][:96]
    #     # weights_dict['norm1.bias']=weights_dict['norm.bias'][:96]
    #     # weights_dict['norm2.weight']=weights_dict['norm.weight'][96:288]
    #     # weights_dict['norm2.bias']=weights_dict['norm.bias'][96:288]
    #     # weights_dict['norm3.weight']=weights_dict['norm.weight'][288:672]
    #     # weights_dict['norm3.bias']=weights_dict['norm.bias'][288:672]
    #     # weights_dict['norm.weight']=weights_dict['norm.weight'][:512]
    #     # weights_dict['norm.bias']=weights_dict['norm.bias'][:512]
    #     print(model.load_state_dict(weights_dict, strict=False))
    model = torch.load(args.weights)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=lr, weight_decay=5E-2)

    t_loss=[]
    t_acc=[]
    v_loss=[]
    v_acc=[]
    v_EMR_=0
    loss1=[]
    loss2=[]
    loss3=[]
    loss4=[]
    v_acc2=0

    for epoch in range(args.epochs):
        if (epoch+1)%50==0 :
            lr/=2

            optimizer = optim.AdamW(pg, lr=lr, weight_decay=5E-2)
            print('lr:',lr)
        # train
        train_loss, train_acc ,_= train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch
                                                  )
        t_loss.append(train_loss)
        t_acc.append(train_acc)

        # validate
        val_loss1, val_EMR1 ,val_accu1,liangexing1,gaihua1,mass1,midu1,loss11= evaluate(model=model,
                                     data_loader=val_loader1,
                                     device=device,
                                     epoch=epoch
                                                                         ,X=0)

        # val_loss, val_acc ,_,abnormal,liangexing,midu,loss55= evaluate(model=model,
        #                              data_loader=val_loader5,
        #                              device=device,
        #                              epoch=epoch)
        if(val_EMR1>v_EMR_):
            v_EMR_=val_EMR1
            v_acc2=val_accu1
            print('最好birads:',val_EMR1,jiaocha)
            torch.save(model, "./weights8/{}/model-{}.pth".format(jiaocha,epoch))
        elif val_EMR1==v_EMR_ and val_accu1>v_acc2:
            v_acc2=val_accu1
            print('最好birads再出现:',val_EMR1,jiaocha)
            torch.save(model, "./weights8/{}/model-{}.pth".format(jiaocha,epoch))

        v_loss.append(val_loss1)
        v_acc.append(val_EMR1)
        loss1.append(loss11)

        # loss5.append(loss55)
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')  # 后端设置'Agg' 参考：https://cloud.tencent.com/developer/article/1559466

        plt.figure()  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
        plt.plot(t_loss, 'b',label='t_loss')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
        plt.plot(v_loss, 'r', label='v_loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()  # 个性化图例（颜色、形状等）
        plt.savefig(os.path.join('./weights7', "1_recon_loss.jpg"))

        plt.figure()  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
        plt.plot(t_acc, 'b',label='t_acc')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
        plt.plot(v_acc, 'r', label='v_acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend()  # 个性化图例（颜色、形状等）
        plt.savefig(os.path.join('./weights7', "1_recon_acc.jpg"))

        plt.figure()  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
        plt.plot(loss1, 'b',label='t_acc')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
        plt.plot(loss2, 'r',label='t_acc')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
        plt.plot(loss3, 'y',label='t_acc')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
        plt.plot(loss4, 'g',label='t_acc')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
        # plt.plot(loss5, 'c',label='t_acc')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()  # 个性化图例（颜色、形状等）
        plt.savefig(os.path.join('./weights7', "1_recon_loss1.jpg"))
        #
        # tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)



import sys


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    #最全的
    seed=3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)  # 保证随机结果可复现
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(seed)


    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=13)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=8)


    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default=r"D:\pycharm project\dataset\cdd-cesm")
    #PATH 去utils里改

    # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default=r'D:\浏览器下载\swin_base_patch4_window12_384_22k.pth',
    #                     help='initial weights path')
    parser.add_argument('--weights', type=str, default=r'D:\pycharm project\swin2\swin_transformer\weights10\model-23.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    for i in range(10):
        print(i)
        main(opt,i)
