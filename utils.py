import numpy as np
import os
import csv
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings



def read_split_data(root = r'D:\pycharm project\dataset\cdd-cesm\ALL_DATA', val_rate: float = 0.1,jiaocha=0):


    root = r'C:\dataset\dm'
    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    xianyan=[]
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    file = []

    for files in os.listdir(root):
        if os.path.splitext(files)[-1] in supported:
            file.append(files.replace('.jpg', ''))
    images = [os.path.join(root, i) for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]
    # 按比例随机采样验证样本
    # val_path = random.sample(images, k=int(len(images) * val_rate))
    images.sort()
    i = 0
    for img_path in images:

        labels = np.zeros(9, dtype=int)
        flag = 0
        f = open(r'C:\dataset\7ALL_DATA.csv', 'r')
        csv_reader = csv.reader(f)

        for line_no, line in enumerate(csv_reader):

            if file[i] in line[0]:

                if str(line[4]) == 'D' or str(line[4]) == '4':
                    labels[8] = 3
                elif str(line[4]) == 'C'or str(line[4]) == '3':
                    labels[8] = 2
                elif str(line[4]) == 'B'or str(line[4]) == '2':
                    labels[8] = 1
                elif str(line[4]) == 'A'or str(line[4]) == '1':
                    labels[8] = 0
                else:
                    print('1')

                if str(line[3]) == 'BOTH':
                    labels[6] = 1
                    labels[7] = 1
                elif str(line[3]) == 'MASS':
                    labels[6] = 0
                    labels[7] = 1
                elif str(line[3]) == 'CALC':
                    labels[6] = 1
                    labels[7] = 0
                elif str(line[3]) == 'NONE':
                    labels[6] = 0
                    labels[7] = 0
                else:
                    print('1')


                if str(line[5]) == '1':
                    labels[5] = 1
                elif str(line[5]) == '0':
                    labels[5] = 0
                else:
                    print('1')

                labels[np.array(str(line[1]).split("$"), dtype=int) - 1] = 1
                flag = 1
        f.close()

        if flag==1:
            # if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
            if (i )%10==jiaocha:
                val_images_path.append(img_path)
                val_images_label.append(labels)

                i += 1
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(labels)
                i += 1
            xianyan.append(labels)
        else:

            #
            print(file[i])
            i += 1
    print(val_images_path)
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."
    return train_images_path, train_images_label, val_images_path, val_images_label

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    warnings.filterwarnings("ignore")
    model.train()
    # suoyoude = torch.zeros([13,4]).to(device)
    lossfunction = torch.nn.BCEWithLogitsLoss()
    loss_function2 = torch.nn.CrossEntropyLoss()
    loss_function3 = torch.nn.BCEWithLogitsLoss()
    loss_function4 = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    EMR_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    birads_precicion = torch.zeros(1).to(device)
    F1 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    hamming = torch.zeros(1).to(device)   # 累计预测正确的样本数

    accu_num =torch.zeros(1).to(device)
    accu_num2 =torch.zeros(1).to(device)#TP+TN

    accu_num4 =torch.zeros(1).to(device)
    yuzhi = torch.tensor([ 0.55]).to(device)
    train_preds2 = []
    train_trues2 = []#AD
    train_preds3 = []
    train_trues3 = []#钙化
    train_preds35 = []
    train_trues35 = []#肿块
    train_preds4 = []
    train_trues4 = []
    optimizer.zero_grad()

    sample_num = 0
    loss1 = torch.zeros(1).to(device)
    loss2 = 0
    loss3 = 0
    loss4 = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels1=labels[:,:5].to(device)
        labels2 = labels[:,5].to(device)
        labels3 = labels[:, 6:8].to(device)
        labels4 = labels[:, 8].to(device)

        sample_num += images.shape[0]

        pred ,pred_= model(images.to(device),False,epoch)

        # 将最大值添加到原始 tensor 的开头
        pred5 = torch.cat((-torch.max(pred, dim=1, keepdim=True)[0], pred), dim=1)
        pred2=pred_[:,0]
        pred3=pred_[:,1:3]
        pred4=pred_[:,3:7]

        pred_classes4 = torch.max(pred4, dim=1)[1]

        accu_num4 += torch.eq(pred_classes4, labels4).sum()



        Sigmoid =torch.nn.Sigmoid()
        acc_pred = torch.empty_like(pred5).copy_(pred5)
        acc_pred = Sigmoid(acc_pred)
#######转化钙化肿块
        acc_pred2 = torch.zeros_like(pred2)
        acc_pred3 = torch.zeros_like(pred3[:,0])
        acc_pred35 = torch.zeros_like(pred3[:,1])
        acc_pred2[pred2 > 0] = 1
        acc_pred3[pred3[:,0]>0]=1
        acc_pred35[pred3[:,1]>0]=1
#######
        ######## 收集全部数据
        train_preds2.extend(acc_pred2.detach().cpu().numpy())
        train_trues2.extend(labels2.detach().cpu().numpy())
        train_preds3.extend(acc_pred3.detach().cpu().numpy())
        train_trues3.extend(labels3[:,0].detach().cpu().numpy())
        train_preds35.extend(acc_pred35.detach().cpu().numpy())
        train_trues35.extend(labels3[:,1].detach().cpu().numpy())
        train_preds4.extend(pred_classes4.detach().cpu().numpy())
        train_trues4.extend(labels4.detach().cpu().numpy())
#################
#####
        for i in range(yuzhi.shape[0]):
            acc_pred1 = torch.empty_like(pred5).copy_(acc_pred)
            acc_pred2 = torch.zeros([images.shape[0],5],dtype=torch.float).to(device)
            for j in range(images.shape[0]):
                if acc_pred1[j][1:].topk(1)[0]<yuzhi[i]:
                    acc_pred2[j][0]=1
                # elif acc_pred[i][0]>yuzhi and acc_pred[i][1:].topk(1)[0]>yuzhi:
                #     idx=acc_pred[i].topk(1)[1]
                #     acc_pred[i].fill_(0)
                #     acc_pred[i][idx]=1
                elif acc_pred1[j][1:].topk(2)[0][1].item() > yuzhi[i]:
                    acc_pred2[j][acc_pred1[j][1:].topk(2)[1]+1] = 1
                else:
                    for k in range(1,5):
                        if acc_pred1[j][k]>= yuzhi[i].item():
                            acc_pred2[j][k]=1
            accu_num[i] += torch.eq(acc_pred2, labels1).sum()/5

            for k in range(images.shape[0]):
                if(torch.eq(acc_pred2[k],labels1[k]).sum()==5):
                    EMR_num[i] +=1
                # suoyoude[i] += labels1[k]
##F1
                F1[i] +=2 * sum(torch.logical_and(labels1[k], acc_pred2[k])) / (sum(torch.logical_and(labels1[k], acc_pred2[k]))+sum(torch.logical_or(labels1[k], acc_pred2[k])))#F1=2*p*r / p+r
##HAM
                hamming[i] += sum(torch.logical_and(labels1[k], acc_pred2[k])) / sum(torch.logical_or(labels1[k], acc_pred2[k]))
##precision
                birads_precicion[i] += sum(torch.logical_and(labels1[k], acc_pred2[k])) / sum(torch.logical_and(torch.ones_like(acc_pred2[k]), acc_pred2[k]))


        loss = 0.65*lossfunction(pred, labels1[:,1:].float().to(device)) +0.15*loss_function3(pred2, labels2.type(torch.float).to(device))+0.15*loss_function3(pred3, labels3.type(torch.float).to(device))+0.05*loss_function4(pred4, labels4.type(torch.LongTensor).to(device))
        loss.backward()
        loss1+=lossfunction(pred, labels1[:,1:].float().to(device))
        loss2+=loss_function3(pred2, labels2.type(torch.float).to(device))
        loss3 += loss_function3(pred3, labels3.type(torch.float).to(device))
        loss4+=loss_function4(pred4, labels4.type(torch.LongTensor).to(device))
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, Exact Match Ratio: {:.3f}, birads_accu: {:.3f},, birads_precision: {:.3f}, f1: {:.3f},HAMMING: {:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               EMR_num.topk(1)[0].item() / sample_num,
                                                                               accu_num.topk(1)[0].item() / sample_num,
                                                                               birads_precicion.topk(1)[0].item() / sample_num,
                                                                               F1.topk(1)[0].item() / sample_num,
                                                                               hamming.topk(1)[0].item() / sample_num,
                                                                                                                                                                                 loss1.item() / (step + 1),
                                                                                                                                                                                 loss2 / (step + 1),
                                                                                                                                                                                 loss3 / (step + 1),
                                                                                                                                                                                 loss4 / (step + 1)
                                                                                                                                                                                 )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    # import matplotlib.pyplot as plt
    # plt.switch_backend('Agg')  # 后端设置'Agg' 参考：https://cloud.tencent.com/developer/article/1559466
    #
    # plt.figure()  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
    # plt.plot(EMR_num.cpu() / sample_num, 'b',label='EMR')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
    # plt.plot(accu_num.cpu()/ sample_num, 'r', label='accu')
    # plt.ylabel('pr')
    # plt.xlabel('t')
    # plt.legend()  # 个性化图例（颜色、形状等）
    # plt.savefig(os.path.join('./train_pr', "pr_{}.jpg".format(epoch)))
    # # print(suoyoude[0])

    ########################评估指标
    AD_accuracy = accuracy_score(train_trues2, train_preds2)
    AD_precision = precision_score(train_trues2, train_preds2, average='weighted')
    AD_recall = recall_score(train_trues2, train_preds2, average='weighted')
    AD_f1 = f1_score(train_trues2, train_preds2, average='weighted')

    gaihua_accuracy = accuracy_score(train_trues3, train_preds3)
    gaihua_precision = precision_score(train_trues3, train_preds3)
    gaihua_recall = recall_score(train_trues3, train_preds3)
    gaihua_f1 = f1_score(train_trues3, train_preds3)

    mass_accuracy = accuracy_score(train_trues35, train_preds35)
    mass_precision = precision_score(train_trues35, train_preds35)
    mass_recall = recall_score(train_trues35, train_preds35)
    mass_f1 = f1_score(train_trues35, train_preds35)

    midu_accuracy = accuracy_score(train_trues4, train_preds4)
    midu_precision = precision_score(train_trues4, train_preds4, average='weighted')
    midu_recall = recall_score(train_trues4, train_preds4, average='weighted')
    midu_f1 = f1_score(train_trues4, train_preds4, average='weighted')
    print('\033[3m train-epoch-{},AD_acc:{:.3f},AD_precision:{:.3f},AD_recall:{:.3f},ADf1:{:.3f}'.format(epoch,AD_accuracy,AD_precision,AD_recall,AD_f1),end=' ')
    print('\033[3m 钙化_acc:{:.3f},钙化_precision:{:.3f},钙化_recall:{:.3f},钙化f1:{:.3f}'.format(gaihua_accuracy,gaihua_precision,gaihua_recall,gaihua_f1),end=' ')
    print('\033[3m 肿块_acc:{:.3f},肿块_precision:{:.3f},肿块_recall:{:.3f},肿块f1:{:.3f}'.format(mass_accuracy,mass_precision,mass_recall,mass_f1),end=' ')
    print('\033[3m 密度_acc:{:.3f},密度_precision:{:.3f},密度_recall:{:.3f},密度f1:{:.3f}'.format(midu_accuracy,midu_precision,midu_recall,midu_f1))
    ########################

    return accu_loss.item() / (step + 1), EMR_num.topk(1)[0].item() / sample_num ,accu_num.topk(1)[0].item() / sample_num
import scipy
def random_num(size,end):
    range_ls=[i for i in range(end)]
    num_ls=[]
    for i in range(size):
        num=random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls

@torch.no_grad()
def evaluate(model, data_loader, device, epoch
             ,X,wshow=False):
    model.eval()
    lossfunction = torch.nn.BCEWithLogitsLoss()
    loss_function3 = torch.nn.BCEWithLogitsLoss()
    loss_function4 = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    EMR_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    birads_precicion = torch.zeros(1).to(device)
    F1 = torch.zeros(1).to(device)   # 累计预测正确的样本数
    hamming = torch.zeros(1).to(device)   # 累计预测正确的样本数

    accu_num =torch.zeros(1).to(device)
    accu_num2 =torch.zeros(1).to(device)#TP+TN

    accu_num4 =torch.zeros(1).to(device)
    yuzhi = torch.tensor([ 0.55]).to(device)
    train_preds2 = []
    train_trues2 = []#AD
    train_preds3 = []
    train_trues3 = []#钙化
    train_preds35 = []
    train_trues35 = []#肿块
    train_preds4 = []
    train_trues4 = []
    # cuowude = torch.zeros([13,4]).to(device)
    suoyoude = torch.zeros([1,4]).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    loss1 = torch.zeros(1).to(device)
    loss2 = 0
    loss3 = 0
    loss4 = 0
    label_roc0=[]
    pre_roc0=[]
    label_roc1=[]
    pre_roc1=[]
    label_roc2=[]
    pre_roc2=[]
    label_roc3=[]
    pre_roc3=[]
    for step, data in enumerate(data_loader):
        images, labels = data
        labels1=labels[:,:5].to(device)
        labels2 = labels[:,5].to(device)
        labels3 = labels[:,6:8].to(device)
        labels4 = labels[:,8].to(device)

        sample_num += images.shape[0]

        pred ,pred_= model(images.to(device))

        label_roc0.extend(labels[:,0].detach().cpu().numpy())
        pre_roc0.extend(pred[:,0].detach().cpu().numpy())

        label_roc1.extend(labels[:,1].detach().cpu().numpy())
        pre_roc1.extend(pred[:,1].detach().cpu().numpy())

        label_roc2.extend(labels[:,2].detach().cpu().numpy())
        pre_roc2.extend(pred[:,2].detach().cpu().numpy())

        label_roc3.extend(labels[:,3].detach().cpu().numpy())
        pre_roc3.extend(pred[:,3].detach().cpu().numpy())


        pred2=pred_[:,0]
        pred3=pred_[:,1:3]
        pred4=pred_[:,3:7]

        pred5 = torch.cat((-torch.max(pred, dim=1, keepdim=True)[0], pred), dim=1)
        pred_classes4 = torch.max(pred4, dim=1)[1]
        accu_num4 += torch.eq(pred_classes4, labels4).sum()

        Sigmoid =torch.nn.Sigmoid()
        acc_pred = torch.empty_like(pred5).copy_(pred5)
        acc_pred = Sigmoid(acc_pred)

        #######转化钙化肿块
        acc_pred2 = torch.zeros_like(pred2)
        acc_pred3 = torch.zeros_like(pred3[:, 0])
        acc_pred35 = torch.zeros_like(pred3[:, 1])
        acc_pred2[pred2 > 0] = 1
        acc_pred3[pred3[:, 0] > 0] = 1
        acc_pred35[pred3[:, 1] > 0] = 1
        #######
        ######## 收集全部数据
        train_preds2.extend(acc_pred2.detach().cpu().numpy())
        train_trues2.extend(labels2.detach().cpu().numpy())
        train_preds3.extend(acc_pred3.detach().cpu().numpy())
        train_trues3.extend(labels3[:, 0].detach().cpu().numpy())
        train_preds35.extend(acc_pred35.detach().cpu().numpy())
        train_trues35.extend(labels3[:, 1].detach().cpu().numpy())
        train_preds4.extend(pred_classes4.detach().cpu().numpy())
        train_trues4.extend(labels4.detach().cpu().numpy())
        #################

        for i in range(yuzhi.shape[0]):
            acc_pred1 = torch.empty_like(pred5).copy_(acc_pred)
            acc_pred2 = torch.zeros([images.shape[0],5],dtype=torch.float).to(device)
            for j in range(images.shape[0]):
                if acc_pred1[j][1:].topk(1)[0] < yuzhi[i]:
                    acc_pred2[j][0] = 1
                # elif acc_pred[i][0]>yuzhi and acc_pred[i][1:].topk(1)[0]>yuzhi:
                #     idx=acc_pred[i].topk(1)[1]
                #     acc_pred[i].fill_(0)
                #     acc_pred[i][idx]=1
                elif acc_pred1[j][1:].topk(2)[0][1].item() > yuzhi[i]:
                    acc_pred2[j][acc_pred1[j][1:].topk(2)[1] + 1] = 1
                else:
                    for k in range(1, 5):
                        if acc_pred1[j][k] >= yuzhi[i].item():
                            acc_pred2[j][k] = 1

            accu_num[i] += torch.eq(acc_pred2, labels1).sum()/5

            for k in range(images.shape[0]):##EMR
                if(torch.eq(acc_pred2[k],labels1[k]).sum()==5):
                    EMR_num[i] +=1
                # else:
                #     cuowude[i]+=labels1[k]
                # suoyoude[i] += labels1[k]
##F1
                F1[i] +=2 * sum(torch.logical_and(labels1[k], acc_pred2[k])) / (sum(torch.logical_and(labels1[k], acc_pred2[k]))+sum(torch.logical_or(labels1[k], acc_pred2[k])))#F1=2*p*r / p+r
##HAM
                hamming[i] += sum(torch.logical_and(labels1[k], acc_pred2[k])) / sum(torch.logical_or(labels1[k], acc_pred2[k]))
##precision
                birads_precicion[i] += sum(torch.logical_and(labels1[k], acc_pred2[k])) / sum(torch.logical_and(torch.ones_like(acc_pred2[k]), acc_pred2[k]))

        loss = 0.65*lossfunction(pred, labels1[:,1:].float().to(device)) +0.15*loss_function3(pred2, labels2.type(torch.float).to(device))+0.15*loss_function3(pred3, labels3.type(torch.float).to(device))+0.05*loss_function4(pred4, labels4.type(torch.LongTensor).to(device))
        accu_loss += loss

        loss1+=lossfunction(pred, labels1[:,1:].float().to(device))
        loss2+=loss_function3(pred3, labels3.type(torch.float).to(device))
        loss3 += loss_function3(pred3, labels3.type(torch.float).to(device))
        loss4+=loss_function4(pred4, labels4.type(torch.LongTensor).to(device))


        data_loader.desc = "[valid epoch {}] loss: {:.3f}, Exact Match Ratio: {:.3f}, birads_accu: {:.3f}, precicion: {:.3f}, f1: {:.3f}, hamming: {:.3f}，loss1: {:.3f}，loss2: {:.3f}，loss3: {:.3f}，loss4: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               EMR_num.topk(1)[0].item() / sample_num,
                                                                               accu_num[EMR_num.topk(1)[1]].item() / sample_num,
                                                                               birads_precicion[EMR_num.topk(1)[1]].item() / sample_num,
                                                                               F1[EMR_num.topk(1)[1]].item() / sample_num,
                                                                               hamming[EMR_num.topk(1)[1]].item() / sample_num,
        loss1.item()/(step + 1),
        loss2/(step + 1),
        loss3/(step + 1),
        loss4/(step + 1)
        )


    # plt.switch_backend('Agg')  # 后端设置'Agg' 参考：https://cloud.tencent.com/developer/article/1559466
    #
    # plt.figure()  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
    # plt.plot(EMR_num.cpu() / sample_num, 'b',label='EMR')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
    # plt.plot(accu_num.cpu()/ sample_num, 'r', label='accu')
    # plt.ylabel('pr')
    # plt.xlabel('t')
    # plt.legend()  # 个性化图例（颜色、形状等）
    # plt.savefig(os.path.join('./val_pr', "pr_{}.jpg".format(epoch)))
    # print(cuowude[EMR_num.topk(1)[1]],suoyoude[0])
    # plt.close()

    ########################评估指标
    AD_accuracy = accuracy_score(train_trues2, train_preds2)
    AD_precision = precision_score(train_trues2, train_preds2, average='weighted')
    AD_recall = recall_score(train_trues2, train_preds2, average='weighted')
    AD_f1 = f1_score(train_trues2, train_preds2, average='weighted')

    gaihua_accuracy = accuracy_score(train_trues3, train_preds3)
    gaihua_precision = precision_score(train_trues3, train_preds3)
    gaihua_recall = recall_score(train_trues3, train_preds3)
    gaihua_f1 = f1_score(train_trues3, train_preds3)

    mass_accuracy = accuracy_score(train_trues35, train_preds35)
    mass_precision = precision_score(train_trues35, train_preds35)
    mass_recall = recall_score(train_trues35, train_preds35)
    mass_f1 = f1_score(train_trues35, train_preds35)

    midu_accuracy = accuracy_score(train_trues4, train_preds4)
    midu_precision = precision_score(train_trues4, train_preds4, average='weighted')
    midu_recall = recall_score(train_trues4, train_preds4, average='weighted')
    midu_f1 = f1_score(train_trues4, train_preds4, average='weighted')
    print('\033[3m valid-epoch-{},AD_acc:{:.3f},AD_precision:{:.3f},AD_recall:{:.3f},ADf1:{:.3f}'.format(epoch,AD_accuracy,AD_precision,AD_recall,AD_f1),end=' ')
    print('\033[3m 钙化_acc:{:.3f},钙化_precision:{:.3f},钙化_recall:{:.3f},钙化f1:{:.3f}'.format(gaihua_accuracy,gaihua_precision,gaihua_recall,gaihua_f1),end=' ')
    print('\033[3m 肿块_acc:{:.3f},肿块_precision:{:.3f},肿块_recall:{:.3f},肿块f1:{:.3f}'.format(mass_accuracy,mass_precision,mass_recall,mass_f1),end=' ')
    print('\033[3m 密度_acc:{:.3f},密度_precision:{:.3f},密度_recall:{:.3f},密度f1:{:.3f}'.format(midu_accuracy,midu_precision,midu_recall,midu_f1))
    ########################
    ########################ROC
    return accu_loss.item() / (step + 1), EMR_num.topk(1)[0].item() / sample_num ,accu_num[EMR_num.topk(1)[1]].item() / sample_num ,accu_num2.item() / sample_num, gaihua_accuracy,mass_accuracy, accu_num4.item() / sample_num ,loss1.item()/ (step + 1)
