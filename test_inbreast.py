import os
import json
import argparse
import sys
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from my_dataset import MyDataSet

import csv
from sklearn.utils import resample

def compute_auc_ci(y_true, y_scores, n_bootstraps=1000, ci=0.95):
    """
    Compute the AUC and its confidence interval using bootstrap.

    Parameters:
    y_true (torch.Tensor): True binary labels as a tensor.
    y_scores (torch.Tensor): Target scores as a tensor.
    n_bootstraps (int): Number of bootstrap samples to use.
    ci (float): Confidence interval (0 < ci < 1).

    Returns:
    auc_mean (float): Mean AUC.
    auc_lower (float): Lower bound of the confidence interval.
    auc_upper (float): Upper bound of the confidence interval.
    """
    # Convert tensors to numpy arrays
    y_true_np = y_true.cpu().numpy()
    y_scores_np = y_scores.cpu().numpy()

    # Array to store the bootstrap samples
    bootstrapped_aucs = []

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Bootstrap sampling
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = resample(np.arange(len(y_true_np)), replace=True, n_samples=len(y_true_np))
        if len(np.unique(y_true_np[indices])) < 2:
            # If the sample has only one class, skip the iteration
            continue
        score = roc_auc_score(y_true_np[indices], y_scores_np[indices])
        bootstrapped_aucs.append(score)

    # Compute the mean AUC
    auc_mean = np.mean(bootstrapped_aucs)

    # Compute the lower and upper bounds of the confidence interval
    sorted_aucs = np.sort(bootstrapped_aucs)
    lower_bound = np.percentile(sorted_aucs, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(sorted_aucs, (1 + ci) / 2 * 100)

    return auc_mean, lower_bound, upper_bound

from sklearn.preprocessing import label_binarize


def compute_multiclass_auc_ci(y_true, y_scores, n_classes=4, n_bootstraps=1000, ci=0.95):
    """
    Compute the aggregate AUC and its confidence interval for multi-class classification using bootstrap.

    Parameters:
    y_true (torch.Tensor): True labels as a tensor (shape: [n_samples]).
    y_scores (torch.Tensor): Target scores as a tensor (shape: [n_samples, n_classes]).
    n_classes (int): Number of classes.
    n_bootstraps (int): Number of bootstrap samples to use.
    ci (float): Confidence interval (0 < ci < 1).

    Returns:
    auc_mean (float): Mean aggregate AUC.
    auc_lower (float): Lower bound of the confidence interval.
    auc_upper (float): Upper bound of the confidence interval.
    """
    # Convert tensors to numpy arrays
    y_true_np = y_true.cpu().numpy()
    y_scores_np = y_scores.cpu().numpy()

    # Binarize the output for multiclass AUC calculation
    y_true_binarized = label_binarize(y_true_np, classes=np.arange(n_classes))

    # Array to store bootstrap AUCs
    bootstrapped_aucs = []

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Bootstrap sampling
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = resample(np.arange(len(y_true_np)), replace=True, n_samples=len(y_true_np))

        # Calculate the micro-average ROC AUC score
        try:
            auc_score = roc_auc_score(y_true_binarized[indices], y_scores_np[indices], average='micro',
                                      multi_class='ovr')
            bootstrapped_aucs.append(auc_score)
        except ValueError:
            # If we encounter an issue due to lack of positive or negative examples, skip this iteration
            continue

    # Compute the mean AUC
    auc_mean = np.mean(bootstrapped_aucs)

    # Compute the lower and upper bounds of the confidence interval
    sorted_aucs = np.sort(bootstrapped_aucs)
    lower_bound = np.percentile(sorted_aucs, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(sorted_aucs, (1 + ci) / 2 * 100)

    return auc_mean, lower_bound, upper_bound

def read_split_data(root = r'D:\pycharm project\dataset\cdd-cesm\ALL_DATA', val_rate: float = 0.1):


    root = r'C:\dataset\inbreast'
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
        f = open(r'C:\dataset\inbreast.csv', 'r')
        csv_reader = csv.reader(f)

        for line_no, line in enumerate(csv_reader):

            if file[i] in line[0]:

                if str(line[5]) == 'D' or str(line[5]) == '4':
                    labels[8] = 3
                elif str(line[5]) == 'C'or str(line[5]) == '3':
                    labels[8] = 2
                elif str(line[5]) == 'B'or str(line[5]) == '2':
                    labels[8] = 1
                elif str(line[5]) == 'A'or str(line[5]) == '1':
                    labels[8] = 0
                else:
                    print('1')

                if str(line[2]) == '1':#mass
                    labels[7] = 1
                elif str(line[2]) == '0':#mass
                    labels[7] = 0
                else:
                    print('1')

                if str(line[3]) == '1':#calc
                    labels[6] = 1
                elif str(line[3]) == '0':
                    labels[6] = 0
                else:
                    print('1')

                if str(line[4]) == '1':#ad
                    labels[5] = 1
                elif str(line[4]) == '0':#ad
                    labels[5] = 0
                else:
                    print('1')


                labels[np.array(str(line[1]).split("$"), dtype=int) - 1] = 1
                flag = 1
                if str(line[6]) == 'test':
                    trainortest = 1
                elif str(line[6]) == 'train':
                    trainortest = 0
        f.close()

        if flag==1:
            # if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
            if trainortest == 1:
                val_images_path.append(img_path)
                val_images_label.append(labels)
                i += 1
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(labels)
                i += 1
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
@torch.no_grad()
def main(args, jiaocha=None,weights=None):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data()

    # img_size = 226

    img_size = 480
    img_size_h=768
    data_transform = {
        "val": transforms.Compose([transforms.Resize([int(img_size_h * 1.04),int(img_size * 1.04)]),
                                   transforms.CenterCrop([img_size_h,img_size]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.397], [0.257])])}

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = 1

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=8,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             worker_init_fn=np.random.seed(3407))
    model = torch.load(weights)

    model.eval()


    data_loader = val_loader
    labels2=torch.zeros([1,9])
    pred2=torch.zeros([1,4]).to(device)
    pred2_=torch.zeros([1,7]).to(device)



    for step, data in enumerate(data_loader):
        images, labels = data
        labels2=torch.cat([labels2,labels],dim=0)
        pred, pred_= model(images.to(device))
        pred2=torch.cat([pred2,pred],dim=0)
        pred2_=torch.cat([pred2_,pred_],dim=0)



    return labels2[1:,:],pred2[1:,:],pred2_[1:,:]

def pinggu(a, b, c):
    accu_num = torch.zeros([1]).to(device)
    accu_num2 = torch.zeros(1).to(device)
    accu_num4 = torch.zeros(1).to(device)
    # yuzhi = torch.tensor([0.25,0.3,0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]).to(device)
    EMR_num = torch.zeros([1]).to(device)  # 累计预测正确的样本数
    EMR_num2 = torch.zeros([1]).to(device)  # 累计预测正确的样本数
    birads_precicion = torch.zeros([1]).to(device)
    F1 = torch.zeros([1]).to(device)
    hamming = torch.zeros([1]).to(device)
    sample_num = 0
    train_preds2 = []
    train_trues2 = []#ad
    train_preds3 = []
    train_trues3 = []#钙化
    train_preds35 = []
    train_trues35 = []#肿块
    train_preds4 = []
    train_trues4 = []

    TN_DULI = torch.zeros([5]).to(device)

    TNFP_DULI = torch.zeros([5]).to(device)+0.0000001

    labels=a
    pred=b
    pred_=c
    print(labels.shape)


    labels1 = labels[:, :5].to(device)
    labels2 = labels[:, 5].to(device)

    labels3 = labels[:, 6:8].to(device)
    labels31 = labels3[:, 0]
    labels32 = labels3[:, 1]
    labels4 = labels[:, 8].to(device)
    labels41=np.zeros([len(labels4),4])
    for i, label in enumerate(labels4.cpu()):
        labels41[i, int(label)]=1

    pred2 = pred_[:, 0]

    pred3 = pred_[:, 1:3]

    pred31=pred3[:, 0]
    pred32=pred3[:, 1]
    pred4 = pred_[:, 3:7]
    pred5 = torch.cat((-torch.max(pred, dim=1, keepdim=True)[0], pred), dim=1)
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fpr2, tpr2, _ = roc_curve(labels2.cpu(), pred2.cpu())
    fpr310, tpr310, _ = roc_curve(labels31.cpu(), pred31.cpu())
    fpr320, tpr320, _ = roc_curve(labels32.cpu(), pred32.cpu())
    fpr411, tpr411, _ = roc_curve(labels41[:,0], pred4[:,0].cpu())
    fpr412, tpr412, _ = roc_curve(labels41[:,1], pred4[:,1].cpu())
    fpr413, tpr413, _ = roc_curve(labels41[:,2], pred4[:,2].cpu())
    fpr414, tpr414, _ = roc_curve(labels41[:,3], pred4[:,3].cpu())
    roc_auc2 = auc(fpr2, tpr2)
    roc_auc310 = auc(fpr310, tpr310)
    roc_auc320 = auc(fpr320, tpr320)

    roc_auc411 = auc(fpr411, tpr411)
    roc_auc412 = auc(fpr412, tpr412)
    roc_auc413 = auc(fpr413, tpr413)
    roc_auc414 = auc(fpr414, tpr414)

    all_fpr = np.unique(np.concatenate((fpr411,fpr412,fpr413,fpr414)))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    mean_tpr += np.interp(all_fpr, fpr411, tpr411)
    mean_tpr += np.interp(all_fpr, fpr412, tpr412)
    mean_tpr += np.interp(all_fpr, fpr413, tpr413)
    mean_tpr += np.interp(all_fpr, fpr414, tpr414)
    mean_tpr /= 4
    fpr_4 = all_fpr
    tpr_4 = mean_tpr
    roc_auc_4 = auc(fpr_4, tpr_4)

    # #########
    plt.figure()
    plt.title('Receiver Operating Characteristic',fontsize=14)
    # plt.rcParams['figure.figsize'] = (30.0, 30.0)
    plt.rcParams['image.interpolation'] = 'bilinear'
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置标题大小
    plt.rcParams['font.size'] = '0'
    plt.plot(fpr2, tpr2,  color='r', linestyle='-', linewidth=2, markerfacecolor='none',
             label=u'ROC curve of AD (AUC = %0.2f)' % roc_auc2)
    plt.plot(fpr310, tpr310,  color='y', linestyle='-', linewidth=2, markerfacecolor='none',
             label=u'ROC curve of CALC (AUC = %0.2f)' % roc_auc310)
    plt.plot(fpr320, tpr320,  color='g', linestyle='-', linewidth=2, markerfacecolor='none',
             label=u'ROC curve of Mass  (AUC = %0.2f)' % roc_auc320)

    plt.legend(loc='lower right',prop = {'size':14})
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate' ,fontdict={'family' : 'Times New Roman', 'size'   : 14})
    plt.xlabel('False Positive Rate', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.xticks(fontproperties='Times New Roman', size=18)
    plt.grid(linestyle='-.')
    plt.grid(True)
    plt.savefig('plt/findings.jpg'%(),dpi=300)
    plt.close()

    print('AD auc %0.3f'% roc_auc2)
    print('CALC auc %0.3f'% roc_auc310)
    print('MASS auc %0.3f' % roc_auc320)

    print(compute_auc_ci(labels2.cpu(), pred2.cpu()))
    print(compute_auc_ci(labels31.cpu(), pred31.cpu()))
    print(compute_auc_ci(labels32.cpu(), pred32.cpu()))
    print(compute_multiclass_auc_ci(torch.tensor(labels41).cpu(), pred4.cpu()))

    plt.figure()
    plt.title('Receiver Operating Characteristic',fontsize=14)
    # plt.rcParams['figure.figsize'] = (30.0, 30.0)
    plt.rcParams['image.interpolation'] = 'bilinear'
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置标题大小
    plt.rcParams['font.size'] = '0'
    plt.plot(fpr411, tpr411,  color='r', linestyle='-', linewidth=2, markerfacecolor='none',
             label=u'ROC curve of Density A (AUC = %0.2f)' % roc_auc411)
    plt.plot(fpr412, tpr412,  color='y', linestyle='-', linewidth=2, markerfacecolor='none',
             label=u'ROC curve of Density B (AUC = %0.2f)' % roc_auc412)
    plt.plot(fpr413, tpr413,  color='g', linestyle='-', linewidth=2, markerfacecolor='none',
             label=u'ROC curve of Density C  (AUC = %0.2f)' % roc_auc413)
    plt.plot(fpr414, tpr414,  color='b', linestyle='-', linewidth=2, markerfacecolor='none',
             label=u'ROC curve of Density D  (AUC = %0.2f)' % roc_auc414)
    plt.plot(fpr_4, tpr_4,  color='c', linestyle='dashed', linewidth=2, markerfacecolor='none',
             label=u'macro-average ROC curve (AUC = %0.2f)' % roc_auc_4)
    plt.legend(loc='lower right',prop = {'size':14})
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate' ,fontdict={'family' : 'Times New Roman', 'size'   : 14})
    plt.xlabel('False Positive Rate', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.xticks(fontproperties='Times New Roman', size=18)
    plt.grid(linestyle='-.')
    plt.grid(True)
    plt.savefig('plt/midu.jpg'%(),dpi=300)
    plt.close()
    print('MIDU auc %0.3f'% roc_auc_4)




    pred_classes4 = torch.max(pred4, dim=1)[1]
    accu_num4 += torch.eq(pred_classes4, labels4).sum()
    #######转化钙化肿块
    acc_pred20 = torch.zeros_like(pred2)
    acc_pred3 = torch.zeros_like(pred3[:, 0])
    acc_pred35 = torch.zeros_like(pred3[:, 1])

    acc_pred20[pred2 > -1.5] = 1
    acc_pred3[pred3[:, 0] > 0] = 1
    acc_pred35[pred3[:, 1] > 0] = 1
    #######
    ######## 收集全部数据
    train_preds2.extend(acc_pred20.detach().cpu().numpy())
    train_trues2.extend(labels2.detach().cpu().numpy())
    train_preds3.extend(acc_pred3.detach().cpu().numpy())
    train_trues3.extend(labels3[:, 0].detach().cpu().numpy())
    train_preds35.extend(acc_pred35.detach().cpu().numpy())
    train_trues35.extend(labels3[:, 1].detach().cpu().numpy())
    train_preds4.extend(pred_classes4.detach().cpu().numpy())
    train_trues4.extend(labels4.detach().cpu().numpy())

    sample_num += labels.shape[0]


    Sigmoid = torch.nn.Sigmoid()
    acc_pred = torch.empty_like(pred5).copy_(pred5)
    acc_pred = Sigmoid(acc_pred)
    with torch.no_grad():


        acc_pred1 = torch.empty_like(pred5).copy_(acc_pred)
        acc_pred2 = torch.zeros([labels.shape[0], 5], dtype=torch.float).to(device)
        yuzhi=0.55
        for j in range(labels.shape[0]):
            if acc_pred1[j][1:].topk(1)[0] < yuzhi:
                acc_pred2[j][0] = 1
            # elif acc_pred[i][0]>yuzhi and acc_pred[i][1:].topk(1)[0]>yuzhi:
            #     idx=acc_pred[i].topk(1)[1]
            #     acc_pred[i].fill_(0)
            #     acc_pred[i][idx]=1
            elif acc_pred1[j][1:].topk(2)[0][1].item() > yuzhi:
                acc_pred2[j][acc_pred1[j][1:].topk(2)[1] + 1] = 1
            else:
                for k in range(1, 5):
                    if acc_pred1[j][k] >= yuzhi:
                        acc_pred2[j][k] = 1
            acc_pred3=acc_pred2.cpu()
        accu_num+= torch.eq(acc_pred2, labels1).sum() / 5

        for k in range(labels.shape[0]):
            if (torch.eq(acc_pred2[k], labels1[k]).sum() == 5):
                EMR_num+= 1
                if train_preds2[k]==train_trues2[k] and train_preds3[k]==train_trues3[k] and train_preds35[k]==train_trues35[k] and train_preds4[k]==train_trues4[k]:
                    EMR_num2+=1
            # suoyoude[i] += labels1[k]
            ##F1
            F1+= 2 * sum(torch.logical_and(labels1[k], acc_pred2[k])) / (
                        sum(torch.logical_and(labels1[k], acc_pred2[k])) + sum(
                    torch.logical_or(labels1[k], acc_pred2[k])))  # F1=2*p*r / p+r
            ##HAM
            hamming += sum(torch.logical_and(labels1[k], acc_pred2[k])) / sum(
                torch.logical_or(labels1[k], acc_pred2[k]))
            ##precision
            birads_precicion+= sum(torch.logical_and(labels1[k], acc_pred2[k])) / sum(torch.logical_and(torch.ones_like(acc_pred2[k]), acc_pred2[k]))
            for z in range(5):
                # ACCURATE_DULI[z, i] += torch.eq(acc_pred2[:, z], labels1[:, z]).sum()  # TP+TN
                # TP_DULI[z, i] += sum(torch.logical_and(labels1[:, z], acc_pred2[:, z]))  # TP
                TN_DULI[z] += (labels1.shape[0]-sum(torch.logical_or(labels1[:, z], acc_pred2[:, z])))
                # TPFP_DULI[z, i] += sum(acc_pred2[:, z])  # TP+FP
                # TPFN_DULI[z, i] += sum(labels1[:, z])  # TP+FP
                TNFP_DULI[z] +=( labels1.shape[0]-sum(labels1[:, z]) ) # TP+FP
        TP = torch.logical_and(labels1 == 1, acc_pred2 == 1).sum()
        TN = torch.logical_and(labels1 == 0, acc_pred2 == 0).sum()
        FN = torch.logical_and(labels1 == 1, acc_pred2 == 0).sum()
        # False positives (FP): The label is negative, but the prediction is positive
        FP = torch.logical_and(labels1 == 0, acc_pred2 == 1).sum()
        # Precision = TP / (TP + FP)
        precision = TP.float() / (TP + FP).float()

        # Recall = TP / (TP + FN)
        recall = TP.float() / (TP + FN).float()

        # F1 = 2 * (precision * recall) / (precision + recall)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Handle cases where TP + FP or TP + FN is 0
        f1_score[torch.isnan(f1_score)] = 0

        # Overall Specificity = TN / (TN + FP)
        overall_specificity = TN.float() / (TN + FP).float()
        print('pre:',precision)
        print('recall:',recall)
        print('F1:',f1_score)
        print('SPE:',overall_specificity)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score






    AD_accuracy = accuracy_score(train_trues2, train_preds2)
    AD_precision = precision_score(train_trues2, train_preds2)
    AD_recall = recall_score(train_trues2, train_preds2)
    AD_f1 = f1_score(train_trues2, train_preds2)

    gaihua_accuracy = accuracy_score(train_trues3, train_preds3)
    gaihua_precision = precision_score(train_trues3, train_preds3)
    gaihua_recall = recall_score(train_trues3, train_preds3)
    gaihua_f1 = f1_score(train_trues3, train_preds3)

    mass_accuracy = accuracy_score(train_trues35, train_preds35)
    mass_precision = precision_score(train_trues35, train_preds35)
    mass_recall = recall_score(train_trues35, train_preds35)
    mass_f1 = f1_score(train_trues35, train_preds35)

    midu_accuracy = accuracy_score(train_trues4, train_preds4)
    midu_precision = precision_score(train_trues4, train_preds4, average='macro')
    midu_recall = recall_score(train_trues4, train_preds4, average='macro')
    midu_f1 = f1_score(train_trues4, train_preds4, average='macro')
    print('\033[3m AD_acc:{:.3f},AD_precision:{:.3f},AD_recall:{:.3f},ADf1:{:.3f}'.format(AD_accuracy,AD_precision,AD_recall,AD_f1),end=' ')
    print('\033[3m 钙化_acc:{:.3f},钙化_precision:{:.3f},钙化_recall:{:.3f},钙化f1:{:.3f}'.format(gaihua_accuracy,gaihua_precision,gaihua_recall,gaihua_f1),end=' ')
    print('\033[3m 肿块_acc:{:.3f},肿块_precision:{:.3f},肿块_recall:{:.3f},肿块f1:{:.3f}'.format(mass_accuracy,mass_precision,mass_recall,mass_f1),end=' ')
    print('\033[3m 密度_acc:{:.3f},密度_precision:{:.3f},密度_recall:{:.3f},密度f1:{:.3f}'.format(midu_accuracy,midu_precision,midu_recall,midu_f1))
    print()
    print('birads:',EMR_num.topk(1)[0].item() / sample_num,accu_num[EMR_num.topk(1)[1]].item() / sample_num,
                                                                           birads_precicion[EMR_num.topk(1)[1]].item() / sample_num,
                                                                           F1[EMR_num.topk(1)[1]].item() / sample_num,
                                                                  hamming[EMR_num.topk(1)[1]].item() / sample_num,EMR_num.topk(1)[1])
    print(EMR_num2.topk(1)[0].item() / sample_num  )
    print('accurate0',accuracy_score(labels1[:,0].cpu(),acc_pred3[:,0]),end=' ')
    print('accurate1',accuracy_score(labels1[:,1].cpu(),acc_pred3[:,1]),end=' ')
    print('accurate2',accuracy_score(labels1[:,2].cpu(),acc_pred3[:,2]),end=' ')
    print('accurate3',accuracy_score(labels1[:,3].cpu(),acc_pred3[:,3]),end=' ')
    print('accurate4',accuracy_score(labels1[:,4].cpu(),acc_pred3[:,4]))
    print((accuracy_score(labels1[:,0].cpu(),acc_pred3[:,0])+accuracy_score(labels1[:,1].cpu(),acc_pred3[:,1])+accuracy_score(labels1[:,2].cpu(),acc_pred3[:,2])+accuracy_score(labels1[:,3].cpu(),acc_pred3[:,3])+accuracy_score(labels1[:,4].cpu(),acc_pred3[:,4]))/5)

    print('precision0',precision_score(labels1[:,0].cpu(),acc_pred3[:,0]),end=' ')
    print('precision1',precision_score(labels1[:,1].cpu(),acc_pred3[:,1]),end=' ')
    print('precision2',precision_score(labels1[:,2].cpu(),acc_pred3[:,2]),end=' ')
    print('precision3',precision_score(labels1[:,3].cpu(),acc_pred3[:,3]),end=' ')
    print('precision4',precision_score(labels1[:,4].cpu(),acc_pred3[:,4]))
    print((precision_score(labels1[:,0].cpu(),acc_pred3[:,0])+precision_score(labels1[:,1].cpu(),acc_pred3[:,1])+precision_score(labels1[:,2].cpu(),acc_pred3[:,2])+precision_score(labels1[:,3].cpu(),acc_pred3[:,3])+precision_score(labels1[:,4].cpu(),acc_pred3[:,4]))/5)

    print('recall0',recall_score(labels1[:,0].cpu(),acc_pred3[:,0]),end=' ')
    print('recall1',recall_score(labels1[:,1].cpu(),acc_pred3[:,1]),end=' ')
    print('recall2',recall_score(labels1[:,2].cpu(),acc_pred3[:,2]),end=' ')
    print('recall3',recall_score(labels1[:,3].cpu(),acc_pred3[:,3]),end=' ')
    print('recall4',recall_score(labels1[:,4].cpu(),acc_pred3[:,4]))
    print((recall_score(labels1[:,0].cpu(),acc_pred3[:,0])+recall_score(labels1[:,1].cpu(),acc_pred3[:,1])+recall_score(labels1[:,2].cpu(),acc_pred3[:,2])+recall_score(labels1[:,3].cpu(),acc_pred3[:,3])+recall_score(labels1[:,4].cpu(),acc_pred3[:,4]))/5)

    print('Specificity0',TN_DULI[0].item() / TNFP_DULI[0].item(),end=' ')
    print('Specificity1',TN_DULI[1].item() / TNFP_DULI[1].item(),end=' ')
    print('Specificity2',TN_DULI[2].item() / TNFP_DULI[2].item(),end=' ')
    print('Specificity3',TN_DULI[3].item() / TNFP_DULI[3].item(),end=' ')
    print('Specificity4',TN_DULI[4].item() / TNFP_DULI[4].item())

    print('F10',f1_score(labels1[:,0].cpu(),acc_pred3[:,0]),end=' ')
    print('F11',f1_score(labels1[:,1].cpu(),acc_pred3[:,1]),end=' ')
    print('F12',f1_score(labels1[:,2].cpu(),acc_pred3[:,2]),end=' ')
    print('F13',f1_score(labels1[:,3].cpu(),acc_pred3[:,3]),end=' ')
    print('F14',f1_score(labels1[:,4].cpu(),acc_pred3[:,4]))
    print((f1_score(labels1[:,0].cpu(),acc_pred3[:,0])+f1_score(labels1[:,1].cpu(),acc_pred3[:,1])+f1_score(labels1[:,2].cpu(),acc_pred3[:,2])+f1_score(labels1[:,3].cpu(),acc_pred3[:,3])+f1_score(labels1[:,4].cpu(),acc_pred3[:,4]))/5)

    import matplotlib.pyplot as plt
    from sklearn.metrics import multilabel_confusion_matrix
    conf_matrices = multilabel_confusion_matrix(labels[:,:5], acc_pred2.cpu().numpy())  # 可将'1'等替换成自己的类别，如'cat'。

    # 计算比例
    conf_matrices_normalized = []
    for cm in conf_matrices:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        conf_matrices_normalized.append(cm_normalized)

    # 设置图的大小
    fig, axes = plt.subplots(1, len(conf_matrices_normalized), figsize=(10 * len(conf_matrices_normalized), 8))

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.02)

    # 确保axes是数组
    if len(conf_matrices_normalized) == 1:
        axes = [axes]

    # 绘制混淆矩阵
    for i, cm in enumerate(conf_matrices_normalized):
        im = axes[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[i].set_xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 30})
        axes[i].set_ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 30})
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].tick_params(axis='both', which='major', labelsize=30)
        for (j, k), val in np.ndenumerate(cm):
            axes[i].text(k, j, f"{val:.2f}", ha="center", va="center", fontsize=50)

    # 创建一个专门用于颜色条的Axes
    cbar_ax = fig.add_axes([0.9, 0.08, 0.005, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=30)

    plt.show()




    pred0 = torch.empty_like(pred).copy_(pred)
    pred0 = torch.max(pred0,dim=1)[0]
    pred0=torch.ones_like(pred0)-pred0
    #
    fpr0, tpr0, thresholds1 = roc_curve(labels[:,0], pred0.cpu())
    fpr1, tpr1, thresholds1 = roc_curve(labels[:,1], pred[:,0].cpu())
    fpr2, tpr2, thresholds2 = roc_curve(labels[:,2], pred[:,1].cpu())
    fpr3, tpr3, thresholds3 = roc_curve(labels[:,3], pred[:,2].cpu())
    fpr4, tpr4, thresholds4 = roc_curve(labels[:,4], pred[:,3].cpu())
    roc_auc0 = auc(fpr0, tpr0)
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    roc_auc3 = auc(fpr3, tpr3)
    roc_auc4 = auc(fpr4, tpr4)
    # micro（方法二）
    labels_roc=[]
    pre_roc=[]
    labels_roc.extend(labels[:,0])
    labels_roc.extend(labels[:,1])
    labels_roc.extend(labels[:,2])
    labels_roc.extend(labels[:,3])
    labels_roc.extend(labels[:,4])
    pre_roc.extend(pred0.cpu())
    pre_roc.extend(pred[:,0].cpu())
    pre_roc.extend(pred[:,1].cpu())
    pre_roc.extend(pred[:,2].cpu())
    pre_roc.extend(pred[:,3].cpu())
    fpr_i, tpr_i, _ = roc_curve(labels_roc, pre_roc)
    roc_auc_i = auc(fpr_i, tpr_i)
    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate((fpr0,fpr1,fpr2,fpr3)))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    mean_tpr += np.interp(all_fpr, fpr0, tpr0)
    mean_tpr += np.interp(all_fpr, fpr1, tpr1)
    mean_tpr += np.interp(all_fpr, fpr2, tpr2)
    mean_tpr += np.interp(all_fpr, fpr3, tpr3)
    mean_tpr /= 4
    fpr_a = all_fpr
    tpr_a = mean_tpr
    roc_auc_a = auc(fpr_a, tpr_a)
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    # plt.rcParams['figure.figsize'] = (30.0, 30.0)
    plt.rcParams['image.interpolation'] = 'bilinear'
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置标题大小
    plt.rcParams['font.size'] = '0'
    plt.plot(fpr0, tpr0,  color='r', linestyle='-', linewidth=2, markerfacecolor='none',
             label=u'ROC curve of BI-RADS 1 (AUC = %0.2f)' % roc_auc0)
    plt.plot(fpr1, tpr1,  color='orange', linestyle='-', linewidth=2, label=u'ROC curve of BI-RADS 2 (AUC = %0.2f)' % roc_auc1)
    plt.plot(fpr2, tpr2,  color='y', linestyle='-', linewidth=2, markerfacecolor='none',
             label=u'ROC curve of BI-RADS 3 (AUC = %0.2f)' % roc_auc2)
    plt.plot(fpr3, tpr3,  color='g', linestyle='-', linewidth=2, markerfacecolor='none',
             label=u'ROC curve of BI-RADS 4 (AUC = %0.2f)' % roc_auc3)
    plt.plot(fpr4, tpr4,  color='b', linestyle='-', linewidth=2, markerfacecolor='none',
             label=u'ROC curve of BI-RADS 5 (AUC = %0.2f)' % roc_auc4)


    plt.plot(fpr_i, tpr_i,  color='purple', linestyle='dashed', linewidth=2, markerfacecolor='none',
             label=u'micro-average ROC curve (AUC = %0.2f)' % roc_auc_i)
    plt.plot(fpr_a, tpr_a,  color='c', linestyle='dashed', linewidth=2, markerfacecolor='none',
             label=u'macro-average ROC curve (AUC = %0.2f)' % roc_auc_a)

    print('bi-rads auc %0.3f'% roc_auc_i)
    print(compute_auc_ci(torch.tensor(labels_roc), torch.tensor(pre_roc)))

    plt.legend(loc='lower right',prop = {'size':14})
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate' ,fontdict={'family' : 'Times New Roman', 'size'   : 14})
    plt.xlabel('False Positive Rate', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.xticks(fontproperties='Times New Roman', size=18)
    plt.grid(linestyle='-.')
    plt.grid(True)
    plt.savefig('plt/9.jpg'%(),dpi=300)

    plt.close()

if __name__ == '__main__':
    import numpy as np
    import random
    seed=3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)  # 保证随机结果可复现
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=1)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")


    # 是否冻结权重
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    # for i in range(15):
    #     for j in range(15):
    #         print(i,j)
    #         main(opt,1+i*0.01,1+j*0.01)
    device = torch.device('cuda:0' )
    a2=torch.zeros([1,9])
    b2=torch.zeros([1,4]).to(device)
    c2=torch.zeros([1,7]).to(device)
    a,b,c=main(opt,jiaocha=0,weights=r'C:\Users\Administrator\Desktop\效果记录2\DM10000\final\model\model_inbreast.pth')
    a2=torch.cat([a2,a],dim=0)
    b2=torch.cat([b2,b],dim=0)
    c2=torch.cat([c2,c],dim=0)
    pinggu(a2[1:,:],b2[1:,:],c2[1:,:])
