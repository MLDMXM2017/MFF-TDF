import os
import sys
import shutil
import random
import math
import csv
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import roc_auc_score, roc_curve, auc

def preprocess(img, rgb_mean, rgb_std, pad_val, image_shape):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(4, fill=pad_val),
            torchvision.transforms.Resize(image_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
        return transforms(img).unsqueeze(0)

# get iteration
def data_iter(rand_shuf, img_dir,
              img_name, features, labels,
              images_per_gpu, rgb_mean, rgb_std, pad_val,
              image_shape):
    num_examples = len(img_name)
    indices = list(range(num_examples))
    if rand_shuf:
        random.shuffle(indices)

    for i in range(0, num_examples, images_per_gpu):

        if i + images_per_gpu > num_examples:
            batch_indices = indices[i:] + indices[: images_per_gpu - num_examples + i]
        else:
            batch_indices = indices[i: i + images_per_gpu]  
    
        imgs_tensor = preprocess(Image.open(img_dir+img_name[batch_indices[0]]).convert('RGB'), rgb_mean, rgb_std, pad_val, image_shape)
        for j in batch_indices[1:]:
            img_tongue = preprocess(Image.open(img_dir+img_name[j]).convert('RGB'), rgb_mean, rgb_std, pad_val, image_shape)
            imgs_tensor = torch.cat((imgs_tensor, img_tongue), dim=0)

        yield imgs_tensor, features[batch_indices], labels[batch_indices]

def train(net, train_img_name, train_features, train_labels, 
          valid_img_name, valid_features, valid_labels,
          num_epochs, learning_rate, weight_decay, momentum, batch_size, fold_i):
    save_file_name = work_dir + f'fold_{fold_i}_'

    best_epoch, save_i = -1, 0
    best_metric = [0, 0, 0, 0, 0]
    train_ls, valid_ls = [], []

    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(),
                                lr = learning_rate,
                                weight_decay = weight_decay,
                                momentum = momentum)
    # Create a learning rate scheduler object
    scheduler = lr_scheduler.LambdaLR(optimizer, 
                                      lr_lambda=lambda epoch: adjust_learning_rate(epoch, warmup_factor, warmup_epochs))     

    for epoch in range(num_epochs):        
        print('Epoch: ',epoch, '  Batch Size = ', batch_size, f"  lr = {optimizer.param_groups[0]['lr']:.2e}")
        train_ls_batch, valid_ls_batch = [], []
        net.train()

        optimizer.zero_grad()
        i = 0
        for img_tensor, feature_tensor, labels in data_iter(True, img_dir, train_img_name, train_features, train_labels,
                                                                images_per_gpu, rgb_mean, rgb_std, pad_val, image_shape): 
            X, Xf, y = img_tensor.to(device), feature_tensor.to(device), labels.to(device)
            l = colunm_respective_loss(net(X, Xf), y) # mean loss of images_per_gpu
            train_ls_batch.append(l.item())
            (l/mini_batch_num).backward() # add grad
            # print(l.item())
            i += 1
            if i >= mini_batch_num:
                optimizer.step()
                optimizer.zero_grad()
                i = 0
                print(l.item(), end="\r")              
        if i:
            optimizer.step()
        scheduler.step() # adjust learning rate
        train_ls.append(sum(train_ls_batch)/len(train_ls_batch))
        print('train_loss = ', train_ls[-1])

        if valid_labels is not None:
            net.eval()

            label_hub, yhat_hub = [], []
            samples_num = len(valid_img_name)
            val_result = torch.zeros((num_labels, 3), dtype=int)# [TP, FN, FP]
            val_matrix = torch.zeros((num_labels, num_labels), dtype=int)
            for img_tensor, feature_tensor, labels in data_iter(False, img_dir, valid_img_name, valid_features, valid_labels,
                                                                    images_per_gpu, rgb_mean, rgb_std, pad_val, image_shape): 
                X, Xf, y = img_tensor.to(device), feature_tensor.to(device), labels
                y_hat = net(X, Xf).detach().to('cpu')
                samples_num -= images_per_gpu
                if samples_num < 0:
                    last_num = images_per_gpu + samples_num
                    y_hat = y_hat[:last_num]
                    y = y[:last_num]

                label_hub = label_hub + y.argmax(dim=1).tolist()
                yhat_hub = yhat_hub + y_hat.tolist()
                pred = y_hat.squeeze(1).argmax(dim=1)
                for i in range(pred.shape[0]):
                    yi = y[i].argmax() # label index
                    pi = pred[i] # prediction index
                    val_matrix[pi, yi] += 1 
                    if y[i, pi]: # TP
                        val_result[pi][0] = val_result[pi][0] + 1
                    else:
                        val_result[yi][1] = val_result[yi][1] + 1
                        val_result[pi][2] = val_result[pi][2] + 1
                valid_ls_batch.append(colunm_respective_loss(y_hat, y).item())
            
            pred_array = np.asarray(yhat_hub)
            pred_exp = np.exp(pred_array - np.max(pred_array, axis=1, keepdims=True)) # 减np.max有利于数值稳定，且不会改变softmax的理论结果
            pred_softmax = pred_exp / pred_exp.sum(axis=1, keepdims=True)
            Auc = roc_auc_score(label_hub, pred_softmax[:, 1])
            Accuracy = (val_result[:, 0].sum() / len(yhat_hub)).item()
            Precision = (val_result[1,0]/(val_result[1,0]+val_result[1,2])).item()
            Recall = (val_result[1,0]/(val_result[1,0]+val_result[1,1])).item()
            F1 = 2 * (Precision * Recall) / (Precision + Recall)
            
            valid_ls.append(sum(valid_ls_batch)/len(valid_ls_batch))
            print('valid_loss = ', valid_ls[-1])
            print('Precision  = ', Precision)                
            print('Recall     = ', Recall)
            print('F1         = ', F1)
            print('Accuracy   = ', Accuracy)
            print('Auc        = ', Auc)    
            print('Confusion Matrix: ')
            print(val_matrix)

            if valid_ls.index(min(valid_ls))==len(valid_ls)-1: # if now epoch is best according to valid_loss
                best_metric = [Precision, Recall, F1, Accuracy, Auc]
                net.to('cpu')
                if best_epoch >= 0:
                    os.remove(save_file_name+f'best_epoch_{best_epoch:03}.params') # delete old
                best_epoch = epoch
                torch.save(net.state_dict(), save_file_name+f'best_epoch_{best_epoch:03}.params') # save params
                print(f'Saved best params of epoch_{best_epoch}')
                net.to(device)
        print('Best epoch is: ', best_epoch)

        # save params
        save_i += 1
        if save_i == save_interval:
            net.to('cpu')
            torch.save(net.state_dict(), save_file_name+f'epoch_{epoch:03}.params') # save params
            print('Save parameters of Epoch: ', epoch)
            save_i = 0
            net.to(device)
        print('-------------------------------------')
    
    output_ls = [[x, y] for x, y in zip(train_ls, valid_ls)]
    with open(save_file_name + 'train_and_valid_lose.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows([['train_lose', 'valid_lose']] + output_ls)
    return best_metric

def get_k_fold_data(k, i, X, Xf, y):
    assert k > 1
    fold_size = len(X) // k
    X_train, Xf_train, y_train = None, None, None
    indices = list(range(len(X)))
    random.shuffle(indices) 
    for j in range(k):
        idx = indices[slice(j * fold_size, (j + 1) * fold_size)] # the indices for this fold
        X_part = []
        for index in idx:
             X_part.append(X[index])
        Xf_part = Xf[idx]
        y_part = y[idx]
        if j == i:
            X_valid, Xf_valid, y_valid = X_part, Xf_part, y_part
        elif X_train is None:
            X_train, Xf_train, y_train = X_part, Xf_part, y_part
        else:
            X_train = X_train + X_part
            Xf_train = torch.cat([Xf_train, Xf_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    
    return X_train, Xf_train, y_train, X_valid, Xf_valid, y_valid

def k_fold(k, X_train, Xf_train, y_train, num_epochs, 
           learning_rate, weight_decay, momentum, batch_size):
    metrics_hub = []
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, Xf_train, y_train)
        net = model.get_net(num_features, num_labels, drop_rate) # !!!
        metrics = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, momentum, batch_size, i)
        metrics_hub.append(metrics)
        print(f'Results for Flod {i}')
        print([['Precision', 'Recall', 'F1', 'Accuracy', 'Auc'], metrics])

    metrics_array = np.asarray(metrics_hub)
    mean_metrics = metrics_array.mean(axis=0).tolist()
    return mean_metrics

def colunm_respective_loss(out_put, label):
    l = loss(out_put, label).mean()
    return l

# Define the learning rate adjustment function
def adjust_learning_rate(epoch, warmup_factor, warmup_epochs):
    max_lr = 1.0
    min_lr = warmup_factor * max_lr
    if epoch < warmup_epochs:
        # Preheat stage: linearly increase the learning rate
        return min_lr + (max_lr - min_lr) * epoch / warmup_epochs
    else:
        # Trionometric function learning rate decay strategy
        t = epoch - warmup_epochs
        cycle_length = num_epochs
        return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * t / cycle_length)) / 2

from models import TIPNet_9 as model # 1

if __name__ == "__main__":

    data_path = './data/Physiological_Indicators_and_Labels.csv'
    work_dir = './test_work_dir/'
    img_dir = './data/Tongue_Images/'
    train_data = pd.read_csv(data_path)
                    
    lr = 0.01
    weight_decay = 0.001
    momentum = 0.6
    batch_size = 64
    images_per_gpu = 1 # 

    drop_rate = 0.20 
    loss_exp = 0.90 #

    fold_num = 5
    num_epochs = 120
    device = 'cpu' # cpu # cuda
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
 
    warmup_epochs, warmup_factor = 5, 0.01
    save_interval = int(num_epochs/10) if int(num_epochs/10) else 1 # num_epochs/10
    balance_loss = True

    image_shape = (448, 448) # (448, 448) # (224, 224)
    rgb_mean = torch.tensor([123.675, 116.28, 103.53])/255 # COCO dataset
    rgb_std = torch.tensor([58.395, 57.12, 57.375])/255
    pad_val = [0, 0, 0]# images masked with dark

    # get images' name
    train_img_name = list(train_data.iloc[:,0])
    # get features
    all_features = train_data.iloc[:, 1:-1]
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=False, dtype=int)
    all_features = torch.tensor(all_features.values, dtype=torch.float32)

    # get labels
    all_labels = train_data.iloc[:, -1]
    all_labels = pd.get_dummies(all_labels, dummy_na=False, dtype=int)
    all_labels = torch.tensor(all_labels.values, dtype=torch.float32)
    if balance_loss: # 102272
        label_weight = torch.pow((all_labels.sum() / all_labels.sum(dim=0) / all_labels.shape[1]), loss_exp)
        all_labels = all_labels * label_weight

    num_features = all_features.shape[1]
    num_labels = all_labels.shape[1]

    mini_batch_num = batch_size/images_per_gpu
    if mini_batch_num!=round(mini_batch_num):
        mini_batch_num = int(mini_batch_num + 1)

    loss = nn.CrossEntropyLoss(reduction='none')

    mean_metrics = k_fold(fold_num, train_img_name, all_features, all_labels,
                                num_epochs, lr, weight_decay, momentum, batch_size)
    
    print('Five flods mean results:')
    print([['Precision', 'Recall', 'F1', 'Accuracy', 'Auc'], mean_metrics])


