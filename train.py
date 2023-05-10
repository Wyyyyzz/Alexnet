import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import os



#使用GPU
device = "cuda" if torch.cuda.is_available() else 'cpu'
model = AlexNet().to(device)
print(device)

#导入训练集测试集
ROOT_TRAIN = "D:/pycharm/PyCharm Community Edition 2020.3.2/Alexnet/224cifar_data/train"
ROOT_TEST = "D:/pycharm/PyCharm Community Edition 2020.3.2/Alexnet/224cifar_data/val"

#将图像归一化到（-1，1）之间
normalize=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])#数据归一化的好处是方便特征提取，


#训练集数据转换
train_transform=transforms.Compose([
    transforms.Resize((224,224)),#将数据裁剪到224*224像素，论文要求
    transforms.RandomVerticalFlip(),#随机垂直全展，数据强化的一种
    transforms.ToTensor(),#数据转化为张量
    normalize
])
#测试集
val_transform=transforms.Compose([
    transforms.Resize((224,224)),#将数据裁剪到224*224像素，论文要求   ##为确保验证真实性不需要全展
    transforms.ToTensor(),#数据转化为张量
    normalize
])

train_dataset= ImageFolder(ROOT_TRAIN,transform=train_transform)
val_dataset= ImageFolder(ROOT_TEST,transform=val_transform)

train_datalodar = DataLoader(train_dataset,batch_size=32,shuffle=True)#打乱
val_datalodar = DataLoader(val_dataset,batch_size=32,shuffle=True)

#损失函数:这里用交叉熵
loss_function = nn.CrossEntropyLoss()

#优化器 这里用随机梯度
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#设置学习率,每隔10轮变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

#定义训练函数
def train(dataloader,model,loss_function,optimizer):#传入数据，模型，损失和优化器

    loss,current,n=0.0,0.0,0#定义精确度和loss，指示器

    for batch,(x,y) in enumerate(dataloader):#定义一个循环将数据取出
        image,y=x.to(device),y.to(device) #将数据导入显卡
        output = model(image)
        cur_loss =loss_function(output,y)#将真实值和预测值进行误差分析
        _,pred =torch.max(output,axis=1)#取出准确率最高的值
        cur_acc=torch.sum(y==pred)/output.shape[0]#累加精确率

        #反向传播
        optimizer.zero_grad()#将梯度降为0
        cur_loss.backward()
        optimizer.step()#更新梯度
        loss+=cur_loss.item()
        current+=cur_acc.item()#累加精确度
        n+=1

    train_loss = loss/n
    train_acc = current/n
    print("train_loss"+str(train_loss))
    print("train_acc"+str(train_acc))
    return train_loss ,train_acc

#定义验证函数
def val(dataloader, model, loss_function):  # 传入数据，模型，损失,,不需要反向传播
    # 将模型转换为验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0  # 定义精确度和loss
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):  # 定义一个循环将数据取出
            image, y = x.to(device), y.to(device)  # 将数据导入显卡
            output = model(image)
            cur_loss = loss_function(output, y)  # 将真实值和预测值进行误差分析
            _, pred = torch.max(output, axis=1)  # 取出准确率最高的值
            cur_acc = torch.sum(y == pred / output.shape[0])  # 累加精确率
            n+=1


    val_loss = loss / n
    val_acc = current / n
    print("val_loss" + str(val_loss))
    print("val_acc" + str(val_acc))
    return val_loss, val_acc

#训练开始
loss_train=[]
acc_train=[]
loss_val=[]
acc_val=[]

epoch=10#训练轮次
min_acc =0
for t in range(epoch):
    lr_scheduler.step()
    print(f"[epoch %d]{t+1}\n---------")
    train_loss,train_acc=train(train_datalodar,model,loss_function,optimizer)#把训练loss，精确度导出
    val_loss,val_acc=val(val_datalodar,model,loss_function)#验证同理

    loss_train.append(train_loss)#把值加到集合
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    #保存最好模型权重
    if val_acc> min_acc:
        folder="save_model"
        if not os.path.exists(folder)  :
            os.mkdir("save_model")
        min_acc= val_acc  #如果最小权重小于测试，保存
        print(f'save best model , in {t+1}epoch')
        torch.save(model.state_dict(),"save_modle/best_model.pth")

    #保存最后一轮权重文件
    if t==epoch -1:
        torch.save(model.state_dict(), "save_modle/last_model.pth")


print('finished training')


##画图判断loss值
def matplot_loss(train_loss,val_loss):
    plt.plot(train_loss,label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.lgend(loc="best")
    plt.ylabel('loss value')
    plt.xlabel("epoch")
    plt.title("loss值对比图")
    plt.show()


##画图判断acc值
def matplot_acc(train_acc,val_acc):
    plt.plot(train_acc,label="train acc")
    plt.plot(val_acc, label="val acc")
    plt.lgend(loc="best")
    plt.ylabel('acc value')
    plt.xlabel("epoch")
    plt.title("acc值对比图")
    plt.show()

print(list(model.parameters()))