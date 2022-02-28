# -*- coding:utf-8 -*-

"""
@ Author: WeiMiQu
@ Contact: 1830465230@qq.com
@ Date: 2022/2/19 下午2:09 
@ Software: PyCharm
@ File: train.py
@ Desc: 
"""


import paddle
import numpy as np

from os import path
from paddle.vision import transforms

from models.convnext import ConvNeXt
from dataset.dataset import MyImageNetDataset


def train(model=None,
          train_dataset=None,
          valid_dataset=None,
          test_dataset=None,
          optimize=None,
          epochs=100,
          batch_size=32,
          loss_function=None,
          save_path='weight'):

    if train_dataset is not None:

        train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=0,
                                            drop_last=False, use_shared_memory=False)

        extent4train = len(train_dataset)

    else:

        raise Exception("input error, len(train_dataset) must > 0")

    if valid_dataset is not None:
        valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=0,
                                            drop_last=False, use_shared_memory=False)
        extent4valid = len(valid_dataset)

    if test_dataset is not None:
        test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0,
                                           drop_last=True, use_shared_memory=False)

    accuracy4best = 0

    for epoch in range(epochs):

        loss4train = 0
        model.train()
        for i, data in enumerate(train_loader()):

            images, labels = data
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss4train += loss.numpy()[0]
            # acc = paddle.metric.accuracy(input=outputs, label=labels.unsqueeze(1)).numpy()[0]
            # print("train", "\tepoch:", epoch, "\titer:", i, "\taccuracy:", acc)
            loss.backward()  # 反向传播计算梯度
            optimize.step()  # 更新参数
            optimize.clear_grad()

        loss4valid = 0
        accuracy = []
        model.eval()
        for i, data in enumerate(valid_loader()):

            images, labels = data
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss4valid += loss.numpy()[0]

            acc = paddle.metric.accuracy(input=outputs, label=labels.unsqueeze(1)).numpy()[0]
            # print("valid", "\tepoch:", epoch, "\titer:", i, "\taccuracy:", acc)
            accuracy.append(acc)

        accuracy = np.array(accuracy).mean()
        if accuracy > accuracy4best:
            state_dict = model.state_dict()
            paddle.save(state_dict, path.join(save_path, "best.pdparams"))
            accuracy4best = accuracy

        print('epoch:', epoch + 1,
              '\tlr:', optimize.get_lr(),
              '\ttrain_loss_mean:', loss4train / extent4train,
              '\tval_loss_mean:', loss4valid / extent4valid,
              '\tbest accuracy:', accuracy4best,
              '\tlast accuracy:', accuracy)

    state_dict = model.state_dict()
    paddle.save(state_dict, path.join(save_path, "last.pdparams"))

    if test_dataset is not None:

        model.eval()
        for i, data in enumerate(test_loader()):

            images, labels = data
            outputs = model(images)
            loss = loss_function(outputs, labels)


if __name__ == '__main__':

    train_dir = r"Rock_Classification_Dataset/train"
    valid_dir = r"Rock_Classification_Dataset/valid"
    lr_schedule = 0.001

    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.52563523, 0.55325405, 0.59199389],
                                                         std=[0.21688389, 0.20846758, 0.20908272])])

    train_dataset = MyImageNetDataset(root=train_dir, transform=transform)
    valid_dataset = MyImageNetDataset(root=valid_dir, transform=transform)

    loss_function = paddle.nn.CrossEntropyLoss(reduction='mean')

    model = ConvNeXt(num_classes=7)
    optimize = paddle.optimizer.Adam(learning_rate=lr_schedule, parameters=model.parameters())
    train(model=model,
          train_dataset=train_dataset,
          valid_dataset=valid_dataset,
          optimize=optimize,
          loss_function=loss_function)
