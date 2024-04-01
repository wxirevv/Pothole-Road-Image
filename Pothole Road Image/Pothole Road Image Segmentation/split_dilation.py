#!/user/bin/python
# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler, optimizer
import torchvision
import os, sys
import cv2 as cv
from torch.utils.data import DataLoader, sampler
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F

class SegmentationDataset(object):
    def __init__(self, image_dir, mask_dir):
        self.images = []
        self.masks = []
        files = os.listdir(image_dir)
        sfiles = os.listdir(mask_dir)
        for i in range(len(sfiles)):
            img_file = os.path.join(image_dir, files[i])
            mask_file = os.path.join(mask_dir, sfiles[i])
            # print(img_file, mask_file)
            self.images.append(img_file)
            self.masks.append(mask_file)

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        else:
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  # BGR order
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        # 调整图像和目标标签的尺寸
        img = cv.resize(img, (224, 224))  # 替换desired_width和desired_height为你希望的尺寸
        mask = cv.resize(mask, (224, 224))

        # 输入图像
        img = np.float32(img) / 255.0
        img = np.expand_dims(img, 0)

        # 目标标签0 ~ 1， 对于
        mask[mask <= 128] = 0
        mask[mask > 128] = 1
        mask = np.expand_dims(mask, 0)
        sample = {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask),}
        return sample
class UNetModel(torch.nn.Module):

    def __init__(self, in_features=1, out_features=2, init_features=32):
        super(UNetModel, self).__init__()
        features = init_features
        dilation = 2  # 设置空洞卷积的空洞大小

        self.encode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_features, out_channels=features, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU()
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=features*2, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features*2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU()
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*4, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*4, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU()
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*8, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_decode_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*16, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features * 16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*16, out_channels=features*16, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features * 16),
            torch.nn.ReLU()
        )
        self.upconv4 = torch.nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decode_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*16, out_channels=features*8, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features*8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decode_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*4, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*4, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU()
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*2, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU()
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=dilation, stride=1, dilation=dilation),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU()
        )
        self.out_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=out_features, kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        enc1 = self.encode_layer1(x)
        enc2 = self.encode_layer2(self.pool1(enc1))
        enc3 = self.encode_layer3(self.pool2(enc2))
        enc4 = self.encode_layer4(self.pool3(enc3))

        bottleneck = self.encode_decode_layer(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decode_layer4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decode_layer3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decode_layer2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decode_layer1(dec1)

        out = self.out_layer(dec1)

        return out

# 添加评估函数来计算准确率
def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy
def postprocess(output):
    # 对模型输出结果进行后处理
    processed_output = output.argmax(dim=1)  # 假设直接取预测结果的最大值作为最终预测
    return processed_output


if __name__ == '__main__':
    index = 0
    num_epochs = 100
    train_on_gpu = True
    unet = UNetModel().cuda()
    # model_dict = unet.load_state_dict(torch.load('unet_road_model-100.pt'))

    image_dir = "D:\\桌面\\dataset_folder\\picture"
    mask_dir = "D:\\桌面\\dataset_folder\\result"
    dataset = SegmentationDataset(image_dir, mask_dir)



    # 划分训练集和验证集
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size))  # 使用80%的数据作为训练集
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # 创建采样器
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # 创建训练集和验证集的数据加载器
    train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=8, sampler=valid_sampler)
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        train_loss = 0.0
        total_accuracy = 0.0

        # 训练模型
        unet.train()
        for i_batch, sample_batched in enumerate(train_loader):
            images_batch, target_labels = \
                sample_batched['image'], sample_batched['mask']

            if train_on_gpu:
                images_batch, target_labels = images_batch.cuda(), target_labels.cuda()

            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            m_label_out_ = unet(images_batch)

            # calculate the batch loss
            target_labels = target_labels.contiguous().view(-1)
            m_label_out_ = m_label_out_.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
            target_labels = target_labels.long()
            loss = torch.nn.functional.cross_entropy(m_label_out_, target_labels)
            # backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_loss += loss.item()

            # compute accuracy for this batch
            batch_accuracy = compute_accuracy(m_label_out_, target_labels)
            total_accuracy += batch_accuracy


            # 计算平均损失和准确率
        train_loss = train_loss / len(train_loader.dataset)
        average_accuracy = total_accuracy / len(train_loader)

        # 显示训练集的损失函数和准确率
        print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}%'.format(epoch, train_loss,
                                                                                      average_accuracy * 100))

        # 在验证集上评估模型
        unet.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_accuracy = 0.0
            for i_batch, sample_batched in enumerate(valid_loader):
                images_batch, target_labels = sample_batched['image'], sample_batched['mask']
                if train_on_gpu:
                    images_batch, target_labels = images_batch.cuda(), target_labels.cuda()

                m_label_out_ = unet(images_batch)
                target_labels = target_labels.contiguous().view(-1)
                m_label_out_ = m_label_out_.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
                target_labels = target_labels.long()
                loss = torch.nn.functional.cross_entropy(m_label_out_, target_labels)
                valid_loss += loss.item()
                batch_accuracy = compute_accuracy(m_label_out_, target_labels)
                valid_accuracy += batch_accuracy
                processed_masks = postprocess(m_label_out_)

            valid_loss /= len(valid_loader.dataset)
            valid_accuracy /= len(valid_loader)
            print('Epoch: {} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.2f}%'.format(epoch, valid_loss,
                                                                                              valid_accuracy * 100))

        # save model
    unet.eval()
    torch.save(unet.state_dict(), 'unet_road_split_dilation3.pt')

