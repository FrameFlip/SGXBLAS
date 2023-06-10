'''
Author: shawn233
Date: 2021-04-02 15:13:26
LastEditors: shaofeng
LastEditTime: 2022-12-07 16:04:08
Description: Export PyTorch model
'''

import os
import sys
import torch
from torchvision.models import resnet34, resnet50, vgg16
from torchvision.models import ResNet34_Weights, ResNet50_Weights, VGG16_Weights
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
import argparse


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output, label):
        predictions = output.data.argmax(dim=1)
        correct = predictions.eq(label.data).sum().item()

        self.correct += correct
        self.count += output.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

def export_network(args):
    model_folder = args.dataset + "_" + args.network
    model_abs_path = os.path.join(args.save_root, model_folder)
    model_path = os.path.join(model_abs_path, args.best_name)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model = model.state_dict()
    np.set_printoptions(precision=8, threshold=sys.maxsize)
    output_path = os.path.join(model_abs_path, args.output_name)
    with open(output_path, "w") as out_f:
        for name in model:
            value = model[name].cpu().numpy()
            out_f.write(f"{name} {value.shape}\n")
            out_f.write(str(value) + "\n")


def infer(args):
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    cifar10_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar10_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    gt_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    gt_test_transform = transforms.Compose([
        transforms.Resize(size=(32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    imagenet_test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.network == "resnet50":
        if args.dataset == "cifar10":
            kwargs = {"num_classes": 10}
            test_transform = cifar10_test_transform
        elif args.dataset == "gtsrb":
            kwargs = {"num_classes": 43}
            test_transform = gt_test_transform
        elif args.dataset == "imagenet":
            kwargs = {"weights": ResNet50_Weights.IMAGENET1K_V2}
            test_transform = imagenet_test_transform
        else:
            print("Unknown dataset")
            return 
        net = resnet50(kwargs).to(device)
    elif args.network == "resnet34":
        if args.dataset == "cifar10":
            kwargs = {"num_classes": 10}
            test_transform = cifar10_test_transform
        elif args.dataset == "gtsrb":
            kwargs = {"num_classes": 43}
            test_transform = gt_test_transform
        elif args.dataset == "imagenet":
            kwargs = {"weights": ResNet34_Weights.IMAGENET1K_V1}
            test_transform = imagenet_test_transform
        else:
            print("Unknown dataset")
            return 
        net = resnet34(kwargs).to(device)
    elif args.network == "vgg16":
        if args.dataset == "cifar10":
            kwargs = {"num_classes": 10}
            test_transform = cifar10_test_transform
        elif args.dataset == "gtsrb":
            kwargs = {"num_classes": 43}
            test_transform = gt_test_transform
        elif args.dataset == "imagenet":
            kwargs = {"weights": VGG16_Weights.IMAGENET1K_V1}
            test_transform = imagenet_test_transform
        else:
            print("Unknown dataset")
            return 
        net = vgg16(weights = None, progress = True, **kwargs).to(device)
    else:
        print("Unknown network")
    model_folder = args.dataset + "_" + args.network
    model_abs_path = os.path.join(args.save_root, model_folder)
    model_path = os.path.join(model_abs_path, args.best_name)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    dataset_folder = os.path.join(args.root, args.dataset)
    testset_folder = os.path.join(dataset_folder, "test")
    testset = ImageFolder(root=testset_folder, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False)
    # test_loss = Average()
    test_acc = Accuracy()
    
    model.eval()

    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            # loss = loss_func(output, label)

            # test_loss.update(loss.item(), data.size(0))
            test_acc.update(output, label)
    print("Test Acc.:", test_acc)



def main():
    print(os.getcwd())
    # export_resnet("./cifar100/best.checker.1.ckpt", "./cifar100/best.checker.txt")
    # export_resnet("./cifar100/best.task.3.ckpt", "./cifar100/best.task.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--test-batch', type=int, default=100)
    parser.add_argument('--root', type=str, default='../data')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save-root', type=str, default='../trained_models')
    parser.add_argument('--best-name', type=str, default='best.task.3.ckpt')
    parser.add_argument('--output-name', type=str, default='best.task.txt')
    
    args = parser.parse_args()
    print(args)

    infer(args)
    # export_network(args)


if __name__ == "__main__":
    main()
