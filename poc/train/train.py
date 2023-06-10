from __future__ import division, print_function

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
from torchvision.models import resnet34, resnet50, vgg16
from torchvision.models import ResNet34_Weights, ResNet50_Weights, VGG16_Weights
from eval_imagenet import evaluate
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Average(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count

    def __str__(self):
        return '{:.6f}'.format(self.average)


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


class Trainer(object):
    def __init__(self, net, optimizer, train_loader, test_loader, device, scheduler):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.scheduler = scheduler

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            self.scheduler.step()
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()
            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc))


    def train(self):
        train_loss = Average()
        train_acc = Accuracy()
        print("_______")
        self.net.train()
        a=0
        for data, label in self.train_loader:
            data = data.to(self.device)
            label = label.to(self.device)
            a+=1
            output = self.net(data)
            loss = self.loss_func(output, label)
            #print(a, loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, label)
            #print(a, train_acc)
        return train_loss, train_acc

    def evaluate(self):
        test_loss = Average()
        test_acc = Accuracy()
        
        self.net.eval()

        with torch.no_grad():
            for data, label in self.test_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.net(data)
                loss = self.loss_func(output, label)

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, label)

        return test_loss, test_acc


def get_dataloader(root, train_transform, test_transform, batch_size, test_batch):

    trainset_folder = os.path.join(root, "train")
    trainset = ImageFolder(root=trainset_folder,transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    
    testset_folder = os.path.join(root, "test")
    testset = ImageFolder(root=testset_folder, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch, shuffle=False)

    return train_loader, test_loader

def run(args):
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
            train_transform = cifar10_train_transform
            test_transform = cifar10_test_transform
        elif args.dataset == "gtsrb":
            kwargs = {"num_classes": 43}
            train_transform = gt_train_transform
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
            train_transform = cifar10_train_transform
            test_transform = cifar10_test_transform
        elif args.dataset == "gtsrb":
            kwargs = {"num_classes": 43}
            train_transform = gt_train_transform
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
            train_transform = cifar10_train_transform
            test_transform = cifar10_test_transform
        elif args.dataset == "gtsrb":
            kwargs = {"num_classes": 43}
            train_transform = gt_train_transform
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
    
    dataset_folder = os.path.join(args.root, args.dataset)

    if args.dataset == "imagenet":
        evaluate(net, dataset_folder, imagenet_test_transform, device, args.test_num, args.test_batch, args.print_freq )
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9,weight_decay=5e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        train_loader, test_loader = get_dataloader(dataset_folder, train_transform, test_transform, args.batch_size, args.test_batch)
        trainer = Trainer(net, optimizer, train_loader, test_loader, device, scheduler)
        trainer.fit(args.epochs)

    # save trained models
    save_foler = args.dataset + "_" + args.network
    save_dir = os.path.join(args.save_root, save_foler)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    trained_model_path = os.path.join(save_dir, args.best_name)
    torch.save(net, trained_model_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01)
    parser.add_argument('--root', type=str, default='../data')
    parser.add_argument('--save-root', type=str, default='../trained_models')
    parser.add_argument('--best-name', type=str, default='best.task.3.ckpt')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-num', type=int, default=4000)
    parser.add_argument('--test-batch', type=int, default=100)
    parser.add_argument('--print-freq', type=int, default=10)


    args = parser.parse_args()
    print(args)
    print(os.getcwd())
    run(args)

if __name__ == '__main__':
    main()
