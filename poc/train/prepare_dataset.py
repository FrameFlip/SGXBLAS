import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torchvision

def download(args):

    dataset_folder = os.path.join(args.root, args.dataset)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    download_tmp = os.path.join(args.root, 'download')
    if not os.path.exists(download_tmp):
        os.makedirs(download_tmp)

    trainset_folder = os.path.join(dataset_folder, "train")
    if not os.path.exists(trainset_folder):
        os.makedirs(trainset_folder)
    testset_folder = os.path.join(dataset_folder, "test")
    if not os.path.exists(testset_folder):
        os.makedirs(testset_folder)

    if args.dataset == "cifar10":
        for class_id in range(10):
            cls_folder_train = os.path.join(trainset_folder, str(class_id))
            if not os.path.exists(cls_folder_train):
                os.makedirs(cls_folder_train)
            cls_folder_test = os.path.join(testset_folder, str(class_id))
            if not os.path.exists(cls_folder_test):
                os.makedirs(cls_folder_test)
        trainset = torchvision.datasets.CIFAR10(root=download_tmp, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root=download_tmp, train=False, download=True)
        for idx in range(trainset.data.shape[0]):
            cls_folder = os.path.join(trainset_folder, str(trainset.targets[idx]))
            img_name = str(trainset.targets[idx]) + "_" + str(idx).zfill(6) + ".png"
            img_path = os.path.join(cls_folder, img_name)
            print(img_path)
            cv2.imwrite(img_path, trainset.data[idx])

        for idx in range(testset.data.shape[0]):
            cls_folder = os.path.join(testset_folder, str(testset.targets[idx]))
            img_name = str(testset.targets[idx]) + "_" + str(idx).zfill(6) + ".png"
            img_path = os.path.join(cls_folder, img_name)
            print(img_path)
            cv2.imwrite(img_path, testset.data[idx])
    elif args.dataset == "gtsrb":
        for class_id in range(43):
            cls_folder_train = os.path.join(trainset_folder, str(class_id).zfill(3))
            if not os.path.exists(cls_folder_train):
                os.makedirs(cls_folder_train)
            cls_folder_test = os.path.join(testset_folder, str(class_id).zfill(3))
            if not os.path.exists(cls_folder_test):
                os.makedirs(cls_folder_test)
        trainset = torchvision.datasets.GTSRB(root=download_tmp, split="train", download=True)
        testset = torchvision.datasets.GTSRB(root=download_tmp, split="test", download=True)
        for idx in range(trainset.__len__()):
            data, label = trainset.__getitem__(idx)
            cls_folder = os.path.join(trainset_folder, str(label).zfill(3))
            img_name = str(label).zfill(3) + "_" + str(idx).zfill(6) + ".png"
            img_path = os.path.join(cls_folder, img_name)
            print(img_path)
            cv2.imwrite(img_path, np.array(data))

        for idx in range(testset.__len__()):
            data, label = testset.__getitem__(idx)
            cls_folder = os.path.join(testset_folder, str(label).zfill(3))
            img_name = str(label).zfill(3) + "_" + str(idx).zfill(6) + ".png"
            img_path = os.path.join(cls_folder, img_name)
            print(img_path)
            cv2.imwrite(img_path, np.array(data))
    else:
        pass

    # plt.imshow(testset.data[idx])
    # plt.show()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--root', type=str, default='../data')
    
    args = parser.parse_args()
    print(args)
    download(args);


if __name__ == '__main__':
    main()

