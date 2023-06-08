# Requirements
* OS: ``` Ubuntu 20.04```

* Pytorch: ```1.12.1+cu113```

* Torchvision: ```0.13.1+cu113 ```

* gcc : ```gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0```

* OpenBLAS: ```0.3.20```

* clang: ```12.0.0-3ubuntu1~20.04.5```

## 1. Install OpenBLAS
1. step into the ```poc``` folder 

```
cd poc
```

2. run ```make install``` to compile and install openblas-0.3.20
```
make install
```

# 2. Run ML inference
## 2.1 Train a DNN model with python
We provide the training code to obtain a well trained DNN model weight parameters. The supported network architectures and tasks are listed as following:

**Networks:** 
* ResNet-34
* ResNet-50
* VGG-16

**Datasets:**
* CIFAR-10
* GTSRB

Step 1: go into ```infras``` folder.

```
cd infras
```

Step 2: prepare training data. To build an unified training and test data read and write format. We use ```ImageFolder``` API provided by ```torchvision.datasets``` to save all of our datasets as ```.png``` images. With the followed structure:
```
--cifar10
    --train
        --0
        --1
        --...
        --9
    --test
        --0
        --1
        --...
        --9
```

The functionality is implemented in ```prepare_datasets.py```, and just run 

```
python prepare_datasets.py --dataset cifar10
``` 


Step 2: run ```train.py``` to obtain a well trained DNN model weight parameters. 

```
python train.py --network resnet34 --dataset cifar10 --
```
