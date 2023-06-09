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
* ImageNet [Only used for inference]

Step 1: go into ```train_on_torch``` folder.

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
python prepare_dataset.py --dataset cifar10
``` 
After this step, you should have a re-arranged folder to ```ImageFolder``` to read image data, and convert them as ```dataloader```.


Step 2: run ```train.py``` to obtain a well trained DNN model weight parameters. 

```
python train.py --network vgg16 --dataset gtsrb
```

The trained DNN model parameters are stored in ```../trained_models``` folder. The trained pytorch weight parameters are named as ```best.task.3.ckpt``` and storaged according to its network and dataset in ```../trained_models```.

Step 3: we need export the trained model parameters as ```.txt``` format. This is impelemented in file ```export_model.py```.

```
python export_model.py --network vgg16 --dataset gtsrb
```

After that, you will export the model parameters as ```float``` and storaged in ```../trained_models/gtsrb_vgg16/best.task.txt```.


