# Requirements
* OS: ``` Ubuntu 20.04```

* Pytorch: ```1.12.1+cu113```

* Torchvision: ```0.13.1+cu113 ```

# 1. Train DNN models and export model parameters

## 1.1 Train a DNN model with python
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



# 2. Run ML inference

## Requirement

* gcc : ```gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0```

* make: ```GNU Make 4.2.1```

* OpenBLAS: ```0.3.20```


### 2.1 Install OpenBLAS

1. step into the ```poc``` folder 

```
cd poc
```

2. run ```make``` to compile and install openblas-0.3.20
```
make install
```

### 2.2 Run MLInfras

In this step, the exported model parameters will be used to inference by our ML inference Infrastructure which is implemented in folder ```poc/infras```. For example, if you want to run ```gtsrb``` task on ```vgg16``` network, you can perform:

```
cd poc/infras
make gtsrb_vgg16
```
Then, the exported model parameters will loaded in and evaluate on the same testset. 

This infrastructure provides a tesebed to evaluate our fualt injections on ```OpenBLAS``` library. 

All of our attacks, including ```LLVM-VIS``` algorithms and ```rowhammer``` exploitations are evaluated on this infrastructure. 



# 3. Attack 

## Requirement

* cmake: ```3.16.3```

* clang: ```ubuntu clang version 12.0.0-3ubuntu1~20.04.5```


### 3.1 Install LLVM

```
cd ~
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 12
```

#### update soft link 
you need to change your current soft link of ```clang, clang++, opt``` to your installed above:

```
clang --version
which clang
ls -l /usr/bin | grep clang
ln -s /usr/lib/llvm-12/bin/clang /usr/bin/clang
```

repeat the same process to update the soft link of ```clang++, opt```.

### 3.2 complie llvm-pass 

The core function of our LLVM-VIS algorithm is implemented as a llvm-pass (```attack\source\ground_truth\llvm-pass\flipBranches``` and ```attack\source\ground_truth\llvm-pass\flipLauncher```). So, at first, we need to complie this llvm pass as a ```.so``` library. 

Before that, please check whether your ```cmake``` version is ```3.16.3``.

There are two absolute path in file ```flipLauncher\flipLauncher.cpp``` should be replaced as your own absolute path. 
* In line of 20, ``` #define BRANCH_INFO_FILE "/home/xxx/SGXBLAS/attack/source/ground_truth/br_info.tmp" ```

* In line of 79, ``` std::string command_prefix = "opt -enable-new-pm=0 -load ${HOME}/xxx/SGXBLAS/attack/source/ground_truth/llvm-pass/build/flipBranches/libflipBranches.so -flipbranches -o ";```

After check them, we start compile our llvm-pass:

```
cd attack\source\ground_truth\llvm-pass
mkdir build && cd build 
cmake ..
make 
```

### 3.3 Run attack instance

Put all together, we implement our attacks in ```attack\source\ground_truth\Makefile```, the only you need to do is specific the attack instance, e.g., ```gtsrb_vgg16``` in the line of ```attack\source\ground_truth\Makefile``` file. 

Before that, you need check the path of our compiled llvm, in particular, you need check the following files:

```
ground_truth/openblas_makefiles/attack/dirvier/level3/Makefile

ground_truth/openblas_makefiles/attack/interface/Makefile

```
please replace the path ```${HOME}/SGXBLAS-main``` in above files as your own path. 

Before attacking, you need to specific your attack instance in ```attack\source\ground_truth\Makefile``` in line 34, 35, 36: 

```
EXP_CONF = gtsrb_vgg16
EXP_DATASET = gtsrb
EXP_NETWORK = vgg16
```

Finally, we have it!

```
cd attack\source\ground_truth
make ground_truth
```