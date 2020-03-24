# CNN-backbone
This is a summary repository of the classic backbone of the CNNs
## Quickly start

```
│  train.py     # The training process and all parameters
│
├─Data          # Downloaded dataset's location
│  │
│  └─cifar-10-batches-py
│          batches.meta
│          data_batch_1
│          data_batch_2
│          ...
├─lib
│  │  logger.py   # logger func
│  │  utils.py 
│  │  __init__.py
│  │
│  ├─nn
│     │  OctConv.py   # cnn modules
│     │  __init__.py
│
├─models              # CNN models
│  │  LeNet.py
│  │  OctNet.py
│  │  OctResnet.py
│  │  ResNet18.py
│  │  __init__.py
│
├─output               # trained model will be saved here
│  └─test              # a sample
│          net_best.pth    # Best model for validation
│          net_latest.pth  # Newly trained models
│          train_log.txt   # training log
```
### 1. Download CIFAR-10 Dataset
Download the cifar-10 dataset and extract it to the `./Data` directory.   
You can change the download parameter `download=True` in the train file 
or download it from the link below (recommended method):
baiduURL：https://pan.baidu.com/s/16U9FhTlv3BVuB3ixipayXg  pw:ifdu 
### 2. training model
The `./train.py` can be used to train models contained in the `./models` module.  
  Many comments are written in the main function `.train.py` to facilitate the interpretation of the training process
 