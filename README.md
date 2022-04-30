# CNN-backbone
This is a summary repository of the sota backbone of the CNNs
## Quickly start
### 1. Download CIFAR-10 Dataset
Download the cifar-10 dataset and extract it to the `./Data` directory.   
You can change the download parameter `download=True` in the train file 
or download it from the link below (recommended method):  
baiduURL：https://pan.baidu.com/s/16U9FhTlv3BVuB3ixipayXg  pw:ifdu 
### 2. training model
The `./train.py` can be used to train models contained in the `./models` module.  
  Many comments are written in the main function `.train.py` to facilitate the interpretation of the training process
## Code architecture
```
├─train.py     # The training stage and all parameters
│
├─test.py      # The testing stage demo
│
├─onnxruntime_demo.py     # The testing stage using inference lib:onnxruntime
│
├─Data          # Downloaded dataset's location
│  │
│  └─cifar-10-batches-py
│          batches.meta
│          data_batch_1
│          data_batch_2
│          ...
├─lib
│  │  __init__.py         # No implement
│
├─models                  # cnn models
│  │  DeformLeNet.py
│  │  densenet.py
│  │  GhostNet.py
│  │  LeNet.py
│  │  OctNet.py
│  │  OctResnet.py
│  │  ResNet18.py
│  │  __init__.py
│  │
│  ├─layer                 # some common layer
│     │  conv_layer.py
│     │  deform_conv_v2.py
│     │  OctConv.py
│     │  __init__.py
│
│─utils
│     │  common.py      # some tools,such as Logger func
│
├─output               # trained model will be saved here
│  └─test              # a sample
│          net_best.pth    # Best model for validation
│          net_latest.pth  # Newly trained models
│          train_log.txt   # training log
```