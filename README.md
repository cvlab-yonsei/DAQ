# PyTorch implementation of DAQ

This is an official implementation of the paper "Distance-aware Quantization", accepted to ICCV2021.

For more information, checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/DAQ/)].

# Getting started 

## Dependencies
* Python 3.6
* PyTorch = 1.5.0


## Datasets
* Cifar-10
    * This can be automatically downloaded by learning our code
* ImageNet
    * This is available at [here](http://www.image-net.org) 

## Training & Evaluation
First, clone our github repository.
```bash
$ git clone https://github.com/cvlab-yonsei/DAQ.git
```
Cifar-10 dataset (ResNet-20 architecture) 

```bash
# Cifar-10 & ResNet-20 W1A1 model
$ python cifar10_train.py --config configs/DAQ/resnet20_DAQ_W1A1.yml
# Cifar-10 & ResNet-20 W1A32 model
$ python cifar10_train.py --config configs/DAQ/resnet20_DAQ_W1A32.yml
```

ImageNet dataset (ResNet-18 architecture)

- Will be released

```bash
# ImageNet & ResNet-18 W1A1 model
# ImageNet & ResNet-18 W1A1 model
```

---
## Citation
```
@inproceedings{kim2021daq,
    author={Kim, Dohyung  and Lee, Junghyup and Ham, Bumsub},
    title={Distance-aware Quantization},
    booktitle={Proceedings of International Conference on Computer Vision},
    year={2021},
}
```
---
## Credit
* ImageNet training code: [[PyTorch official example code](https://github.com/pytorch/examples/blob/master/imagenet/main.py)]
* ResNet-18 models: [[PyTorch official code](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)]
* ResNet-20 model: [[ResNet on CIFAR10](https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py)] [[IRNet](https://github.com/XHPlus/IR-Net/blob/master/resnet-20-cifar10/1w1a/resnet.py)]
* Quantized modules: [[DSQ](https://github.com/ricky40403/DSQ/blob/master/DSQConv.py#L18)]
