# python异常捕获与处理
This repo holds code for [Sktr:Skip-Transformer Unet]

## 什么是异常

### 1. 异常简介
* 异常不是错误

  错误通常指的是语法错误，可以人为避免。
  异常是指在语法逻辑正确的而前提下，出现的问题。

* 异常即是一个事件
  该事件会在程序执行过程中发生，影响了程序的正常执行。一般情况下，在Python无法正常处理程序时就会发生一个异常。异常是Python对象，表示一个错误。当Python脚本发生异常时我们需要捕获处理它，否则程序会终止执行。

当一个未捕获的异常发生时，Python将结束程序并打印一个堆栈跟踪信息，以及异常名和附加信息。具体如下：
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./dataset/README.md"](datasets/README.md) for details, or please send an Email to 2981431354@mail.dlut.edu.cn to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it.

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations

```bibtex
@
  title={Skip-Transformer Unet},
  author={Tianyu Yan, Fuzi Wan},
  journal={投哪一篇呢嘿嘿},
  year={2021}
}
```
