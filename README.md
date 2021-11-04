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
Traceback (most recent call last): 
File "<ipython-input-1-0bc85e309fb1>", line 1, in <module> 
    min(x,y) 
NameError: name 'x' is not defined
```

* 异常捕获主要目的
  * 错误处理：在运行时出现错误的情况下，应用程序可能会无条件终止。使用异常处理，我们可以处理失败的情况并避免程序终止。

  * 代码分离：错误处理可以帮助我们将错误处理所需的代码与主逻辑分离。与错误相关的代码可以放置在“ except ”块中，该块将其与包含应用程序逻辑的常规代码隔离开来。

  * 错误区分：帮助我们隔离执行过程中遇到的不同类型的错误。我们可以有多个“ except”块，每个块处理一种特定类型的错误。

* 异常捕获其他应用
  * 事件通知：异常也可以作为某种条件的信号，而不需要在程序里传送结果标志或显式地测试它们。

  * 特殊情形处理：有时有些情况是很少发生的，把相应的处理代码改为异常处理会更好一些。

  * 特殊的控制流：异常是一个高层次的”goto”，可以把它作为实现特殊控制流的基础。如反向跟踪等。
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
