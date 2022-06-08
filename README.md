# CTAugment Wrapper for PyTorch #

This repository contains the unofficial implementation of the CTAugment Wrapper that can be used with any classification task in PyTorch.

- The implementation of the CTAugment readily with the present training pipelines is bit difficult, as we may have to change and add the piece of code to support it. On the other hand, this repository will come in hand to directly use the CTAugment with any existing image classification training repositories and codebase.

- CTAugment stands for Control Theory Augment which was introduced in the [ReMixMatch Paper](https://arxiv.org/pdf/1911.09785.pdf). 

- It basically weighs the power of an augmentation to be applied on an image by learning through the error(output of the model and real label). 

More details can be obtained from the Section 3.2.2 of the paper.

## Structure of `ctaugment.py` file ##


The `ctaugment.py` contains two main classes i.e `CTAugment` and `CTUpdater`.
  - `CTAugment ` contains two necessary functions. `__call__` is used when CTAugment object is used to apply transform on an image.
    
    `__probe__` is used to change the weights of the bins of CTAugment. It expects the `threshold`, `decay`, and `depth` parameters. By default the parameters are 0.8, 0.99, and 2 respectively.
    
  - `CTUpdater` contains the probe dataset which will be used to update the weights of the bins throughout the training. It expects the path to the training data `datapath` and `train_transforms` which we will be using for training the model.


### How to use the `CTAugment` and `CTUpdater`

```
# Step 1: Add the CTAugment module to the train_transforms list.

from ctaugment import CTAugment, CTUpdater

train_transforms = [
     transforms.RandomResizedCrop((size, size)),
     transforms.RandomHorizontalFlip(),
     CTAugment(depth=depth, thresh=thresh, decay=decay).
]

train_transforms = transforms.Compose(train_transforms)

# Step 2: Initialize the CTUpdater module

updater = CTUpdater(datapath=datapath, 
          train_transforms=train_transforms, 
          batch_size=probe_batch_size, 
          num_workers=num_workers, 
          gpu=$(gpu ID))

NOTE: Keep the probe_batch_size >> train_batch_size. 
In this example it is 3 times the train_batch_size. 
datapath consist of the path to the training data(ImageFolder like structure).

root/dog/xxx.png
root/dog/xxy.png
root/dog/[...]/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/[...]/asd932_.png

# Step 3: Update the weights of the CTAugment.

output = model(input)

loss = loss_fn(output, target)
loss.backward()
optimizer.step()
.
.
.
updater.update(model)

NOTE: Update the weights by calling the update() function of the 

CTUpdater object after the end of every iteration(For simplicity after optimizer.step()). 

You need to pass the model to calculate the error rate on the probe dataset. 
```

## Running the example script

I have attached `main.py` to test and demonstrate the CTAugment using pytorch models. The script is taken from [PyTorch Examples Repository](url\\\(https://github.com/pytorch/examples/tree/main/imagenet). You can visit the given link for more information on the script usage
___

### Installing the dependencies.
The script requires `python >= 3.6`

Other requirements can be installed by running the command `pip3 install -r requirements.txt`
___

### Structure of the script

```
PyTorch Training with support of CTAugment Dataset

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --use-weighted        Use WeightedRandomSampler
  -a ARCH, --arch ARCH  model architecture: alexnet | densenet121 | densenet161 | densenet169 | densenet201 | googlenet | inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 |
                        mnasnet1_3 | mobilenet_v2 | mobilenet_v3_large | mobilenet_v3_small | resnet101 | resnet152 | resnet18 | resnet34 | resnet50 | resnext101_32x8d |
                        resnext50_32x4d | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 | vgg11 | vgg11_bn | vgg13
                        | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn | wide_resnet101_2 | wide_resnet50_2 (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  -nc N, --num-classes N
                        number of classes
  --data-percent N      percentage of training data to take for training
  --epochs N            number of total epochs to run
  --output-dir V        directory to store output weights and logs
  --size N              image size
  --mu-ratio N          multiplicative ratio for ct augment probe loader
  --min-step-lr N       minimum for step lr
  --max-step-lr N       maxiumum for step lr
  --save-every S        save checkpoints every N epochs
  --rand-depth RAND_DEPTH
                        depth of RandAugment
  --rand-magnitude RAND_MAGNITUDE
                        magnitude of RandAugment
  --ct-depth CT_DEPTH   depth of CT Augment
  --ct-decay CT_DECAY   decay of CT Augment
  --ct-thresh CT_THRESH
                        thresh of CT Augment
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  batch size for training/testing
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --use-scheduler       Flag to use the scheduler during the training
  --use-cosine          use Cosine Scheduler
  --use-ct              use CTAugment strategy
  --no-update-ct        Flag that will disable to update the CTAug.
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.

```

- Enable `--use-ct` flag to train the model with CTAugment. By default the script takes RandAugment strategy during the training.
- Parameters of the `CTAugment` class can be changed using the flags `ct-depth`, `--ct-decay`, and `--ct-depth`.
- By default, the CTAugments weights are saved as a `state_dict()` in the model checkpoints file. If you want to resumme the training, just use 
  `--resume` and path to the checkpoint. Note that the weights will only be loaded if the `resume` path contains the CT checkpoints. 
- You can also use `--no-update-ct` to stop the update of the weights during the training.

- Demo Command: `python3 main.py ../../Datasets/cifar_classification/cifar10_classification --use-cosine --lr 0.01 --gpu 0 -b 64 --size 32 --use-ct --use-scheduler --arch resnet18`.
___

## References and Citations

- Dataset making structure referred from [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) repository.
- Training script taken from [PyTorch Examples](https://github.com/pytorch/examples/tree/main/imagenet).

```
@article{berthelot2019remixmatch,
    title={ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring},
    author={David Berthelot and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Kihyuk Sohn and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:1911.09785},
    year={2019},
}
```
___

**Feel free to submit PR if you find any changes or a better approach to implement CTAugment.**
