# AlexNet-PyTorch-Implementation  
This is a simple implementaiton of AlexNet to classify cats and dogs, as introduced in the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al. ([original paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf))

## Requirments
- torch==1.11.0
- torchvision==0.12.0
- numpy==1.20.3
- Pillow==8.1.2
- tqdm==4.15.0

```bash
pip3 install -r requirements.txt
```
## Dataset structure
├───Data  
│   ├───cats  
│   │       img1.jpg  
│   │       img2.jpg  
│   │  
│   ├───dogs  
│   │       img1.jpg  
│   │       img2.jpg  
       .  
       .  
│   ├───class  
│   │       img1.jpg  
│   │       img2.jpg  

## Training
To train model below is the example of the command. `--data_dir` and `--checkpoint_dir` arguments are mandatory and other arguments are optional. If not specified model will assume default arguments.
```bash
python train.py --data_dir .\Data\ --ckpt_dir .\checkpoints\ --classes 2 --batch_size 128 --epochs 90 --train_split 0.8 --lr 0.0001
```
