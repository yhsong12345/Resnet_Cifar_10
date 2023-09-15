# ResNet_Cifar_10

This is the practice for the implementation of ResNet. This model is trained with Cifar 10.

## Prerequisites
- Pytorch 2.0.1
- Python 3.11.4
- Window 11
- conda 23.7.4

## Training
```
# GPU training
python train.py -m Resnet20 -e 200 -lr 0.01 -b 128 -s 32 -d outputs
```

## Testing
```
python test.py -m Resnet20 -e 200 -lr 0.01 -b 128 -s 32 -d outputs
```

## Result (Accuracy)

| Model             | Acc.        |
| ----------------- | ----------- |
| [ResNet20](https://drive.google.com/file/d/1BIklR-0qXeWw9zhEscAPZZQrk6Q98zFQ/view?usp=drive_link)          | 93.02%      |
| [ResNet32](https://drive.google.com/file/d/1ekH2JjeBiaUtZ2cUxP63PWg0DQlKm8vj/view?usp=drive_link)          | 93.62%      |
| [ResNet44](https://drive.google.com/file/d/1TqbykyFFvf2QxZbwv-k3G0L9iJ90Hd1e/view?usp=drive_link)         | 93.75%      |
| [ResNet56](https://drive.google.com/file/d/1u_k_acCgvQYCjbQdWcJ43mzvka6llX3e/view?usp=drive_link)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
