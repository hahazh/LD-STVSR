# Introduction

This is the implementation of  paper "Continuous Space-Time Video Super-Resolution Utilizing Long-Range Temporal Information".

Our paper is currently under peer review, detail introduction will be coming soon.

# Pre-trained models

[BaiduCloud](https://pan.baidu.com/s/1eiiLGqhOMman6CPgp9LeFA)

password: kzdi 

model_arb refer to model that trained in space-time arbitrary manner.
model_fix refer to model that trained in fixed space-time scale factor(space 4x and time 2x).


# Environment
We are good in the environment:

python 3.7

CUDA 9.1

Pytorch 1.5.0


# Run a demo


you should specify the GT path and output path first, and run:


```
cd src

python test_vid4.py
```

# Acknowledgment
Our code is built on

 [Zooming-Slow-Mo-CVPR-2020](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)

 [softsplatting](https://github.com/sniklaus/softmax-splatting)

 [open-mmlab](https://github.com/open-mmlab)

 [bicubic_pytorch](https://github.com/sanghyun-son/bicubic_pytorch)
 
 We thank the authors for sharing their codes!
