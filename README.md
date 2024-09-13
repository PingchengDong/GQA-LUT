# Genetic Quantization-Aware Approximation for Non-Linear Operations in Transformers
The official PyTorch implementation of ["Genetic Quantization-Aware Approximation for Non-Linear Operations in Transformers"](http://arxiv.org/abs/2403.19591) [DAC 2024]

![Demo](GQA_FIG.png)

## Installation
Clone this repo with submodules:
```
git clone https://github.com/PingchengDong/GQA-LUT
cd GQA-LUT/
```

The code is tested with Python > 3.7, PyTorch >= 1.5. We recommend you to use [anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. To create an anaconda environment:
```
conda env create -f environment.yml
conda activate gqa_lut
```

## Support List
```
├──Non-linear operations
    ├──GELU
    ├──HSwish
    ├──Sigmoid
    ├──Exponent
    ├──Reciprocal
    ├──Reciprocal of square root
    ├──...
```

## Approximation
Example of approximating ```GELU``` with 8 segpoints:
```
python gqa_lut_trainer.py --act_func 'gelu' --x_range -4 4 --sp_range -4.0 4.0 --num_splits 7 --decimal_bit_range 0 6 --total_iters 500 --mutate
```
We provide some pretrained jsons for several non-linear operations with 8 & 16 segpoints, which are mostly used in neural network in the ```pretrained``` folder.

To assist user in reproducing the results, we provide a Makefile file that includes the hyper-parameter settings and execution methods for several supported non-linear functions.

Example for GQA-LUT approximation of GELU function with 8 segpoints:
```
make gelu_8
```

## Pytorch Operator
After perfoming quantization-aware training of FP32 models, user can replace the original activation functions with ```GQA_LUT``` operator in  ```gqa_lut_op.py```, and then perform a new round of finetuning.

## Citation
```
@inproceedings{dong2024gqalut,
  author    = author={Dong, Pingcheng and Tan, Yonghao and Zhang, Dong and Ni, Tianwei and Liu, Xuejiao and Liu, Yu and Luo, Peng and Liang, Luhong and Liu, Shih-Yang and Huang, Xijie and Zhu, Huaiyu and Pan, Yun and An, Fengwei and Cheng, Kwang-Ting},
  title     = {Genetic Quantization-Aware Approximation for Non-Linear Operations in Transformers},
  booktitle = {Design Automation Conference (DAC)},
  year      = {2024}
}

```
