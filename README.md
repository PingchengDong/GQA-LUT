# Genetic Quantization-Aware Approximation for Non-Linear Operations in Transformers
PyTorch implementation of paper "Genetic Quantization-Aware Approximation for Non-Linear Operations in Transformers", DAC 2024.

> Genetic Quantization-Aware Approximation for Non-Linear Operations in Transformers
> [Pingcheng Dong](https://pingchengdong.github.io/), [Yonghao Tan](https://yonghao-tan.github.io/), [Dong Zhang](https://dongzhang89.github.io/), [Tianwei Ni](https://twni2016.github.io/), [Xuejiao Liu](https://www.linkedin.com/in/xuejiao-liu-2b007a144/), [Yu Liu](https://www.researchgate.net/profile/Yu-Liu-133), [Peng Luo](https://www.linkedin.com/in/peng-luo-4ab1a564/), [Luhong Liang](https://www.linkedin.com/in/luhongliang/), [Shih-Yang Liu](https://nbasyl.github.io/), [Xijie Huang](https://huangowen.github.io/), [Huaiyu Zhu](https://person.zju.edu.cn/en/zhuhuaiyu), [Yun Pan](https://person.zju.edu.cn/en/panyun), [Fengwei An](https://www.sustech.edu.cn/en/faculties/anfengwei.html), [Kwang-Ting Cheng](https://seng.hkust.edu.hk/about/people/faculty/tim-kwang-ting-cheng)
> DAC 2024
> 

### [Project page](https://github.com/PingchengDong/GQA-LUT) | [Paper](http://arxiv.org/abs/2403.19591)

## Installation
Clone this repo with submodules:
```
git clone https://github.com/PingchengDong/GQA-LUT
cd GQA-LUT/
```

The code is tested with Python3.7, PyTorch == 1.5. We recommend you to use [anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. To create an anaconda environment:
```
conda env create -f environment.yml
conda activate gqa-lut
```

## Training
Example: train ```GELU``` with 8 segpoints:
```
python gqa-lut.py --act_func 'gelu' --x_range -4 4 --sp_range -4.0 4.0 --num_splits 7 --decimal_bit_range 0 6 --total_iters 500 --mutate
```

At the same time, to assist you in reproducing our results as accurately as possible, we provide a Makefile file. It includes the parameter settings and execution methods for several supported non-linear functions in the GQA-LUT code mentioned above.

For example, for GQA-LUT approximation of GELU function with 8 segpoints, running:
```
make gelu_8
```

## Citation
```
@inproceedings{dong2024gqalut,
  author    = author={Dong, Pingcheng and Tan, Yonghao and Zhang, Dong and Ni, Tianwei and Liu, Xuejiao and Liu, Yu and Luo, Peng and Liang, Luhong and Liu, Shih-Yang and Huang, Xijie and Zhu, Huaiyu and Pan, Yun and An, Fengwei and Cheng, Kwang-Ting},
  title     = {Genetic Quantization-Aware Approximation for Non-Linear Operations in Transformers},
  booktitle = {Design Automation Conference (DAC)},
  year      = {2024}
}

```
