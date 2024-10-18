# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI'21 Best Paper)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is the origin Pytorch implementation of Informer in the following paper: 
[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436). Special thanks to `Jieqi Peng`@[cookieminions](https://github.com/cookieminions) for building this repo.

:triangular_flag_on_post:**News**(Mar 27, 2023): We will release Informer V2 soon.

:triangular_flag_on_post:**News**(Feb 28, 2023): The Informer's [extension paper](https://www.sciencedirect.com/science/article/pii/S0004370223000322) is online on AIJ.

:triangular_flag_on_post:**News**(Mar 25, 2021): We update all experiment [results](#resultslink) with hyperparameter settings.

:triangular_flag_on_post:**News**(Feb 22, 2021): We provide [Colab Examples](#colablink) for friendly usage.

:triangular_flag_on_post:**News**(Feb 8, 2021): Our Informer paper has been awarded [AAAI'21 Best Paper \[Official\]](https://aaai.org/Conferences/AAAI-21/aaai-outstanding-and-distinguished-papers/)[\[Beihang\]](http://scse.buaa.edu.cn/info/1097/7443.htm)[\[Rutgers\]](https://www.business.rutgers.edu/news/hui-xiong-and-research-colleagues-receive-aaai-best-paper-award)! We will continue this line of research and update on this repo. Please star this repo and [cite](#citelink) our paper if you find our work is helpful for you.

<p align="center">
<img src=".\img\informer.png" height = "360" alt="" align=center />
<br><br>
<b>Figure 1.</b> The architecture of Informer.
</p>

## ProbSparse Attention
The self-attention scores form a long-tail distribution, where the "active" queries lie in the "head" scores and "lazy" queries lie in the "tail" area. We designed the ProbSparse Attention to select the "active" queries rather than the "lazy" queries. The ProbSparse Attention with Top-u queries forms a sparse Transformer by the probability distribution.
`Why not use Top-u keys?` The self-attention layer's output is the re-represent of input. It is formulated as a weighted combination of values w.r.t. the score of dot-product pairs. The top queries with full keys encourage a complete re-represent of leading components in the input, and it is equivalent to selecting the "head" scores among all the dot-product pairs. If we choose Top-u keys, the full keys just preserve the trivial sum of values within the "long tail" scores but wreck the leading components' re-represent.
<p align="center">
<img src=".\img\probsparse_intro.png" height = "320" alt="" align=center />
<br><br>
<b>Figure 2.</b> The illustration of ProbSparse Attention.
</p>

## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0


