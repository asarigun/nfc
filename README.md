# Graph Node-Feature Convolution for Representation Learning in TensorFlow


<p align="center">
  <a href="https://arxiv.org/abs/1812.00086"><img src="https://img.shields.io/badge/Paper-Report-red"/></a>
  <a href="https://github.com/LiZhang-github/NFC-GCN"><img src="https://img.shields.io/badge/Official-Code-ff69b4"/></a>
  <a href="https://github.com/asarigun/nfc-gcn/blob/main/LICENSE"><img src="https://img.shields.io/github/license/thudm/cogdl"/></a>
  <a href="https://colab.research.google.com/drive/1yJkNXZmZLa3uUTc3wwn5VhE3KsExMWDi?usp=sharing" alt="license"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
</p>


<p align="center"><img width="20%" src="https://github.com/asarigun/nfc-gcn/blob/main/images/tensorflow_logo.png"></p>

Implementation of Graph Node-Feature Convolution for Representation Learning in TensorFlow

![Graph Node-Feature Convolution for Representation Learning](https://github.com/asarigun/nfc-gcn/blob/main/images/figure.jpg)

Graph convolutional network (GCN) is an emerging neural network approach.  It learns new representation of a node by aggregating feature vectors of all neighbors in the aggregation process without considering whether the neighbors or features are useful or not. Recent methods have improved solutions by sampling a fixed size set of neighbors, or assigning different weights to different neighbors in the aggregation process, but features within a feature vector are still treated equally in the aggregation process. In this paper, a new convolution operation is introduced on regular size feature maps constructed from features of a fixed node bandwidth via sampling to get the first-level node representation, which is then passed to a standard GCN to learn the second-level node representation. [1]<!--[[1](https://arxiv.org/abs/1812.00086)]-->

Li  Zhang , Heda Song, Haiping  Lu, 2018, [Graph Node-Feature Convolution for Representation Learning](https://arxiv.org/abs/1812.00086) 

For official implementation, you can visit [![report](https://img.shields.io/badge/Official-Code-yellow)](https://github.com/LiZhang-github/NFC-GCN)


## Requirements
* tensorflow_version 1.x

## Training

```bash
python train.py
```
You can also try out in colab if you don't have any requirements! <a href="https://colab.research.google.com/drive/1yJkNXZmZLa3uUTc3wwn5VhE3KsExMWDi?usp=sharing" alt="license"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>

Note: Since random inits, your training results may not exact the same as reported in the paper!

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), 
* an N by D feature matrix (D is the number of features per node), and
* an N by E binary label matrix (E is the number of classes). [2]<!--[[2](https://arxiv.org/abs/1609.02907)]-->

Have a look at the `load_data()` function in `utils.py` for an example.

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://www.cs.umd.edu/~sen/lbc-proj/LBC.html. In our version (see `data` folder) we use dataset splits provided by https://github.com/kimiyoung/planetoid (Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861), ICML 2016). 

You can specify a dataset by editing `train.py`
<!--
You can specify a dataset as follows: -->
<!--
* For Citeseer: 
```bash
python train.py --dataset citeseer
```
* For Cora: 
```bash
python train.py --dataset cora
```
* For Pubmed: 
```bash
python train.py --dataset pubmed
``` 
(or by editing `train.py`) -->


## Reference

[1] [Zhang, Song, Lu, Graph Node-Feature Convolution for Representation Learning, 2018](https://arxiv.org/abs/1812.00086)  [![report](https://img.shields.io/badge/Official-Code-yellow)](https://github.com/LiZhang-github/NFC-GCN)

[2] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)  [![report](https://img.shields.io/badge/Official-Code-ff69b4)](https://github.com/tkipf/gcn)

## Citation 

```bibtex
@article{zhang2018graph,
  title={Graph node-feature convolution for representation learning},
  author={Zhang, Li and Song, Heda and Lu, Haiping},
  journal={arXiv preprint arXiv:1812.00086},
  year={2018}
}
``` 
```bibtex
@article{kipf2016semi,
  title={Semi-supervised classification with graph convolutional networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
``` 

