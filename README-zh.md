# Supervised Dialogue Segmentation / 有监督对话分割
## 前言
| [English](README.md) | [中文](README-zh.md) | [日本語](README-jp.md) |

## 快速开始
### 安装
```shell
## clone github仓库
git clone https://github.com/Coldog2333/SuperDialseg.git
## 安装superdialseg包
pip3 install -e .
```
### 下载数据集
Google Drive：https://drive.google.com/drive/folders/19YiHVfeI_M4HivrErIi9bghvUvsw9-Ws?usp=sharing


## 分割第一个对话
```python
from models.texttiling.modeling_texttiling import TexttilingSegmenter

segmenter = TexttilingSegmenter(w=10, k=6)

dialogue = {'utterances': ['Hello, how are you today?', 'It is fine, how about you?', 'Yes, good. Do you know what is dialogue segmentation?', 'I dont know, can you explain to me?', 'Of course. It is ...']}

predictions = segmenter.forward(dialogue)
```

## TODO
### 现支持的对话分割模型
- [X] RandomSegmenter, EvenSegmenter
- [X] TexttilingSegmenter
- [X] TexttilingCLSSegmenter
- [X] GreedySegmenter
- [X] TexttilingNSPSegmenter
- [X] CSMSegmenter
- [] BayesSeg
- [] GraphSeg
- [] TextTiling+Glove
- [] TextSeg-text
- [] TextSeg-dial
- [] BERT
- [] RoBERTa
- [] TOD-BERT
- [] T5
- [] RetroTS-T5
- [] MTRoBERTa
- [] MVRoBERTa
- [] InstructGPT
- [] 增加ChatGPT和GPT-4.

