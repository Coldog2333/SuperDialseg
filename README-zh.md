# 有监督对话分割
## 前言
| [English](README.md) | [中文](README-zh.md) | [日本語](README-jp.md) |

## 快速开始
### 安装
```shell
## clone github仓库
git clone https://github.com/Coldog2333/SuperDialseg.git
## 安装super_dialseg包
pip3 install -e .
```
### 下载数据集
Google Drive：https://drive.google.com/drive/folders/19YiHVfeI_M4HivrErIi9bghvUvsw9-Ws?usp=sharing


## 分割第一个对话
```python
from super_dialseg import TexttilingSegmenter

segmenter = TexttilingSegmenter(w=10, k=6)

dialogue = {
    'utterances': [
        'User: Hello, I\'d like to learn about your retirement program.',
        'Agent: Are you looking for family benefits?',
        'User: Not necessarily.',
        'Agent: Would you like to learn about maximum family benefits?',
        'User: Sure.',
        'Agent: Do any of your children qualify for benefits?',
        'User: I believe so.',
        'Agent: As for as the maximum family benefits, there is an upper payout limit.',
        'User: Will my child receive the full payment up front or monthly payments?',
        'Agent: If your child qualifies, he or she will receive monthly payments.'
    ]
}

predictions = segmenter.forward(dialogue)

segmenter(dialogue['utterances'])
```
#### 输出
```
User: Hello, I'd like to learn about your retirement program.
Agent: Are you looking for family benefits?
User: Not necessarily.
Agent: Would you like to learn about maximum family benefits?
User: Sure.
Agent: Do any of your children qualify for benefits?
User: I believe so.
Agent: As for as the maximum family benefits, there is an upper payout limit.
--------------------
User: Will my child receive the full payment up front or monthly payments?
Agent: If your child qualifies, he or she will receive monthly payments.
```

## TODO
### 现支持的对话分割模型
- [X] RandomSegmenter, EvenSegmenter
- [X] BayesSegmenter
- [X] TexttilingSegmenter
- [ ] GraphSeg
- [X] EmbeddingSegmenter
- [X] TexttilingCLSSegmenter
- [X] GreedySegmenter
- [X] TexttilingNSPSegmenter
- [X] CSMSegmenter
- [ ] InstructGPT
- [ ] 增加ChatGPT和GPT-4
- [ ] TextSeg-text
- [ ] TextSeg-dial
- [ ] BERT
- [ ] RoBERTa
- [ ] TOD-BERT
- [ ] T5
- [ ] RetroTS-T5
- [ ] MTRoBERTa
- [ ] MVRoBERTa
