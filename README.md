# Supervised Dialogue Segmentation
## Introduction
| [English](README.md) | [中文](README-zh.md) | [日本語](README-jp.md) |

## Quick Start
### Installation
For keeping the repository's downloading speed, we place the datasets to the Google Drive.
You can download the supervised datasets from: https://drive.google.com/drive/folders/19YiHVfeI_M4HivrErIi9bghvUvsw9-Ws?usp=sharing

```shell
git clone https://github.com/Coldog2333/SuperDialseg.git
pip3 install -e .
```

### Segment a dialogue
```python
from models.texttiling.modeling_texttiling import TexttilingSegmenter

segmenter = TexttilingSegmenter(w=10, k=6)

dialogue = {'utterances': ['Hello, how are you today?', 'It is fine, how about you?', 'Yes, good. Do you know what is dialogue segmentation?', 'I dont know, can you explain to me?', 'Of course. It is ...']}

predictions = segmenter.forward(dialogue)
```

## TODO
### Implemented Dialseg Models
- [X] RandomSegmenter, EvenSegmenter
- [X] TexttilingSegmenter
- [X] TexttilingCLSSegmenter
- [X] GreedySegmenter
- [X] TexttilingNSPSegmenter
- [X] CSMSegmenter
- [X] BayesSegmenter
- [ ] GraphSeg
- [ ] TextTiling+Glove
- [ ] InstructGPT
- [ ] TextSeg-text
- [ ] TextSeg-dial
- [ ] BERT
- [ ] RoBERTa
- [ ] TOD-BERT
- [ ] T5
- [ ] RetroTS-T5
- [ ] MTRoBERTa
- [ ] MVRoBERTa
- [ ] InstructGPT
- [ ] ChatGPT & GPT-4.
