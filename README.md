# Supervised Dialogue Segmentation
## Introduction
| [English](README.md) | [中文](README-zh.md) | [日本語](README-jp.md) |

## Quick Start
### Installation
```shell
## glone this repository
git clone https://github.com/Coldog2333/SuperDialseg.git
## install super_dialseg package
pip3 install -e .
```
### Download Dataset
For keeping the repository's downloading speed, we place the datasets to the Google Drive.
You can download the supervised datasets from: https://drive.google.com/drive/folders/19YiHVfeI_M4HivrErIi9bghvUvsw9-Ws?usp=sharing


### Segment a dialogue
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
#### Output
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
### Implemented Dialseg Models
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
- [ ] ChatGPT & GPT-4.
- [ ] TextSeg-text
- [ ] TextSeg-dial
- [ ] BERT
- [ ] RoBERTa
- [ ] TOD-BERT
- [ ] T5
- [ ] RetroTS-T5
- [ ] MTRoBERTa
- [ ] MVRoBERTa
