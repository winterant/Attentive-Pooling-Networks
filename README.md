Attentive Pooling Networks
===
> Implementation for the paper：  
Santos, Cicero dos, Ming Tan, Bing Xiang, and Bowen Zhou. "Attentive pooling networks." arXiv preprint arXiv:1602.03609 (2016).
>https://arxiv.org/abs/1602.03609

This project contains 4 models: QA-CNN, QA-biLSTM, AP-CNN and AP-biLSTM

# Environments
  + python 3.8
  + pytorch 1.7

# Dataset
You need to prepare the following documents:  
1. Dataset(`data/insuranceQA/*`)  
  Source: https://github.com/shuzi/insuranceQA/tree/master/V1

2. Word Embedding(`embedding/glove.6B.100d.txt`)  
Download from https://nlp.stanford.edu/projects/glove

# Running

Train and evaluate:
```
python main.py --model_name QA-biLSTM
```

Only test:
```
python test_only.py --model_name QA-biLSTM --trained_model QA-biLSTM20201224_105449.pt
```

# Experiment

<table align="center">
    <tr>
        <th>Hyper parameter</th>
        <th>QA-CNN</th>
        <th>QA-biLSTM</th>
        <th>AP-CNN</th>
        <th>AP-biLSTM</th>
    </tr>
    <tr>
        <td align="center">epoch size</td>
        <td align="center">-</td>
        <td align="center">20</td>
        <td align="center">150</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">batch size</td>
        <td align="center">-</td>
        <td align="center">20</td>
        <td align="center">20</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">learning rate</td>
        <td align="center">-</td>
        <td align="center">1.1</td>
        <td align="center">1.1</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">lr decay</td>
        <td align="center">Exponential(gamma=0.99)</td>
        <td align="center">Reciprocal</td>
        <td align="center">Reciprocal</td>
        <td align="center">-</td>
    </tr>
    <tr>
        <td align="center">loss margin</td>
        <td align="center">-</td>
        <td align="center">0.1</td>
        <td align="center">0.2</td>
        <td align="center">-</td>
    </tr>
</table>

**Other hyper parameters**: View at `config.py`

**Note1**: In the training, I randomly sample 50 negative answer 
from whole answer pool for every question,
but only use the one with max score (cosine value with question) 
to update our trainable parameters. 

**Note2**: In the training, I make word embedding to be trainable 
and initialize to pre-trained word embedding.

---
Result of accuracy:

| Model Name | dev | test1 | test2 |
|---|---|---|---|
| QA-CNN | - | - | - |
| QA-biLSTM | 66.60% | 67.89% | 63.67% |
| AP-CNN | 54.80% | 53.61% | 50.61% |
| AP-biLSTM | - | - | - |

# Warning

**该项目尚未调试完成，效果没有论文中的实验结果好，请暂时不要用于科研实验！我将继续调试！  
若您发现该代码效果差的原因，恳请您在issue区告知我，不胜感激！-- 2020.12.22**  
**The project has not yet been successful, the performance is very POOR,
please DO NOT use it for scientific research experiment!
I will continue debugging when I have enough time!  
If you find out the reason for the poor performance of the code,
please tell me at issue.
Thank you very much-- 12.22, 2020**
