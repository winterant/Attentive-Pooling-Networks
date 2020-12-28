Attentive Pooling Networks
===
> Implementation for the paper：  
Santos, Cicero dos, Ming Tan, Bing Xiang, and Bowen Zhou. "Attentive pooling networks." arXiv preprint arXiv:1602.03609 (2016).
>https://arxiv.org/abs/1602.03609

This project contains 4 models: QA-CNN, QA-biLSTM, AP-CNN and AP-biLSTM.

# Environments

+ python 3.8
+ pytorch 1.7

# Dataset
You have to prepare the following documents:  
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
python test_only.py --model_name QA-biLSTM --trained_model model/QA-biLSTM20201224_105449.pt
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
        <td align="center">40</td>
        <td align="center">20</td>
        <td align="center">100</td>
        <td align="center">40</td>
    </tr>
    <tr>
        <td align="center">batch size</td>
        <td align="center">20</td>
        <td align="center">20</td>
        <td align="center">20</td>
        <td align="center">20</td>
    </tr>
    <tr>
        <td align="center">init. learning rate</td>
        <td align="center">0.5</td>
        <td align="center">11.0</td>
        <td align="center">0.01</td>
        <td align="center">0.7</td>
    </tr>
    <tr>
        <td align="center">lr decay</td>
        <td align="center">Exponential(0.96)</td>
        <td align="center">Reciprocal</td>
        <td align="center">Exponential(0.99)</td>
        <td align="center">Exponential(0.92)</td>
    </tr>
    <tr>
        <td align="center">loss margin</td>
        <td align="center">0.1</td>
        <td align="center">0.1</td>
        <td align="center">0.5</td>
        <td align="center">0.2</td>
    </tr>
    <tr>
        <td align="center">kernel count/size</td>
        <td align="center">4000/2</td>
        <td align="center"></td>
        <td align="center">400/3</td>
        <td align="center"></td>
    </tr>
    <tr>
        <td align="center">rnn hidden size</td>
        <td align="center"></td>
        <td align="center">150</td>
        <td align="center"></td>
        <td align="center">150</td>
    </tr>
</table>

**Note 1**: In the training, I randomly sample 50 negative answer 
from whole answer pool for every question,
but only use the one with max score (cosine value with question) 
to update our trainable parameters. 

**Note 2**: In the training, I make word embedding to be trainable 
and initialize to pre-trained word embedding.


<table align="center">
    <tr>
        <th>Accuracy</th>
        <th>dev</th>
        <th>test1</th>
        <th>test2</th>
    </tr>
    <tr>
        <td>QA-CNN</td>
        <td>59.90%</td>
        <td>60.94%</td>
        <td>56.22%</td>
    </tr>
    <tr>
        <td>QA-biLSTM</td>
        <td>66.90%</td>
        <td>66.11%</td>
        <td>63.00%</td>
    </tr>
    <tr>
        <td>AP-CNN</td>
        <td>61.90%</td>
        <td>63.00%</td>
        <td>59.17%</td>
    </tr>
    <tr>
        <td>AP-biLSTM</td>
        <td>65.20%</td>
        <td>67.00%</td>
        <td>62.72%</td>
    </tr>
</table>

**Note 3**: Accuracy is equivalent to precision at top one answer
among about 500 answers for every question.
In other words, it's considered to be correct if the model find out a right answer
among a set of 500 answers which contains 1~3 right answers.

**Note 4**: We can see running log in folder `log/` to learn more about training process.
