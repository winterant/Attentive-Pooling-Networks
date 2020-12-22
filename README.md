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
python main.py --net_name AP-CNN
```

Only test:
```
python test.py --net_name AP-CNN --model_file model/best_model.pt
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
        <td>learning rate</td>
        <td>-</td>
        <td>-</td>
        <td>0.05</td>
        <td>-</td>
    </tr>
    <tr>
        <td>lr decay</td>
        <td colspan="4">Exponential(gamma=0.99)</td>
    </tr>
    <tr>
        <td>l2 regularization</td>
        <td colspan="4">1e-6</td>
    </tr>
    <tr>
        <td>loss margin</td>
        <td colspan="2">0.2</td>
        <td>0.5</td>
        <td>0.2</td>
    </tr>
    <tr>
        <td>test1 accuracy</td>
        <td>-</td>
        <td>-</td>
        <td>44.80%</td>
        <td>-</td>
    </tr>
</table>

# Warning
**该项目尚未调试完成，效果远远没有论文中的实验结果好，请不要用于科研实验！我将在时间充裕时继续调试！  
若您发现该代码效果差的原因，恳请您在issue区告知我，不胜感激！-- 2020.12.22**  
**The project has not yet been successful, the performance is very POOR,
please DO NOT use it for scientific research experiment!
I will continue debugging when I have enough time!  
If you find out the reason for the poor performance of the code,
please tell me at issue.
Thank you very much-- 12.22, 2020**
