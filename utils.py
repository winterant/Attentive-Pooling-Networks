import random
import time
from collections import defaultdict

import torch
from torch.utils.data import Dataset


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def process_bar(percent, start_str='', end_str='', auto_rm=True):
    bar = '=' * int(percent * 50)
    bar = '\r{}|{}| {:.1%} | {}'.format(start_str, bar.ljust(50), percent, end_str)
    print(bar, end='', flush=True)
    if percent == 1:
        print(end='\r' if auto_rm else '\n', flush=True)


def load_embedding(word2vec_file):
    with open(word2vec_file, encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_emb.append([0])
        word_dict['<UNK>'] = 0
        for line in f.readlines():
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])
            word_dict[tokens[0]] = len(word_dict)
        word_emb[0] = [0] * len(word_emb[1])
    return word_emb, word_dict


def evaluate(model, dataloader, device):
    predict = defaultdict(list)
    for batch in dataloader:
        qid, q, a, y = batch
        cos = model(q.to(device), a.to(device))
        for i, pred, label in zip(qid.numpy(), cos.detach().cpu().numpy(), y.numpy()):
            predict[i].append((pred, label))
    accuracy = 0
    MRR = 0
    for p in predict.values():
        p.sort(key=lambda x: -x[0])
        if p[0][1] == 1:
            accuracy += 1
        for i, t in enumerate(p):
            if t[1] == 1:
                MRR += 1 / (i + 1)
                break
    accuracy = accuracy / len(predict)
    MRR = MRR / len(predict)
    return accuracy, MRR


class IQADataset(Dataset):
    def __init__(self, word_dict, config, qa_file, mode='train'):
        assert mode in ['train', 'valid', 'test'], '"mode" must be "train","valid" or "test"!'
        self.mode = mode
        pad_num = word_dict[config.PAD_WORD]

        answer_dict, qa_vocab = dict(), dict()
        with open(config.answer_dict_file, 'r') as f:
            for line in f.readlines():
                i = line.strip().split()
                answer_dict[i[0]] = i[1:]
        with open(config.qa_vocab_file, 'r') as f:
            for line in f.readlines():
                i = line.strip().split()
                qa_vocab[i[0]] = i[1]

        def sent_process(sent, p_len):  # vocab to id -> padding
            return [word_dict.get(w.lower(), pad_num) for w in sent[:p_len]] + [pad_num] * (p_len - len(sent))

        if mode == 'train':
            quests, answer1, answer2 = [], [], []
            with open(qa_file, 'r') as f:
                for line in f.readlines():
                    i = line.strip().split('\t')
                    quest = [qa_vocab[idx] for idx in i[0].split()]
                    pos_ans = set(i[1].split())
                    for ans_id in pos_ans:
                        ans1 = [qa_vocab[idx] for idx in answer_dict[ans_id]]
                        neg_ans_ids = set()  # answer ids had sampled
                        for j in range(config.train_neg_count):
                            while True:
                                k = random.randint(1, len(answer_dict) - 1)
                                if str(k) not in pos_ans and k not in neg_ans_ids:
                                    break
                                neg_ans_ids.add(k)
                            ans2 = [qa_vocab[idx] for idx in answer_dict[str(k)]]
                            quests.append(sent_process(quest, config.q_length))
                            answer1.append(sent_process(ans1, config.a_length))
                            answer2.append(sent_process(ans2, config.a_length))

            self.q = torch.LongTensor(quests)
            self.a1 = torch.LongTensor(answer1)
            self.a2 = torch.LongTensor(answer2)
        else:
            qids, quests, answers, labels = [], [], [], []
            with open(qa_file, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    i = line.strip().split('\t')
                    quest = [qa_vocab[idx] for idx in i[1].split()]
                    pos_ans = set(i[0].split())
                    for j in set(list(pos_ans) + i[2].split()):
                        ans = [qa_vocab[idx] for idx in answer_dict[j]]
                        qids.append(idx)
                        quests.append(sent_process(quest, config.q_length))
                        answers.append(sent_process(ans, config.a_length))
                        labels.append(1 if j in pos_ans else 0)
                    # if idx == 100:
                    #     break

            self.qid = torch.LongTensor(qids)
            self.q = torch.LongTensor(quests)
            self.a = torch.LongTensor(answers)
            self.y = torch.LongTensor(labels)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.q[idx], self.a1[idx], self.a2[idx]
        return self.qid[idx], self.q[idx], self.a[idx], self.y[idx]

    def __len__(self):
        return self.q.shape[0]

    def print_info(self):
        print("-------  数据集{}  ---------".format(self.mode))
        print('样本数量：', len(self.q))
        if self.mode in ['test', 'valid']:
            print('正样本数量：', sum(self.y))
            print('负样本数量：', len(self.y) - sum(self.y))
        print("-------------------------------")


# The following delay way of learning rate is proposed by the author of paper "Attentive Pooling Networks"
class ReciprocalLR:
    def __init__(self, optimizer):
        self.opt = optimizer
        self.init_lr = optimizer.state_dict()['param_groups'][0]['lr']
        self.epoch = 1

    def step(self):
        self.epoch += 1
        for p in self.opt.param_groups:
            p['lr'] = self.init_lr / self.epoch

    def get_last_lr(self):
        return [self.init_lr / self.epoch]
