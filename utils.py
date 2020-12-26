import logging
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def get_logger(log_file=None, file_level=logging.INFO, stdout_level=logging.DEBUG, logger_name=__name__):
    logging.root.setLevel(0)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    _logger = logging.getLogger(logger_name)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level=file_level)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=stdout_level)
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)
    return _logger


def process_bar(percent, start_str='', end_str='', auto_rm=True):
    bar = '=' * int(percent * 50)
    bar = '\r{}|{}| {:.1%} | {}'.format(start_str, bar.ljust(50), percent, end_str)
    print(bar, end='', flush=True)
    if percent == 1:
        print(end=('\r' + ' ' * 100 + '\r') if auto_rm else '\n', flush=True)


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


def evaluate(model, dataloader):
    predict = defaultdict(list)
    total = 0
    for qid, q, a, y in dataloader:
        cos = model(q, a)
        for i, pred, label in zip(qid.numpy(), cos.detach().cpu().numpy(), y.numpy()):
            predict[i].append((pred, label))
        total += len(qid)
        process_bar(total / len(dataloader.dataset), start_str='Evaluation ')
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

        for k, v in answer_dict.items():
            ans = [qa_vocab[idx] for idx in answer_dict[k]]
            answer_dict[k] = sent_process(ans, config.a_length)

        if mode == 'train':
            quests, answer_pos, answer_neg = [], [], []
            with open(qa_file, 'r') as f:
                for line in f.readlines():
                    i = line.strip().split('\t')
                    quest = [qa_vocab[idx] for idx in i[0].split()]
                    quest = sent_process(quest, config.q_length)
                    pos_ans = set(i[1].split())
                    for ans_id in pos_ans:
                        quests.append(quest)
                        answer_pos.append(answer_dict[ans_id])
                        answer_neg.append([])
                        neg_ans_ids = set()  # negative sample
                        while len(neg_ans_ids) < config.train_neg_count:
                            rand_idx = np.random.randint(1, len(answer_dict), config.train_neg_count * 2)
                            neg_ans_ids = set([i for i in rand_idx if i not in pos_ans][:config.train_neg_count])
                        for k in neg_ans_ids:
                            answer_neg[-1].append(answer_dict[str(k)])

            self.q = torch.LongTensor(quests)
            self.a_pos = torch.LongTensor(answer_pos)
            self.a_neg = torch.LongTensor(answer_neg)
        else:
            qids, quests, answers, labels = [], [], [], []
            with open(qa_file, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    i = line.strip().split('\t')
                    quest = [qa_vocab[idx] for idx in i[1].split()]
                    quest = sent_process(quest, config.q_length)
                    pos_ans = set(i[0].split())
                    for j in set(list(pos_ans) + i[2].split()):
                        qids.append(idx)
                        quests.append(quest)
                        answers.append(answer_dict[j])
                        labels.append(1 if j in pos_ans else 0)
                    # if idx == 100:
                    #     break

            self.qid = torch.LongTensor(qids)
            self.q = torch.LongTensor(quests)
            self.a = torch.LongTensor(answers)
            self.y = torch.LongTensor(labels)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.q[idx], self.a_pos[idx], self.a_neg[idx]
        return self.qid[idx], self.q[idx], self.a[idx], self.y[idx]

    def __len__(self):
        return self.q.shape[0]

    def __str__(self):
        return f'Dataset {self.mode}: {len(self.q)} samples.'


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
