import inspect
import argparse
import torch


class Config:
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    train_epochs = 20
    batch_size = 20
    learning_rate = 1.1
    lr_decay = 'Reciprocal'
    # lr_decay = 'Exponential'
    lr_decay_gamma = 0.92  # lr decay base number of ExponentialLR
    l2_regularization = 1e-6
    loss_margin = 0.1

    answer_dict_file = 'data/insuranceQA/answers.label.token_idx'
    qa_vocab_file = 'data/insuranceQA/vocabulary'
    qa_train_file = 'data/insuranceQA/question.train.token_idx.label'
    qa_dev_file = 'data/insuranceQA/question.dev.label.token_idx.pool'
    qa_test1_file = 'data/insuranceQA/question.test1.label.token_idx.pool'
    qa_test2_file = 'data/insuranceQA/question.test2.label.token_idx.pool'

    word2vec_file = 'embedding/glove.6B.100d.txt'
    punctuation_file = 'data/punctuations.txt'
    trained_model = 'model/model_name_here'  # The trained model saved from training and uses in testing

    train_neg_count = 50  # Amount of random negative answers for every question in training
    q_length = 20
    a_length = 100
    PAD_WORD = '<UNK>'
    # model_name = 'QA-CNN'
    model_name = 'QA-biLSTM'
    # model_name = 'AP-CNN'
    # model_name = 'AP-biLSTM'
    kernel_count = 400
    kernel_size = 3
    rnn_hidden = 150

    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = 'Global Config:\n'
        for key, val in attributes:
            to_str += '\t{} = {}\n'.format(key, val)
        return to_str
