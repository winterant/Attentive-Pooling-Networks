import os
import pickle
import time
import torch
from torch.utils.data import DataLoader

from config import Config
from utils import date, load_embedding, IQADataset, evaluate, process_bar, ReciprocalLR
from model import QAModel


def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training')
    accuracy, MRR = evaluate(model, valid_dataloader, config.device)
    print(f"{date()}## Initial valid accuracy {accuracy * 100:.2f}%, MRR {MRR:.6f}")

    start_time = time.perf_counter()
    opt = torch.optim.SGD(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    if config.lr_decay == 'Exponential':
        lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.lr_decay_gamma)
    else:  # Reciprocal
        lr_sch = ReciprocalLR(opt)

    max_accuracy = 0
    for epoch in range(config.train_epochs):
        model.train()
        total_loss, total_samples = 0, 0
        for batch in train_dataloader:
            q, a1, a2 = map(lambda x: x.to(config.device), batch)
            cos1 = model(q, a1)
            cos2 = model(q, a2)
            loss = torch.max(torch.zeros(1).to(config.device), config.loss_margin - cos1 + cos2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_samples += len(cos1)
            process_bar(total_samples / len(train_dataloader.dataset), start_str=f'Epoch {epoch}')
        curr_lr = lr_sch.get_last_lr()[0]
        lr_sch.step()
        model.eval()

        train_loss = total_loss / total_samples
        accuracy, MRR = evaluate(model, valid_dataloader, config.device)
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            torch.save(model, model_path)
        print(f'{date()}#### Epoch {epoch:3d}; learning rate {curr_lr:.4f}; train loss {train_loss:.6f}; '
              f'valid accuracy {accuracy * 100:.2f}%, MRR {MRR:.4f}')
    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def test(test_dataloader, model):
    start_time = time.perf_counter()
    accuracy, MRR = evaluate(model, test_dataloader, device=config.device)
    end_time = time.perf_counter()
    print(f'{date()}## Test accuracy {accuracy * 100:.2f}%; MRR {MRR:.4f} Time used {end_time - start_time:.0f}S.')


if __name__ == '__main__':
    config = Config()
    print(config)

    print(f'{date()}## Loading word embedding')
    word_emb, word_dict = load_embedding(config.word2vec_file)

    print(f'{date()}## Loading dataset')
    try:
        f = open(os.path.abspath(f'data/train_neg{config.train_neg_count}.pkl'), 'rb')
        train_data = pickle.load(f)
    except:
        train_data = IQADataset(word_dict, config, config.qa_train_file, mode='train')
        pickle.dump(train_data, open(os.path.abspath(f'data/train_neg{config.train_neg_count}.pkl'), 'wb'))
    valid_data = IQADataset(word_dict, config, config.qa_dev_file, mode='valid')
    train_data.print_info()
    valid_data.print_info()
    train_dlr = DataLoader(train_data, batch_size=config.batch_size, num_workers=4, shuffle=True)
    valid_dlr = DataLoader(valid_data, batch_size=config.batch_size, num_workers=4)

    Model = QAModel(config, word_emb).to(config.device)
    os.makedirs(os.path.dirname(config.model_file), exist_ok=True)  # mkdir if not exist
    train(train_dlr, valid_dlr, Model, config, config.model_file)

    test_data = IQADataset(word_dict, config, config.qa_test1_file, mode='test')
    test_dlr = DataLoader(test_data, batch_size=config.batch_size, num_workers=4)
    test(test_dlr, torch.load(config.model_file))
