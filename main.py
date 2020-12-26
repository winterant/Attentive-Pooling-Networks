import os
import pickle
import time
import torch
from torch.utils.data import DataLoader

from config import Config
from utils import date, load_embedding, IQADataset, evaluate, process_bar, ReciprocalLR, get_logger
from model import QAModel


def train(train_dataloader, valid_dataloader, model, config, model_path):
    logger.debug(f'Start the training')
    accuracy, MRR = evaluate(model, valid_dataloader)
    logger.info(f"Initial valid accuracy {accuracy * 100:.2f}%, MRR {MRR:.6f}")

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
        for q, a_pos, a_neg in train_dataloader:
            cos_pos = model(q, a_pos)
            # Only the negative answer with max cosine value updates parameters
            input_q = q.unsqueeze(-2).expand(-1, a_neg.shape[-2], -1).reshape(-1, q.shape[-1])
            input_a = a_neg.view(-1, a_neg.shape[-1])
            cos_neg = model(input_q, input_a)
            cos_neg = cos_neg.view(len(q), -1).max(dim=-1)[0]

            loss = torch.max(torch.zeros(1).to(config.device), config.loss_margin - cos_pos + cos_neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(q)
            total_samples += len(q)
            process_bar(total_samples / len(train_dataloader.dataset), start_str=f'Epoch {epoch}')
        curr_lr = lr_sch.get_last_lr()[0]
        lr_sch.step()
        model.eval()

        train_loss = total_loss / total_samples
        accuracy, MRR = evaluate(model, valid_dataloader)
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module, model_path)
            else:
                torch.save(model, model_path)
        logger.info(f'Epoch {epoch:3d}; learning rate {curr_lr:.4f}; train loss {train_loss:.6f}; '
                    f'valid accuracy {accuracy * 100:.2f}%, MRR {MRR:.4f}')
    end_time = time.perf_counter()
    logger.info(f'End of training! Time used {end_time - start_time:.0f} seconds.')


def test(test_dataset, test_batch_size, model):
    test_dlr = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=4)
    start_time = time.perf_counter()
    accuracy, MRR = evaluate(model, test_dlr)
    end_time = time.perf_counter()
    logger.info(f'Test accuracy {accuracy * 100:.2f}%; MRR {MRR:.4f} Time used {end_time - start_time:.0f}S.')


if __name__ == '__main__':
    config = Config()
    start_dt = date("%Y%m%d_%H%M%S")
    log_file = f'log/{config.model_name}{start_dt}.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # mkdir if not exist
    logger = get_logger(log_file)
    logger.info(config)

    logger.debug(f'Loading word embedding')
    word_emb, word_dict = load_embedding(config.word2vec_file)

    logger.debug(f'Loading dataset')
    train_data = IQADataset(word_dict, config, config.qa_train_file, mode='train')
    # pickle.dump(train_data, open(os.path.abspath(f'data/train_neg{config.train_neg_count}.pkl'), 'wb'))
    # train_data = pickle.load(open(os.path.abspath(f'data/train_neg{config.train_neg_count}.pkl'), 'rb'))
    valid_data = IQADataset(word_dict, config, config.qa_dev_file, mode='valid')
    logger.info(train_data)
    logger.info(valid_data)
    train_dlr = DataLoader(train_data, batch_size=config.batch_size, num_workers=4)
    valid_dlr = DataLoader(valid_data, batch_size=config.test_batch_size, num_workers=4)

    # Train
    Model = QAModel(config, word_emb).to(config.device)
    # Model = torch.nn.DataParallel(Model)
    save_path = f'model/{config.model_name}{start_dt}.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # mkdir if not exist
    train(train_dlr, valid_dlr, Model, config, save_path)
    del train_data, train_dlr

    logger.debug('Start to evaluate')
    Model = torch.load(save_path)
    # Model = torch.nn.DataParallel(Model)
    # Dev
    test(valid_data, config.test_batch_size, Model)

    # Test1
    test_data = IQADataset(word_dict, config, config.qa_test1_file, mode='test')
    test(test_data, config.test_batch_size, Model)

    # Test2
    test_data = IQADataset(word_dict, config, config.qa_test2_file, mode='test')
    test(test_data, config.test_batch_size, Model)
