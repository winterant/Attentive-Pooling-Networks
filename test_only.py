import time
import torch
from torch.utils.data import DataLoader

from config import Config
from utils import date, load_embedding, IQADataset, evaluate


def test(dataset, test_batch_size, model):
    test_dlr = DataLoader(dataset, batch_size=test_batch_size, num_workers=4)
    start_time = time.perf_counter()
    accuracy, MRR = evaluate(model, test_dlr)
    end_time = time.perf_counter()
    print(f'{date()}## Test accuracy {accuracy * 100:.2f}%; MRR {MRR:.5f}; Time used {end_time - start_time:.0f}S.')


def test_analysis(dataset, test_batch_size, model):
    # Sentence by sentence analysis. Please set breakpoints and run in debug mode
    test_dlr = DataLoader(dataset, batch_size=test_batch_size, num_workers=4)
    words = {i: w for w, i in word_dict.items()}
    for batch in test_dlr:
        qid, q, a, y = batch
        if y[0].item() == 1:
            quest = [words[int(i)] for i in q[0]]
            answer = [words[int(i)] for i in a[0]]
            print([i for i in enumerate(quest)])
            print([i for i in enumerate(answer)])
            print(' '.join(quest))
            print(' '.join(answer))
            cos = model(q, a)


if __name__ == '__main__':
    config = Config()

    print(f'{date()}## Load embedding and test data...')
    word_emb, word_dict = load_embedding(config.word2vec_file)
    valid_data = IQADataset(word_dict, config, config.qa_dev_file, mode='test')
    test1_data = IQADataset(word_dict, config, config.qa_test1_file, mode='test')
    test2_data = IQADataset(word_dict, config, config.qa_test2_file, mode='test')
    print(valid_data)
    print(test1_data)
    print(test2_data)

    # config.trained_model = 'model/QA-biLSTM20201224_123138.pt'
    Model = torch.load(config.trained_model)
    # Model = torch.nn.DataParallel(Model)
    test(valid_data, config.test_batch_size, Model)
    test(test1_data, config.test_batch_size, Model)
    test(test2_data, config.test_batch_size, Model)
    # test_analysis(test2_data, 1, Model)
