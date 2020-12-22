import time
import torch
from torch.utils.data import DataLoader

from config import Config
from utils import date, load_embedding, IQADataset, evaluate

if __name__ == '__main__':
    config = Config()

    print(f'{date()}## Load embedding and test data...')
    word_emb, word_dict = load_embedding(config.word2vec_file)
    test_data = IQADataset(word_dict, config, config.qa_test2_file, mode='test')
    test_data.print_info()
    test_dlr = DataLoader(test_data, batch_size=config.batch_size, num_workers=4)
    model = torch.load(config.model_file).to(config.device)

    start_time = time.perf_counter()
    accuracy, MRR = evaluate(model, test_dlr, device=config.device)
    end_time = time.perf_counter()
    print(f'{date()}## Test accuracy {accuracy * 100:.2f}%; MRR {MRR:.5f}; Time used {end_time - start_time:.0f}S.')

    # The following is a sentence by sentence analysis. Please set breakpoints and run in debug mode
    test_dlr = DataLoader(test_data, batch_size=1)
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
            cos = model(q.to(config.device), a.to(config.device))
