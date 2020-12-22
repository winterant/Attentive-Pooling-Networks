import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, word_dim, kernel_count, kernel_size):
        super().__init__()
        self.encode = nn.Conv1d(
            in_channels=word_dim,
            out_channels=kernel_count,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2)

    def forward(self, vec):
        latent = self.encode(vec.permute(0, 2, 1))
        return latent


class BiLSTM(nn.Module):
    def __init__(self, word_dim, hidden_size):
        super().__init__()
        self.encode = nn.LSTM(input_size=word_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)

    def forward(self, vec):
        self.encode.flatten_parameters()
        latent, _ = self.encode(vec)
        return latent.transpose(-1, -2)


class CoAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size))

    def forward(self, Q, A):
        G = Q.transpose(-1, -2) @ self.U.expand(Q.shape[0], -1, -1) @ A
        G = torch.tanh(G)
        Q_pooling = G.max(dim=-1).values
        A_pooling = G.max(dim=-2).values
        Q_pooling = Q_pooling.softmax(dim=-1)
        A_pooling = A_pooling.softmax(dim=-1)
        rq = Q @ Q_pooling.unsqueeze(-1)
        ra = A @ A_pooling.unsqueeze(-1)
        rq = rq.squeeze(-1)
        ra = ra.squeeze(-1)
        return rq, ra


class QAModel(nn.Module):

    def __init__(self, config, word_emb):
        assert config.net_name in ['QA-CNN', 'QA-biLSTM', 'AP-CNN', 'AP-biLSTM'], 'Net name is incorrect!'
        super().__init__()
        self.net_name = config.net_name
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))

        if 'CNN' in config.net_name:
            self.encode = CNN(self.embedding.embedding_dim, config.kernel_count, config.kernel_size)
            if 'AP' in config.net_name:
                self.coAttention = CoAttention(config.kernel_count)
        elif 'biLSTM' in config.net_name:
            self.encode = BiLSTM(self.embedding.embedding_dim, config.rnn_hidden)
            if 'AP' in config.net_name:
                self.coAttention = CoAttention(config.rnn_hidden * 2)

    def forward(self, questions, answers):
        q_emb = self.embedding(questions)
        a_emb = self.embedding(answers)
        Q = self.encode(q_emb)
        A = self.encode(a_emb)
        if 'AP' in self.net_name:
            rq, ra = self.coAttention(Q, A)
        else:
            rq = Q.max(dim=-1).values
            ra = A.max(dim=-1).values
            rq = torch.tanh(rq)
            ra = torch.tanh(ra)
        cos = torch.sum(rq * ra, dim=-1) / (rq.norm(dim=-1) * ra.norm(dim=-1))
        return cos
