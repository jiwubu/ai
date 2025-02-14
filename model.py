import torch
import torch.nn as nn
import numpy as np

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length= 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_length, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x: Tensor, shape [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

# 多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # 线性变换并分头
        q = self.q_linear(q).view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # 注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 加权求和
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(output)
        return output

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 前馈网络子层
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    #tgt是解码器层的输入，memory是编码层的输出
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 自注意力子层
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # 交叉注意力子层
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # 前馈网络子层
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# 编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, num_layers, d_ff, max_seq_len, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 嵌入层 + 位置编码
        src = self.embedding(src)
        src = self.positional_encoding(src)
        src = self.dropout(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

# 解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, num_layers, d_ff, max_seq_len, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    #memory对应编码器侧encoder_output
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 嵌入层 + 位置编码
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return tgt

# Transformer模型
class Transformer(nn.Module):
    #vocab_size:词汇表大小 d_model:词嵌入纬度 n_head:多头数量 num_layers:编码与解码层数量
    #d_ff:前馈神经网络隐藏层 max_seq_len：一句话里面词的最大长度
    def __init__(self, vocab_size, d_model, n_head, num_layers, d_ff, max_seq_len, dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, n_head, num_layers, d_ff, max_seq_len, dropout)
        self.decoder = Decoder(vocab_size, d_model, n_head, num_layers, d_ff, max_seq_len, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask)
        output = self.fc_out(output)
        return output