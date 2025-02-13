import torch

class Config:
    """全局配置参数"""
    csv_path = "wmt_zh_en_training_corpus.csv"
    text_column = "text"
    train_ratio = 0.9
    max_seq_len = 100
    min_freq = 1

    # 模型参数
    vocab_size = 10000  # 词汇表大小
    d_model = 512      # 嵌入维度
    n_head = 8 # 多头注意力头数
    num_layers = 6 # 编码器和解码器层数
    d_ff = 2048 # 前馈网络隐藏层维度
    dropout = 0.1 # Dropout概率

    # 训练参数
    batch_size = 64  # 批量大小
    lr = 1e-4 # 学习率
    epochs = 1 # 训练轮数
    save_path = "best_model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 生成参数
    start_token = "<start>"
    temperature = 0.7
    top_k = 50