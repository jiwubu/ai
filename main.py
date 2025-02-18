import time
import torch
from torch import optim, nn
from torch.utils.data import random_split, DataLoader
from model import Transformer
from tqdm import tqdm
from config import Config
from data import Vocab, TextDataset, InputData

def train_epoch(model, dataloader, vocab, criterion, optimizer, device):
    """单个训练epoch"""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        src = batch.to(device)
        tgt = batch.to(device)

        # 源数据掩码（填充位置）
        src_mask = (src != vocab.pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

        #目标数据掩码（填充掩码 + 因果掩码）
        tgt_pad_mask = (tgt != vocab.pad_idx).unsqueeze(1).unsqueeze(2)   #(batch, seq_len)=>(batch, 1, 1, seq_len)
        seq_len = tgt.size(1)
        tgt_causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(device)  # 因果掩码(seq_len, seq_len)
        tgt_mask = tgt_pad_mask & tgt_causal_mask.unsqueeze(0)  # 组合掩码 (batch, 1, seq_len, seq_len)

        #前向传播（使用tgt的前n-1个词预测后n-1个词）
        outputs = model(src, tgt[:, :-1], src_mask=src_mask, tgt_mask=tgt_mask[:, :, :-1, :-1])
        #计算损失
        loss = criterion(outputs.view(-1, len(vocab)), tgt[:, 1:].contiguous().view(-1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """验证/测试函数"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src = batch.to(device)
            tgt = batch.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            outputs = model(src, tgt_input)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)),
                             tgt_output.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    # 初始化配置
    config = Config()

    # 1. 准备示例数据
    print("Loading data...")
    train_texts = InputData.read_data(config.csv_path, max_rows=10000)

    # 2. 构建词汇表
    print("Building vocabulary...")
    vocab = Vocab.build_vocab(train_texts, config.vocab_size, config.min_freq)

    # 3. 创建数据集和数据加载器
    print("Preparing dataset...")
    full_dataset = TextDataset(train_texts, vocab, config.max_seq_len)

    # 划分训练与验证集
    train_size = int(config.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, config.batch_size)

    # 4. 初始化模型
    print("Initializing model...")
    print("------------------------------------------------------")

    model = Transformer(len(vocab), config.d_model, config.n_head,
                        config.num_layers, config.d_ff, config.max_seq_len,
                        config.dropout).to(config.device)

    # 5. 定义优化器和损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # 6. 训练循环
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        start_time = time.time()

        #开始训练
        train_loss = train_epoch(model, train_loader, vocab, criterion, optimizer, config.device)
        val_loss = evaluate(model, val_loader, criterion, config.device)
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.save_path)
            print(f"Saved new best model with val_loss: {val_loss:.4f}")

        # 打印进度
        epoch_mins = (time.time() - start_time) // 60
        epoch_secs = int((time.time() - start_time) % 60)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\t Val. Loss: {val_loss:.4f}")
        print(f"\tLearning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print("------------------------------------------------------")

    # 示例：生成文本
    start_token = vocab["<how>"]
    src = torch.tensor([[start_token]], device=config.device)  # 初始输入

    # 使用不同采样策略生成文本
    print("Greedy decoding:")
    print(model.generate(src, vocab, max_len=100, temperature=1.0, device=config.device))

    return

    print("\nTemperature sampling (t=0.7):")
    print(model.generate(src, vocab, max_len=20, temperature=0.7, device=config.device))

    print("\nTop-k sampling (k=5):")
    print(model.generate(src, vocab, max_len=20, top_k=5, device=config.device))

    print("\nNucleus sampling (p=0.9):")
    print(model.generate(src, vocab, max_len=20, top_p=0.9, device=config.device))

if __name__ == "__main__":
    main()