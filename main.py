import time

from torch import optim, nn
from torch.utils.data import random_split, DataLoader
from config import Config
from data import Vocab, TextDataset
from model import Transformer
import torch
from tqdm import tqdm
import csv

def train_epoch(model, dataloader, criterion, optimizer, device):
    """单个训练epoch"""
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        src = batch.to(device)
        tgt = batch.to(device)

        # 准备decoder输入输出
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        outputs = model(src, tgt_input)

        # 计算损失
        loss = criterion(outputs.reshape(-1, outputs.size(-1)),
                         tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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


def generate_example(model, vocab, device, max_length=50):
    """生成示例文本"""
    model.eval()
    start_idx = vocab["performance"]
    print("start_idx=", start_idx)
    generated = torch.tensor([[start_idx]], device=device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated, generated)
            logits = outputs[:, -1, :] / Config.temperature
            top_k = torch.topk(logits, Config.top_k)
            probs = torch.softmax(top_k.values, dim=-1)
            next_token = top_k.indices[0, torch.multinomial(probs, 1).item()]

            if next_token == vocab["<end>"]:
                break

            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    indices = generated[0].cpu().numpy()
    tokens = vocab.lookup_tokens(indices)
    filtered = [t for t in tokens if t not in ["<pad>", "<start>", "<end>"]]
    return " ".join(filtered)

def parse_csv(file_path, max_rows=100):
    data = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            for i, row in enumerate(csv_reader):
                if i >= max_rows:  # 如果达到最大行数，停止读取
                    break
                if i >= 1:
                    data.append(row[1])

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
    except Exception as e:
        print(f"解析CSV文件时发生错误: {e}")
    return data

def main():
    # 初始化配置
    config = Config()

    # 加载数据
    print("Loading data...")
    texts = parse_csv(config.csv_path, max_rows=10)

    # 构建词汇表
    print("Building vocabulary...")
    vocab = Vocab.build_vocab(texts, config.vocab_size, config.min_freq)

    for idx in tqdm(range(len(vocab))):
        print(idx, vocab[idx])

    print("vocab[performance]", vocab["performance"], vocab.lookup_indices(["performance"]))

    # 准备数据集
    print("Preparing dataset...")
    full_dataset = TextDataset(texts, vocab, config.max_seq_len)

    # 划分训练验证集
    train_size = int(config.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, config.batch_size)

    # 初始化模型
    print("Initializing model...")
    print("------------------------------------------------------")

    model = Transformer(len(vocab), config.d_model, config.n_head,
                        config.num_layers, config.d_ff, config.max_seq_len,
                        config.dropout).to(config.device)

    # 设置训练组件
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        start_time = time.time()

        # 训练阶段
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)

        # 验证阶段
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

    # 生成示例文本
    example = generate_example(model, vocab, config.device)
    print("\nGenerated Example:")
    print(example + "\n")

    print("------------------------------------------------------")

    # 最终测试
    print("Training complete. Loading best model for final test...")
    model.load_state_dict(torch.load(config.save_path))
    final_loss = evaluate(model, val_loader, criterion, config.device)
    print(f"Final Validation Loss: {final_loss:.4f}")

if __name__ == "__main__":
    main()