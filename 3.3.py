import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==================== GPU并行多头注意力====================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, L, D = x.shape
        # 线性投影+分头，GPU并行
        q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算，完全GPU并行，无任何Python循环
        attn = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)

# ==================== MLP ====================
class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 4
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x):
        identity = x
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.dropout(self.fc2(x))
        return x + identity

# ==================== Transformer Block====================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_hidden=None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_hidden, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(self.attn(self.norm1(x))) + x
        x = self.dropout(self.mlp(self.norm2(x))) + x
        return x

# ==================== 分类器====================
class FashionTransformerClassifier(nn.Module):
    def __init__(self, image_size=28, d_model=128, num_heads=4, num_blocks=4, num_classes=10, dropout=0.1):
        super().__init__()
        self.seq_len = image_size * image_size
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.seq_len, 1)
        x = self.input_proj(x)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        return self.head(x)

# ==================== 训练函数 ====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = output.max(1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(loader), 100. * correct / total

# ==================== 评估函数 ====================
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            _, pred = output.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

# ==================== 主程序 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    batch_size = 64
    epochs = 10
    lr = 0.0005
    d_model = 128
    num_heads = 4
    num_blocks = 4
    dropout = 0.1
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading Fashion-MNIST...")
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    model = FashionTransformerClassifier(
        d_model=d_model, num_heads=num_heads, num_blocks=num_blocks, dropout=dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 40)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    print('\nTraining complete!')

if __name__ == '__main__':
    main()
