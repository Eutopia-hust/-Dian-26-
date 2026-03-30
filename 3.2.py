import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ==================== Gated DeltaNet ====================

class GatedDeltaNet(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None, beta1=1.0, beta2=1.0):
        super().__init__()
        
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model
        self.d_model = d_model
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_v, bias=False)
        self.W_o = nn.Linear(self.d_v, d_model, bias=False)
        
        self.scale = self.d_k ** -0.5
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        q = q * self.scale
        k = F.normalize(k, dim=-1)  # 归一化，防止 nan
        
        S = torch.zeros(batch_size, self.d_v, self.d_k, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            q_t = q[:, t, :]
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            
            k_t_col = k_t.unsqueeze(-1)
            k_t_row = k_t.unsqueeze(-2)
            outer_kk = k_t_col @ k_t_row
            
            I = torch.eye(self.d_k, device=x.device)
            F_t = I - self.beta1 * outer_kk
            
            S = S @ F_t
            
            v_t_col = v_t.unsqueeze(-1)
            outer_vk = v_t_col @ k_t_row
            S = S + self.beta2 * outer_vk
            
            q_t_col = q_t.unsqueeze(-1)
            o_t = (S @ q_t_col).squeeze(-1)
            outputs.append(o_t)
        
        outputs = torch.stack(outputs, dim=0).transpose(0, 1)
        outputs = self.W_o(outputs)
        
        return outputs


# ==================== MLP ====================

class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else d_model * 4
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x):
        identity = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + identity


# ==================== GDN Block ====================

class GDNBlock(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None, mlp_hidden=None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.gdn = GatedDeltaNet(d_model, d_k, d_v)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_hidden, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.gdn(x)
        x = self.dropout(x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        return x + residual


# ====================分类器===================

class FashionGDNClassifier(nn.Module):
    def __init__(self, image_size=28, d_model=128, num_blocks=4, num_classes=10, dropout=0.1):
        super().__init__()
        
        self.seq_len = image_size * image_size 
        
        self.input_proj = nn.Linear(1, d_model)  
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)
        
        # GDN Blocks
        self.blocks = nn.ModuleList([
            GDNBlock(d_model, d_k=d_model//2, d_v=d_model//2, dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)  
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # 展平: (B, 1, 28, 28) -> (B, 784, 1)
        x = x.view(B, self.seq_len, 1)
        
        # 投影
        x = self.input_proj(x)
        
        # 加位置编码
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # GDN Blocks
        for block in self.blocks:
            x = block(x)
        
        # 归一化
        x = self.norm(x)
        
        # 全局平均池化: (B, 784, 128) -> (B, 128)
        x = x.transpose(1, 2)  # (B, 128, 784)
        x = self.global_pool(x)  # (B, 128, 1)
        x = x.squeeze(-1)  # (B, 128)
        
        # 分类
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
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = output.max(1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, pred = output.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


# ==================== 主程序 ====================

def main():
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 超参数
    batch_size = 64
    epochs = 10
    lr = 0.0005
    d_model = 128
    num_blocks = 4
    dropout = 0.1
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading Fashion-MNIST...")
    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # 创建模型
    model = FashionGDNClassifier(
        image_size=28,
        d_model=d_model,
        num_blocks=num_blocks,
        dropout=dropout
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
    
    print(f'\nTraining complete!')


if __name__ == '__main__':
    main()
