import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardMHA(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        
        # 检查hidden_dim能否被num_heads整除
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 定义4个线性层
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape
        
        # 生成Q、K、V 
        Q = self.W_q(x)  # (batch, seq_len, hidden_dim)
        K = self.W_k(x)  # (batch, seq_len, hidden_dim)
        V = self.W_v(x)  # (batch, seq_len, hidden_dim)
        
        # 拆分成多个头
        # view: 把hidden_dim拆成 num_heads × head_dim
        # (batch, seq_len, hidden_dim) → (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # transpose: 把头的维度移到前面
        # (batch, seq_len, num_heads, head_dim) → (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 合并batch和num_heads，让每个头独立计算
        # reshape: 把前两个维度合并
        # (batch, num_heads, seq_len, head_dim) → (batch * num_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        K = K.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        V = V.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        
        # 计算注意力
        # 计算Q和K的相似度分数
        # Q: (batch*num_heads, seq_len, head_dim)
        # K.transpose(-2, -1): (batch*num_heads, head_dim, seq_len)
        # 矩阵乘法后: (batch*num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # 除以√head_dim，防止数值过大导致梯度消失
        scores = scores / (self.head_dim ** 0.5)
        
        # Softmax，把分数转成概率
        attention = F.softmax(scores, dim=-1)
        
        # 用注意力权重加权求和V
        # attention: (batch*num_heads, seq_len, seq_len)
        # V: (batch*num_heads, seq_len, head_dim)
        # 结果: (batch*num_heads, seq_len, head_dim)
        output = torch.matmul(attention, V)
        
        # 恢复形状 
        # 先恢复成 (batch, num_heads, seq_len, head_dim)
        output = output.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # 把头的维度换回后面 (batch, seq_len, num_heads, head_dim)
        output = output.transpose(1, 2)
        
        # 拼接多头，恢复成 (batch, seq_len, hidden_dim)
        output = output.reshape(batch_size, seq_len, hidden_dim)
        
        # 最后投影
        # 用线性层混合多头信息
        output = self.W_o(output)
        
        return output


# 测试代码 
if __name__ == "__main__":
    # 设置参数
    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    num_heads = 4
    
    # 随机生成输入
    x = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"输入形状: {x.shape}")  # (2, 4, 8)
    
    # 创建模型
    model = StandardMHA(hidden_dim, num_heads)
    
    # 前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")  # (2, 4, 8)
    
    # 验证形状一致
    assert output.shape == x.shape, "输出形状和输入不一致"
    print("✓测试通过,输出形状和输入一致")