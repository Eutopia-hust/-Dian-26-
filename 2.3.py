import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads):
        """
        hidden_dim: 每个词的向量维度（比如8）
        num_heads: Q头的数量（比如4）
        num_kv_heads: KV头的数量（比如2）
        """
        super().__init__()
        
        # 检查：hidden_dim必须能被num_heads整除
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        # 检查：num_heads必须能被num_kv_heads整除
        assert num_heads % num_kv_heads == 0, "num_heads必须能被num_kv_heads整除"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        
        # 定义线性层
        # Q: 输入hidden_dim，输出hidden_dim
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        # K: 输入hidden_dim，输出 num_kv_heads * head_dim
        self.W_k = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        # V: 输入hidden_dim，输出 num_kv_heads * head_dim
        self.W_v = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        # 输出投影：输入hidden_dim，输出hidden_dim
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, past_key_values=None):
        """
        x: (batch, seq_len, hidden_dim)
        past_key_values: 可选的KV缓存
        返回: (output, new_key_values)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # 生成Q、K、V 
        # Q: (batch, seq_len, hidden_dim)
        Q = self.W_q(x)
        # K: (batch, seq_len, num_kv_heads * head_dim)
        K = self.W_k(x)
        # V: (batch, seq_len, num_kv_heads * head_dim)
        V = self.W_v(x)
        
        # 拆分成多头
        # Q: (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # K: (batch, seq_len, num_kv_heads, head_dim)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        # V: (batch, seq_len, num_kv_heads, head_dim)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 转置：把头的维度移到前面
        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        # (batch, seq_len, num_kv_heads, head_dim) -> (batch, num_kv_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 处理KV Cache
        if past_key_values is not None:
            K_cache, V_cache = past_key_values
            # K_cache: (batch, num_kv_heads, 历史长度, head_dim)
            # K: (batch, num_kv_heads, 当前长度, head_dim)
            # 拼接后: (batch, num_kv_heads, 历史+当前长度, head_dim)
            K = torch.cat([K_cache, K], dim=2)
            V = torch.cat([V_cache, V], dim=2)
        
        # 更新缓存
        new_key_values = (K, V)
        
        # 重复KV头，让数量匹配Q头
        # 计算每个KV头需要重复几次
        repeat_times = self.num_heads // self.num_kv_heads
        
        # 重复K的头维度
        # K: (batch, num_kv_heads, seq_len, head_dim)
        # 重复后: (batch, num_heads, seq_len, head_dim)
        K = K.repeat_interleave(repeat_times, dim=1)
        V = V.repeat_interleave(repeat_times, dim=1)
        
        #合并batch和heads，并行计算 
        Q = Q.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        K = K.reshape(batch_size * self.num_heads, -1, self.head_dim)
        V = V.reshape(batch_size * self.num_heads, -1, self.head_dim)
        
        #  计算注意力 
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # 恢复形状
        output = output.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        output = output.transpose(1, 2)
        output = output.reshape(batch_size, seq_len, hidden_dim)
        
        # 最后投影 
        output = self.W_o(output)
        
        return output, new_key_values
    # 测试：实例化 + 前向传播
model = GroupedQueryAttention(hidden_dim=8, num_heads=4, num_kv_heads=2)
x = torch.randn(2, 3, 8)  # 随机输入
output, kv = model(x)
print("输出形状:", output.shape)
