import torch
import torch.nn as nn
import torch.nn.functional as F

# 第一部分：带KV Cache的多头注意力模型类

class StandardMHAWithCache(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, past_key_values=None):
        batch_size, seq_len, hidden_dim = x.shape
        
        # 生成Q、K、V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 拆分成多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置：把头的维度移到前面
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # KV Cache处理
        if past_key_values is not None:
            K_cache, V_cache = past_key_values
            # 把历史的K和当前的K拼接起来
            K = torch.cat([K_cache, K], dim=2)
            V = torch.cat([V_cache, V], dim=2) 
        
        # 更新缓存
        new_key_values = (K, V)
        
        # 合并batch和heads，并行计算
        Q = Q.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        K = K.reshape(batch_size * self.num_heads, -1, self.head_dim)
        V = V.reshape(batch_size * self.num_heads, -1, self.head_dim)
        
        # 计算注意力
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


# 第二部分：模拟自回归生成（测试代码）
def simulate_autoregressive_generation():
    """模拟自回归生成过程，演示KV Cache的形状变化"""
    
    # 参数设置
    batch_size = 1
    hidden_dim = 8
    num_heads = 4
    initial_len = 10   # 初始序列长度
    generate_steps = 5  # 要生成的新token数量
    
    # 创建模型
    model = StandardMHAWithCache(hidden_dim, num_heads)
    
    print("=" * 50)
    print("【步骤0】初始输入（前10个token）")
    print("=" * 50)
    
    # 初始输入：10个token
    x_initial = torch.randn(batch_size, initial_len, hidden_dim)
    print(f"初始输入形状: {x_initial.shape}")  # (1, 10, 8)
    
    # 第一次前向传播（没有缓存）
    output, cache = model(x_initial, past_key_values=None)
    print(f"初始输出形状: {output.shape}")  # (1, 10, 8)
    
    # 查看缓存的形状
    K_cache, V_cache = cache
    print(f"初始K缓存形状: {K_cache.shape}")  # (1, 4, 10, 2)
    print(f"初始V缓存形状: {V_cache.shape}")  # (1, 4, 10, 2)
    print(f"→ 缓存了10个token的K和V")
    
    print("\n" + "=" * 50)
    print("开始循环生成新token")
    print("=" * 50)
    
    # 循环生成新token
    for step in range(1, generate_steps + 1):
        print(f"\n【步骤{step}】生成第{initial_len + step}个token")
        
        # 模拟生成1个新token
        x_new = torch.randn(batch_size, 1, hidden_dim)
        print(f"新token输入形状: {x_new.shape}")  # (1, 1, 8)
        
        # 用缓存进行前向传播
        output, cache = model(x_new, past_key_values=cache)
        
        # 查看输出形状
        print(f"输出形状: {output.shape}")  # (1, 1, 8)
        
        # 查看缓存的形状
        K_cache, V_cache = cache
        current_len = initial_len + step
        print(f"K缓存形状: {K_cache.shape}")  # (1, 4, {current_len}, 2)
        print(f"V缓存形状: {V_cache.shape}")  # (1, 4, {current_len}, 2)
        print(f"→ 缓存长度从{initial_len + step - 1}增长到{current_len}")
    
    print("\n" + "=" * 50)
    print("模拟完成！")
    print("=" * 50)
    print("\n观察结论：")
    print("1. 初始缓存长度 = 初始序列长度 (10)")
    print("2. 每生成一个新token，缓存长度 +1")
    print("3. Q的长度始终是1（只输入新token）")
    print("4. K和V的长度 = 历史长度 + 当前长度，在增长")



# 第三部分：运行
if __name__ == "__main__":
    simulate_autoregressive_generation()