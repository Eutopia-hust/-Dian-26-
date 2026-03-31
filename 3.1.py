import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet 的核心实现
    
    参数：
        d_model: 输入输出的维度
        d_k: 键/查询的维度
        d_v: 值的维度
        beta1: 遗忘强度，默认 1.0
        beta2: 学习率/写入强度，默认 1.0
    """
    def __init__(self, d_model, d_k=None, d_v=None, beta1=1.0, beta2=1.0):
        super().__init__()
        
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model
        self.d_model = d_model
        
        # 可学习的参数
        self.beta1 = beta1  
        self.beta2 = beta2
        
        # 三个线性层：生成 q, k, v
        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_v, bias=False)
        
        # 输出投影层：把输出再投影回 d_model
        self.W_o = nn.Linear(self.d_v, d_model, bias=False)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        返回: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 步骤1：生成 q, k, v（对所有时间步一次性生成）
        # 形状: (batch_size, seq_len, d_k) 或 (batch_size, seq_len, d_v)
        q = self.W_q(x)  # (B, L, d_k)
        k = self.W_k(x)  # (B, L, d_k)
        v = self.W_v(x)  # (B, L, d_v)
        
        # 步骤2：初始化隐状态 S
        # S 的形状: (batch_size, d_v, d_k)
        S = torch.zeros(batch_size, self.d_v, self.d_k, device=x.device)
        
        # 步骤3：存储每个时间步的输出
        outputs = []
        
        # 步骤4：循环遍历每个时间步
        for t in range(seq_len):
            # 取出当前时间步的 q_t, k_t, v_t
            # 形状: (batch_size, d_k) 或 (batch_size, d_v)
            q_t = q[:, t, :]   # (B, d_k)
            k_t = k[:, t, :]   # (B, d_k)
            v_t = v[:, t, :]   # (B, d_v)
            
            # 关键操作1：遗忘矩阵 F_t = I - beta1 * k_t @ k_t^T
            # 先计算 k_t @ k_t^T（外积）
            # k_t: (B, d_k) -> 需要变成 (B, d_k, 1) 和 (B, 1, d_k)
            k_t_unsqueezed = k_t.unsqueeze(-1)  # (B, d_k, 1)
            k_t_T = k_t.unsqueeze(-2)           # (B, 1, d_k)
            outer_k = k_t_unsqueezed @ k_t_T    # (B, d_k, d_k)
            
            # 构建遗忘矩阵
            I = torch.eye(self.d_k, device=x.device)  # (d_k, d_k)
            # 注意：I 需要扩展到 batch 维度
            F_t = I - self.beta1 * outer_k  # (B, d_k, d_k)
            
            # 关键操作2：更新 S
            # S = S @ F_t + beta2 * v_t @ k_t^T
            # S 当前是 (B, d_v, d_k)
            # S @ F_t: (B, d_v, d_k) @ (B, d_k, d_k) = (B, d_v, d_k)
            S = S @ F_t
            
            # 写入新信息：v_t @ k_t^T
            # v_t: (B, d_v) -> (B, d_v, 1)
            # k_t^T: (B, 1, d_k)
            v_t_unsqueezed = v_t.unsqueeze(-1)  # (B, d_v, 1)
            k_t_T = k_t.unsqueeze(-2)           # (B, 1, d_k)
            outer_vk = v_t_unsqueezed @ k_t_T   # (B, d_v, d_k)
            
            S = S + self.beta2 * outer_vk
            
            # 关键操作3：计算输出 o_t = S @ q_t
            # S: (B, d_v, d_k)
            # q_t: (B, d_k) -> (B, d_k, 1)
            q_t_unsqueezed = q_t.unsqueeze(-1)  # (B, d_k, 1)
            o_t = S @ q_t_unsqueezed            # (B, d_v, 1)
            o_t = o_t.squeeze(-1)               # (B, d_v)
            
            outputs.append(o_t)
        
        # 步骤5：把 outputs 堆叠成序列
        # outputs 是 list of (B, d_v)，长度 seq_len
        # 堆叠后: (seq_len, B, d_v) -> 转置为 (B, seq_len, d_v)
        outputs = torch.stack(outputs, dim=0)   # (L, B, d_v)
        outputs = outputs.transpose(0, 1)       # (B, L, d_v)
        
        # 步骤6：投影回 d_model
        outputs = self.W_o(outputs)  # (B, L, d_model)
        
        return outputs
    # 测试代码
if __name__ == "__main__":
    # 初始化模型
    model = GatedDeltaNet(d_model=16)
    # 构造输入：batch=2, 序列长度=5, 维度=16
    x = torch.randn(2, 5, 16)
    # 前向传播
    out = model(x)
    print("输出形状:", out.shape)
