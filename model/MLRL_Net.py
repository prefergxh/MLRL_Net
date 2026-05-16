import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GlobalTemporalAttention(nn.Module):
    def __init__(self, channels, seq_len, reduction_ratio=4):
        """
        全局时间注意力机制 (GTA)
        
        参数:
            channels (int): 输入特征的通道数 (对于原始IQ信号，通常为2)
            seq_len (int): 序列长度 (例如 1024 或 2048)
            reduction_ratio (int): 隐藏层降维系数，用于控制参数量
        """
        super(GlobalTemporalAttention, self).__init__()
        
        self.channels = channels
        self.seq_len = seq_len
        hidden_dim = max(1, channels // reduction_ratio)
        
        # 对应公式(2)和(4)中的内容特征投影矩阵 W1 和 W2
        self.fc1 = nn.Linear(channels, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # 对应公式(4)中的位置嵌入矩阵 P
        # 这是一个可学习的参数，形状为 (1, seq_len, 1)，利用广播机制应用于整个Batch
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, 1))
        nn.init.normal_(self.pos_embedding, std=0.02) # 初始化位置先验

    def forward(self, x):
        """
        前向传播
        
        输入:
            x: 形状为 (Batch, Channels, Seq_Len) 的张量
        输出:
            out: 经过注意力加权后的张量，形状与输入相同
            attn_weights: 归一化后的注意力权重，可用于可视化分析
        """
        # 1. 维度转换：将通道维度移到最后以进行线性变换 (B, C, L) -> (B, L, C)
        # 这对应公式中转置输入 F 的操作
        x_trans = x.transpose(1, 2)
        
        # 2. 计算基于内容的初始得分 (Content Evidence)
        # 经过 W1 -> ReLU -> W2
        # 输出形状: (B, L, 1)
        content_score = self.fc2(self.relu(self.fc1(x_trans)))
        
        # 3. 注入位置先验 (Positional Prior)
        # 将学习到的位置偏置 P 加到 pre-Softmax 得分上 (对应公式4的逻辑)
        # 输出形状: (B, L, 1)
        raw_score = content_score + self.pos_embedding
        
        # 4. 竞争性归一化
        # 在时间序列长度维度 (dim=1) 上应用 Softmax，得到最终的注意力权重 A
        # 形状: (B, L, 1)
        attn_weights = F.softmax(raw_score, dim=1)
        
        # 5. 特征加权 (对应公式3: O = F * diag(A))
        # 将注意力权重维度调整为 (B, 1, L) 以便与原始输入 x (B, C, L) 逐元素相乘
        attn_weights_trans = attn_weights.transpose(1, 2)
        out = x * attn_weights_trans
        
        return out

class ResidualGTABlock(nn.Module):
    def __init__(self, in_channels, out_channels,seq_len):
        super(ResidualGTABlock, self).__init__()
        
        # 1. 前置的一维卷积，用于特征平滑和跨通道信息交互
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 2. 全局时间注意力层 (此处复用之前写的 GTA 类)
        self.gta = GlobalTemporalAttention(out_channels, seq_len)
        self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels)
        )
        # 3. 残差融合后的归一化 (Post-Norm 结构)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 4. 引入自适应残差权重 (初始值设为0，让网络初始阶段表现得像恒等映射，便于稳定训练)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 记录残差分支 (Identity Mapping)
        identity = self.shortcut(x)
        
        # 局部特征提取
        out = self.relu(self.bn1(self.conv(x)))
        
        # 全局时间注意力加权
        gta_out = self.gta(out)
        
        # 自适应残差融合
        out = identity + self.alpha * gta_out
        
        # 最后的归一化
        return self.bn2(out)
    
class DeepGTA_Encoder(nn.Module):
    def __init__(self, in_channels, d_model,out_channels, seq_len):
        super().__init__()
        # 初始特征映射
        self.stem = nn.Conv1d(in_channels, d_model, kernel_size=1)
        
        # 堆叠多个带有残差的 GTA 模块
        self.layers = nn.ModuleList([
            ResidualGTABlock(d_model,32, seq_len),
            ResidualGTABlock(32,64, seq_len),
            ResidualGTABlock(64,128,seq_len)
        ])

        self.transition_layer = nn.Sequential(
        # 可以选择 kernel_size=1 (纯通道映射) 或 kernel_size=3, padding=1 (兼顾局部平滑)
        nn.Conv1d(128, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)

        x = self.transition_layer(x)
        return x
    

#可学习的位置编码部分
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        # 初始化一个可学习的张量，形状为 [1, L, D]
        # 使用 1 作为 Batch 维度，利用广播机制加到所有样本上
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        # 截断正态分布初始化，防止初始值过大破坏 CNN 提取的特征
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x 形状必须是 [B, L, d_model]
        trans = x.transpose(1,2)
        return trans + self.pos_embed

# 基于内容的自注意力感知矩阵
class BaseContentAttention(nn.Module):
    def __init__(self, d_model):
        """
        参数:
            d_model: 输入特征的维度 D (例如 128)
        """
        super(BaseContentAttention, self).__init__()
        self.d_model = d_model
        
        # 1. 定义三个独立的线性映射层，用于生成 Query, Key, Value
        # 即使不改变维度 (d_model -> d_model)，这一步也是必须的，
        # 因为它赋予了模型学习到不同特征表达空间的能力。
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 2. 缩放因子：根号下 d_model
        # 作用：防止 Q 和 K 点积后的数值过大，导致 Softmax 梯度消失
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        """
        输入:
            x: 形状为 [B, L, D] 的张量 (Batch_Size, Seq_Len, d_model)
        输出:
            A1: 基础内容感知注意力矩阵，形状为 [B, L, L]
            V: 映射后的 Value 矩阵，保留用于后续特征加权，形状为 [B, L, D]
        """

        # B: 批次大小, L: 序列长度, D: 特征维度
        B, L, D = x.shape
        
        # 第一步：计算 Q, K, V
        # 经过线性层后，形状依然都是 [B, L, D]
        Q = self.W_q(x) 
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 第二步：计算点积打分 (Q * K^T)
        # K.transpose(1, 2) 将 K 的形状从 [B, L, D] 翻转为 [B, D, L]
        # torch.matmul 发生矩阵乘法: [B, L, D] @ [B, D, L] -> 结果形状为 [B, L, L]
        # 结果矩阵的第 i 行第 j 列，代表了第 i 个时间步对第 j 个时间步的原始关注度
        raw_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        
        # 第三步：Softmax 归一化
        # dim=-1 表示在最后一个维度（也就是每一行内部）进行 Softmax
        # 确保每个时间点对其他所有时间点的注意力权重之和为 1
        A1 = torch.softmax(raw_scores, dim=-1)
        
        # 注意：在纯正的自注意力中，下一步应该是 out = torch.matmul(A1, V)。
        # 但因为我们正在构建 PGESA，A1 必须先去和伪高斯矩阵 A2 进行融合。
        # 所以我们这里直接返回 A1 和 V，把融合的权力交给外层模块。
        return A1, V
    
class PseudoGaussianTemporalAttention(nn.Module):
    def __init__(self, d_model):
        """
        参数:
            d_model: 输入特征的维度 D (例如 128)
        """
        super(PseudoGaussianTemporalAttention, self).__init__()
        
        mid_channels = d_model // 4
        # 对应公式 (8): W' 和 b
        # 输入维度 d_model，输出维度 1 (因为每个时间步只需要一个标量 sigma)
        self.sigma_net = nn.Sequential(
            nn.Linear(d_model,mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels,1)
        )

    def forward(self, x):
        """
        输入:
            x: 形状为 [B, L, D] 的张量 (Batch_Size, Seq_Len, d_model)
        输出:
            A2: 伪高斯时间注意力矩阵，形状为 [B, L, L]
        """
        B, L, D = x.shape
        device = x.device
        
        # ---------------------------------------------------------
        # 第一步：计算动态标准差 sigma_i (公式 8)
        # ---------------------------------------------------------
        # self.sigma_net(x) 会逐个时间步进行映射，输出形状 [B, L, 1]
        # 使用 torch.abs 保证 sigma 为正数
        # 加上 1e-6 是深度学习里的基操，防止除以 0 导致梯度爆炸 (NaN)
        sigma = torch.abs(self.sigma_net(x)) + 1e-6
        
        # 计算 sigma_i 的平方: 2 * sigma_i^2
        # 形状依然是 [B, L, 1]
        variance = 2 * (sigma ** 2)
        
        # ---------------------------------------------------------
        # 第二步：计算时间步的相对距离平方矩阵 (i - j)^2
        # ---------------------------------------------------------
        # 创建一个时间步索引序列: [0, 1, 2, ..., L-1]
        grid = torch.arange(L, dtype=torch.float32, device=device)
        
        # 利用广播机制计算距离矩阵
        # grid.unsqueeze(1) 形状: [L, 1] -> 列向量
        # grid.unsqueeze(0) 形状: [1, L] -> 行向量
        # 相减后平方，得到形状为 [L, L] 的对称矩阵
        dist_sq = (grid.unsqueeze(1) - grid.unsqueeze(0)) ** 2
        
        # 为了能和 batch 数据计算，增加 batch 维度: [1, L, L]
        dist_sq = dist_sq.unsqueeze(0)
        
        # ---------------------------------------------------------
        # 第三步：生成伪高斯矩阵 A2 (公式 7)
        # ---------------------------------------------------------
        # 这里的除法利用了 PyTorch 的广播机制 (Broadcasting 魔术)！
        # dist_sq 形状是 [1, L, L]
        # variance 形状是 [B, L, 1]
        # PyTorch 会自动将 variance 的第 i 行标量，应用到 dist_sq 的第 i 行上！
        # 结果形状为 [B, L, L]
        A2 = torch.exp(-dist_sq / variance)
        
        return A2

class PGESA_Encoder(nn.Module):
    def __init__(self,d_model,out_channels):
        super().__init__()
        self.contentAttention = BaseContentAttention(d_model)
        self.pseudoAttention = PseudoGaussianTemporalAttention(d_model)
        # 2.1 Attention Normalization (Transformer 标准采用 LayerNorm)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 2.2 Global Average Pooling (GAP)
        # 将 L 个时间步的高维特征，浓缩成 1 个终极指纹向量
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 2.3 全连接层最终输出
        self.fc = nn.Linear(d_model,out_channels)

    def forward(self,x):
        identity = x
        A1,V = self.contentAttention(x)
        A2 = self.pseudoAttention(x)
        A_fused = (A1 + A2) / 2.0
        A_final = A_fused / (A_fused.sum(dim=-1, keepdim=True) + 1e-9)
        pgesa_out = torch.matmul(A_final, V)
        norm_out = self.layer_norm(pgesa_out)
        out = norm_out + identity
        out_transposed = out.transpose(1,2)
        gap_out = self.gap(out_transposed).squeeze(-1)
        final_out = self.fc(gap_out)
        return final_out
    

class Encoder(nn.Module):
    def __init__(self,in_channles,mid_channels,d_model,seq_len,out_channels):
        super().__init__()
        self.gta_block = DeepGTA_Encoder(in_channles,d_model=mid_channels,out_channels=d_model,seq_len=seq_len)
        self.position_encoder = LearnablePositionalEncoding(seq_len,d_model)
        self.pgesa_block = PGESA_Encoder(d_model,out_channels)

    def forward(self,x):
        out = self.gta_block(x)
        out = self.position_encoder(out)
        out = self.pgesa_block(out)
        return out


class ClosedSetClassifier(nn.Module):
    def __init__(self, in_features=128, hidden_features=64, num_classes=10):
        """
        闭集分类器 (Closed-Set Classifier)
        参数:
            in_features: 输入特征维度 (即上一层 PGESA GAP 后的 d_model, 例如 128)
            hidden_features: 隐藏层维度 (通常是 in_features 的一半或相等，用于特征过渡)
            num_classes: 闭集分类的目标类别数 (你的辐射源设备数量)
        """
        super(ClosedSetClassifier, self).__init__()
        
        # 对应图中的第一个 "FC + ReLU" 模块
        self.block1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features), # 强烈建议加上 BN 层，加速收敛并防止特征偏移
            nn.ReLU(inplace=True)
        )
        
        # 对应图中的第二个 "FC" 模块 (注意：去掉了原图中的第二个 ReLU)
        # 这里将高维特征直接映射到类别数量的维度上
        self.block2 = nn.Linear(hidden_features, num_classes)
        

    def forward(self, x):
        """
        前向传播
        参数:
            x: 形状为 [Batch_Size, in_features] 的特征向量
            return_probs: 布尔值。为 True 时输出概率 (Softmax)；为 False 时输出原始 Logits。
        """
        # 第一层：特征变换与非线性激活
        x = self.block1(x)
        
        # 第二层：输出类别打分 (Logits)
        logits = self.block2(x)

        return logits
    
class  Closed_Train(nn.Module):
    def __init__(self,in_channles,mid_channels,d_model,seq_len,out_channels,hidden_features,num_classes):
        super().__init__()
        self.encoder = Encoder(in_channles,mid_channels,d_model,seq_len,out_channels)
        self.classfier = ClosedSetClassifier(out_channels,hidden_features,num_classes)

    def forward(self,x):
        x = x.transpose(1,2)
        features = self.encoder(x)
        logits = self.classfier(features)
        return logits
    

class FiLMLayer_1D(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        """
        专门处理 [B, L] 维度的 FiLM 层
        参数:
            feature_dim: 输入特征的长度 L (例如 128)
            condition_dim: 条件标签向量的维度
        """
        super(FiLMLayer_1D, self).__init__()
        
        # 将条件向量映射到特征的长度 L 上，生成缩放项 gamma
        self.H_gamma = nn.Linear(condition_dim, feature_dim)
        
        # 将条件向量映射到特征的长度 L 上，生成偏移项 beta
        self.H_beta = nn.Linear(condition_dim, feature_dim)

    def forward(self, features, condition):
        """
        features: 形状为 [Batch, L] 的特征向量
        condition: 形状为 [Batch, condition_dim] 的标签特征向量
        """
        # 生成 gamma 和 beta，形状均为 [Batch, L]
        gamma = self.H_gamma(condition)
        beta = self.H_beta(condition)
        
        # 【核心区别】：因为 features 也是 [Batch, L]
        # 形状完全对齐，直接进行哈达玛乘积（逐元素相乘）即可！
        # 不需要像处理 [B, C, L] 那样在最后 unsqueeze 增加时间维度
        modulated_features = gamma * features + beta
        
        return modulated_features
    

class FiLMLayer_2D(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        """
        处理 [B, 64, L] 形状特征的 FiLM 层
        参数:
            feature_dim: 特征通道数，这里固定为 64
            condition_dim: 你的标签条件向量维度 (假设为 128)
        """
        super(FiLMLayer_2D, self).__init__()
        
        # 定义两个全连接层，专门用来输出 64 个参数
        # 1. 输出 64 个缩放因子 (Gamma)
        self.H_gamma = nn.Linear(condition_dim, feature_dim)
        # 2. 输出 64 个偏移因子 (Beta)
        self.H_beta = nn.Linear(condition_dim, feature_dim)

    def forward(self, features, condition):
        """
        features 形状:  [Batch, 64, 256] (64通道的时序信号)
        condition 形状: [Batch, 128]     (身份标签的嵌入向量)
        """
        # 第一步：根据条件标签，生成针对这 64 个通道的控制参数
        # gamma 形状: [Batch, 64]
        # beta  形状: [Batch, 64]
        gamma = self.H_gamma(condition)
        beta = self.H_beta(condition)
        
        # 第二步：形状对齐 (核心广播机制)
        # 现在的 gamma 是 [B, 64]，但 features 是 [B, 64, 256]
        # 我们需要在最后增加一个维度，变成 [B, 64, 1]
        gamma = gamma.unsqueeze(-1) 
        beta = beta.unsqueeze(-1)   
        
        # 第三步：施加调制魔法
        # [B, 64, 1] * [B, 64, 256] + [B, 64, 1]
        # PyTorch 会自动把 gamma 和 beta 在 256 的时间维度上复制铺开！
        modulated_features = gamma * features + beta
        
        return modulated_features
    

class Decoder(nn.Module):
    def __init__(self, d_model=128, condition_dim=1, original_length=256, out_channels=2):
        super(Decoder, self).__init__()
        
        self.initial_length = original_length // 4  # 64
        self.hidden_channels = 64
        
        # 全链路映射第一层
        self.film_1d = FiLMLayer_1D(d_model,condition_dim)



        # 1. 种子映射层
        self.fc = nn.Linear(d_model, self.hidden_channels * self.initial_length)
        

        # ==========================================
        # 拆解反卷积流水线
        # ==========================================
        # 2. 第一级上采样 (包含 BN 和 ReLU，提供非线性)
        self.up_block_1 = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        
        # 3. 【核心插入点】：中间层 FiLM 调制模块
        # 注意 feature_dim 必须等于 up_block_1 的输出通道数 32
        self.film_2d_first = FiLMLayer_2D(feature_dim=32, condition_dim=condition_dim)
        
        # 4. 第二级上采样
        self.up_block_2 = nn.Sequential(
            nn.ConvTranspose1d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.film_2d_second = FiLMLayer_2D(feature_dim=out_channels,condition_dim=condition_dim)

    def forward(self, x, condition):
        """
        x: [Batch, d_model] - 你的浓缩特征
        condition: [Batch, condition_dim] - 你的身份标签向量
        """
        B = x.shape[0]
        
        # 第一步：先进行1DFiLM映射
        x = self.film_1d(x)  #形状不变。[B,L]
        x = self.fc(x)
        x = x.view(B, self.hidden_channels, self.initial_length)
        
        # 第二步：第一级上采样 [B, 64, 256] -> [B, 32, 512]
        feat_up1 = self.up_block_1(x)
        
        # 第三步：【执行条件调制】
        # 让身份标签在这里对 32 个通道进行缩放和偏移
        feat_modulated = self.film_2d_firstl(feat_up1, condition)
        
        # 第四步：第二级上采样 [B, 32, 512] -> [B, 2, 1024]
        reconstructed_signal = self.up_block_2(feat_modulated)

        # 保留一步看效果怎么样
        # reconstructed_signal = self.film_2d_second(reconstructed_signal)
        
        return reconstructed_signal
    




# ==========================================
# 使用示例 (针对 IQ 信号特征提取的场景)
# ==========================================
if __name__ == "__main__":
    # 假设输入是一个批次的 IQ 信号，Batch=32, Channels=2 (I和Q两路), Seq_Len=1024
    batch_size = 32
    channels = 2
    seq_len = 256
    
    # 生成模拟输入数据
    mock_iq_signal = torch.randn(batch_size, channels, seq_len)
    
    # 实例化 GTA 模块
    # 注意：由于输入通道只有2，我们可以设置 reduction_ratio=1 保持隐藏层维度
    gta_block = Closed_Train(in_channles=2,mid_channels=16,d_model=128,seq_len=256,out_channels=128,hidden_features=64,num_classes=9)
    
    # 前向计算
    out = gta_block(mock_iq_signal)
    
    print(f"输出信号形状: {out.shape}")


    # # 假设输入是一个批次的 IQ 信号，Batch=32, Channels=2 (I和Q两路), Seq_Len=1024
    # batch_size = 32
    # channels = 1
    # features = 128
    
    # # 生成模拟输入数据
    # mock_iq_signal = torch.randn(batch_size, channels)
    # mock_iq_signal1 = torch.randn(batch_size, features)
    
    # # 实例化 GTA 模块
    # # 注意：由于输入通道只有2，我们可以设置 reduction_ratio=1 保持隐藏层维度
    # gta_block = FiLMLayer_1D(features,channels)
    
    # # 前向计算
    # out = gta_block(mock_iq_signal1,mock_iq_signal)
    
    # print(f"输出信号形状: {out.shape}")
    
