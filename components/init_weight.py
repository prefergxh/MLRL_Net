import torch.nn as nn

def init_weights(m):
    """
    针对 MEDAE_Net 的全局参数初始化函数
    """
    # 1. 卷积层与转置卷积层 (对应 Encoder 和 Decoder 中的主要特征提取)
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose2d)):
        # 使用 Kaiming 正态分布初始化，专门针对 ReLU 激活函数优化
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 如果你有的层没写 bias=False，这里做个安全兜底
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    # 2. 批归一化层
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
        
    # 3. 全连接层 (Classifier 和 Decoder 的第一层)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)