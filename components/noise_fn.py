import torch
def add_online_awgn(clean_signal, snr_db):
    """
    在 Tensor 层面实时添加指定 SNR 的高斯白噪声
    clean_signal 形状: (B, W, 2)
    """
    B, W, C = clean_signal.shape
    
    # 1. 计算信号的平均功率
    # 沿着 W(1) 和 C(2) 维度求均方值
    signal_power = torch.mean(clean_signal**2, dim=[1, 2], keepdim=True) 
    
    # 2. 将 SNR 从 dB 转换为线性刻度
    snr_linear = 10 ** (snr_db / 10.0)
    
    # 3. 计算所需的噪声功率
    noise_power = signal_power / snr_linear
    
    # 4. 生成与 clean_signal 形状相同的标准正态分布噪声，并缩放其标准差
    noise = torch.randn_like(clean_signal) * torch.sqrt(noise_power)
    
    # 5. 叠加信号与噪声
    noisy_signal = clean_signal + noise
    return noisy_signal