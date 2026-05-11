import torch
import os

def save_checkpoint(state,filename="my_checkpoint.pth.tar"):
    print("=>Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint_path, model, optimizer_model, device):
    """
    完美的无损恢复断点函数
    """
    print(f"=> 正在尝试加载断点文件: {checkpoint_path}")
    
    if not os.path.isfile(checkpoint_path):
        print(f"❌ 找不到断点文件: {checkpoint_path}，将从头开始全新训练！")
        # 如果没找到文件，返回初始 epoch 0 和初始最好成绩 0.0
        return 0, 0.0 
    
    # 1. 读取断点文件 (加入 map_location 防护，防止跨显卡/CPU加载报错)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 2. 恢复模型权重 (Encoder 和 Decoder)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    # 4. 恢复两个优化器的内部状态 (带着之前的惯性和动量继续冲锋)
    optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
    
    # 5. 提取进度和指标
    start_epoch = checkpoint['epoch']
    
    # 兼容处理：如果你之前的断点里没存 best_Acc_OpenSet，就默认给 0.0
    best_acc = checkpoint.get('best_Acc', 0.0)
    
    print(f"✅ 断点加载成功！将从第 {start_epoch} 轮继续训练...")
    print(f"🌟 历史最高综合开集准确率: {best_acc * 100:.2f}%")
    
    return start_epoch, best_acc

def load_model(checkpoint_path, model,device):
    # 加载模型函数
    #1. 加载断点文件
    checkpoint = torch.load(checkpoint_path, map_location=device)
    #2. 模型恢复
    model.load_state_dict(checkpoint['model_state_dict'])