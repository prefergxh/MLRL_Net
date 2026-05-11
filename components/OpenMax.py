import torch
import numpy as np
import libmr
from scipy.spatial.distance import cosine

#将训练集重新输入训练好的模型中，来计算每个类别的MAV和每个AV到MAV的距离
def compute_mavs_and_distances(model, train_loader, num_classes, device='cpu'):
    model.eval()
    class_logits = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            
            # 仅收集分类正确的样本
            for i in range(len(labels)):
                if preds[i] == labels[i]:
                    class_logits[labels[i].item()].append(logits[i].cpu().numpy())
    
    mavs = {}
    dists = {i: [] for i in range(num_classes)}
    
    for c in range(num_classes):
        if len(class_logits[c]) == 0:
            continue # 防止某些类没有正确样本
            
        logits_c = np.array(class_logits[c])
        # 计算均值激活向量 (MAV)
        mavs[c] = np.mean(logits_c, axis=0)
        
        # 计算该类所有正确样本到 MAV 的距离 (欧氏距离)
        for logit in logits_c:
            # dist = np.linalg.norm(logit - mavs[c]) 
            dist = cosine(logit,mavs[c]) if np.any(logit) else 1.0
            dists[c].append(dist)
            
    return mavs, dists

# 对每个类别的较远样本进行Weibull分布模拟,这里默认较远的离群样本点数是10个
def fit_weibull_models(distances, tail_size=10):

    weibull_models = {}
    for c, dists in distances.items():
        mr = libmr.MR() 
        dists = np.array(dists,dtype=np.float64)
        
        # 提取距离最远的 tail_size 个样本
        # 注意：如果该类样本总数少于 tail_size，则取全部样本
        actual_tail_size = min(tail_size, len(dists))
        
        # 拟合 Weibull 分布
        mr.fit_high(dists, actual_tail_size) 
        weibull_models[c] = mr        
    return weibull_models

# 评分及开集拒识，适用测试阶段
def openmax_inference(batch_logits_tensor, mavs, weibull_models, alpha=3, epsilon=0.4,device='cpu'):
    """
    Algorithm 2：测试阶段，概率重校准与拒识判定
    参数：
        batch_logits_tensor：测试的一批次信号经过模型处理后的得分tensor向量
        mavs：训练集各个类别的中心特征（得分向量）
        weibull_models：各个类别的weibull分布
        alpha：超参数，表示只对得分的前alpha个类别进行处理
        epsilon：防止除零
    """
    batch_logits = batch_logits_tensor.detach().cpu().numpy()
    batch_size,num_classes = batch_logits.shape
    #结果列表
    batch_preds = []
    batch_probs = []

    # 3. 逐个样本进行 OpenMax 运算 (因为 libmr 必须逐个处理)
    for b in range(batch_size):
        logits = batch_logits[b]
        
        # --- 原有 OpenMax 核心逻辑开始 ---
        sorted_indices = np.argsort(logits)[::-1]
        weights = np.ones(num_classes)
        
        for i in range(alpha):
            c = sorted_indices[i]
            # 注意：这里的 mavs[c] 应该是 numpy 数组
            # dist = np.linalg.norm(logits - mavs[c]) 
            dist = cosine(logits,mavs[c]) if np.any(logits) else 1.0
            w_score = weibull_models[c].w_score(dist) 
            weights[c] = 1 - ((alpha - i) / alpha) * w_score

        pos_logits = logits - np.min(logits)

        recalibrated_logits = pos_logits * weights
        unknown_logit = np.sum(pos_logits * (1 - weights))
        
        final_logits = np.append(recalibrated_logits, unknown_logit)
        
        e_x = np.exp(final_logits - np.max(final_logits))
        openmax_probs = e_x / e_x.sum()
        
        predicted_class = np.argmax(openmax_probs)
        max_prob = np.max(openmax_probs)
        
        if predicted_class == num_classes or max_prob < epsilon:
            predicted_class = -1
        # --- 原有 OpenMax 核心逻辑结束 ---
            
        # 收集当前样本的结果
        batch_preds.append(predicted_class)
        batch_probs.append(openmax_probs)

    # 4. 重新组装为 PyTorch Tensor，并发送回原来的计算设备 (GPU)
    # 类别索引使用 torch.long，方便后续做准确率比对
    predicted_classes_tensor = torch.tensor(batch_preds, dtype=torch.long, device=device)
    # 概率矩阵使用 torch.float32
    openmax_probs_tensor = torch.tensor(np.array(batch_probs), dtype=torch.float32, device=device)
    
    return predicted_classes_tensor, openmax_probs_tensor