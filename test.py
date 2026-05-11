import torch
import os
import numpy as np
from tqdm import tqdm
from model.EAGCN_Net import EAGCN_Net
from dataset.dataset_tools import Load_Data_OSR
from components.metric import AccuracyMetric_Openset
from components.utilsall import load_model
from components.OpenMax import compute_mavs_and_distances,fit_weibull_models,openmax_inference
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score

# 参数以及常量设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 数据集导入
dataset_name = 'ManyTx'
dataset_path = './dataset/'
label_known = 9
label_unknown = 3
train_loader,valid_loader,test_loader = Load_Data_OSR(dataset_name,dataset_path,batch_size=32,num_known=label_known,num_unknown=label_unknown)
# 模型导入，只是起一个声明变量的作用
model = EAGCN_Net(in_features=1,num_nodes=256,num_classes=9).to(DEVICE)
num_known_classes = 9
openset_acc_metric = AccuracyMetric_Openset()

def main():
    checkpoint_file = "my_checkpoint.pth.tar"
    load_model(checkpoint_file,model,device=DEVICE)
    # openmax准备阶段
    mavs , distance = compute_mavs_and_distances(model,train_loader,num_known_classes,device=DEVICE)
    weibull_models = fit_weibull_models(distance,tail_size=15)
    #真实测试
    loop = tqdm(test_loader)
    openset_acc_metric.reset()
    model.eval()
    # 测试
    correct_base = 0
    total_known = 0
    # 测试
    all_openmax_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(loop):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            logits = model(data)
            # --- 【拦截测试区 开始】 ---
            # 我们先不管 OpenMax，只看网络自己猜得准不准！
            # 过滤出只属于已知类的样本 (假设标签小于 num_known)
            known_mask = target < num_known_classes
            if known_mask.sum() > 0:
                known_logits = logits[known_mask]
                known_labels = target[known_mask]
                
                # 网络直接取 Logits 最大值的索引作为预测
                base_preds = known_logits.argmax(dim=1)
                correct_base += (base_preds == known_labels).sum().item()
                total_known += known_mask.sum().item()
            # --- 【拦截测试区 结束】 ---
            predicted_class,openmax_probs = openmax_inference(logits,mavs,weibull_models,alpha=3,epsilon=0.7,device=DEVICE)
            openset_acc_metric.update(predicted_class,target)
            Acc_Known,Acc_Rogue,Acc_OpenSet = openset_acc_metric.compute()
            loop.set_postfix(
                    Known=f"{Acc_Known * 100:.2f}%", 
                    Rogue=f"{Acc_Rogue * 100:.2f}%", 
                    Open=f"{Acc_OpenSet * 100:.2f}%"
            )
            all_openmax_probs.extend(openmax_probs.cpu().numpy())
            is_known = (target < num_known_classes).int()
            all_labels.extend(is_known.cpu().numpy())
        probs_array = np.array(all_openmax_probs) 
        labels_array = np.array(all_labels)
        labels_array = 1- labels_array
        anomaly_scores = probs_array[:, -1]
        auc_score = roc_auc_score(labels_array,anomaly_scores)
        Acc_Known,Acc_Rogue,Acc_OpenSet = openset_acc_metric.compute()
        print(f"Known:{Acc_Known * 100:.2f}%,Rogue:{Acc_Rogue * 100:.2f}%,open:{Acc_OpenSet * 100:.2f}%")
        print(f"🔥 开集拒识 AUC 面积: {auc_score:.4f}")
        if total_known > 0:
            base_acc = correct_base / total_known
            print(f"🌟 【拦截诊断】网络原始闭集准确率 (Base Accuracy): {base_acc * 100:.2f}%")
        

    

if __name__ == "__main__":
    main()