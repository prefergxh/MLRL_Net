import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch


def draw_academic_curves(epochs, acc_data_dict, save_path,label="Accuracy (%)"):
    """
    绘制高颜值的多曲线对比图
    :param epochs: x轴的轮数数组，如 np.arange(1, 101)
    :param acc_data_dict: 字典格式，键是曲线名字(图例)，值是准确率数组
    :param save_path: 图片保存路径
    """
    # 1. 设置全局画布大小和分辨率 (学术规范通常为 8x6 或 4x3 比例)
    plt.figure(figsize=(8, 6), dpi=150) # 预览用150，最终导出论文时建议改成 300 或 600

    # 2. 预设一条“高级感”的样式流水线
    # 这里使用了 Tableau 的默认柔和色系 (C0-C9)，并搭配不同的标记
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    markers = ['o', 's', '^', 'D', 'v']
    
    # 3. 优雅地循环画出所有的线
    for i, (model_name, acc_values) in enumerate(acc_data_dict.items()):
        plt.plot(
            epochs, 
            acc_values, 
            label=model_name,          # 图例名称
            color=colors[i % len(colors)], # 自动循环取颜色
            marker=markers[i % len(markers)], # 自动循环取形状
            linewidth=1.5,             # 线条适中偏细，显精致
            markersize=5,              # 标记大小
            markevery=10,              # 🌟 灵魂参数：每隔10个epoch才画一个点，防止毛毛虫！
            linestyle='-', 
            alpha=0.9                  # 稍微加一点点透明度，让曲线交叉时更好看
        )

    # 4. 图表修饰 (去除冗余元素，提升高级感)
    ax = plt.gca()
    # 去掉上方和右侧的边框黑线 (顶级期刊极简风)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置坐标轴标签和字体大小
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel(label, fontsize=12, fontweight='bold')
    
    # 增加极其微弱的网格辅助线，只画横线
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 5. 设置图例 (去掉图例的丑陋边框)
    plt.legend(loc='lower right', frameon=False, fontsize=11)

    # 6. 🌟 终极保存：去除四周多余白边 (bbox_inches='tight')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 高清图表已成功保存至: {save_path}")


# 绘制混淆矩阵
def drawing_confusion_matrices(all_labels,all_preds,classes):
    cm = confusion_matrix(all_labels,all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
            xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (SEI Model)')
    plt.ylabel('True Label')  
    plt.xlabel('Predicted Label') 
    plt.tight_layout()
    plt.savefig("work_dirs/confusion_matrix.png")



# 生成邻接矩阵热力图
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def plot_adjacency_matrix(A_tensor, num_nodes=5, cmap="Blues"):
    """
    可视化生成的邻接矩阵
    A_tensor: PyTorch Tensor, 形状例如 [Batch, 256, 256]
    num_nodes: 想要截取可视化的节点数量 (对应图中的 T1~T5 则设为 5)
    """
    # 1. 从 Batch 中取出第一个样本的矩阵，并将其转为 numpy 数组
    # detach() 用于脱离计算图，cpu() 用于将其移至内存
    A_np = A_tensor[0].detach().cpu().numpy()
    
    # 2. 截取左上角的子矩阵 (例如 5x5)
    A_subset = A_np[:num_nodes, :num_nodes]
    
    # 3. 设置画板大小
    plt.figure(figsize=(6, 5))
    
    # 4. 使用 Seaborn 绘制热力图
    # annot=True 表示在格子中显示具体数值
    # fmt=".1f" 表示保留一位小数 (像图中的 0.8, 1.4)
    # linewidths 增加格子之间的分割线
    ax = sns.heatmap(A_subset, annot=True, fmt=".1f", cmap=cmap, 
                     linewidths=1, linecolor='white', 
                     cbar_kws={'label': 'Connection Weight'})
    
    # 5. 设置坐标轴标签 (T1, T2, T3...)
    labels = [f"T{i+1}" for i in range(num_nodes)]
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels, rotation=0)
    
    # 移动 X 轴的刻度到上方 (使其更像你提供的截图)
    ax.xaxis.tick_top()
    
    plt.title(f"Adjacency Matrix (Top {num_nodes}x{num_nodes} Subset)", pad=20)
    plt.tight_layout()
    plt.show()