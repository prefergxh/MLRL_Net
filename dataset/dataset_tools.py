import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import TensorDataset,DataLoader
import dataset.data_utilities as du

"""
数据处理辅助函数参数设置

sig：信号数据
labels：标签数据
batch_size：批处理
shuffle：是否打乱顺序
"""
# 数据集处理辅助函数，将数据集中原始的numpy数据转换为DataLoader可以进行分批处理
def create_dataloader(sig, labels, batch_size, shuffle=True, drop_last=False):
    sig_tensor = torch.tensor(sig, dtype=torch.float32)
    # PyTorch 的 CrossEntropyLoss 需要 LongTensor 格式的索引标签
    label_tensor = torch.tensor(labels.flatten(), dtype=torch.long)
    dataset = TensorDataset(sig_tensor, label_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,drop_last=drop_last)


"""
数据生成函数参数设置

dataset_name：数据集名称
dataset_path：数据集路径
batch_size：一批中信号的数量
num_tx：选择num_tx个发射机发射的数据
num_rx：选择num_rx个接收机接收的数据
equalized：默认使用未经过信道均衡的数据，防止指纹在信道均衡处理中被抹除
val_frac：验证集占整个数据集的比例，一般是10%
test_frac：测试集占整个数据集的比例，一般是10%
"""
def Load_Data_OSR(dataset_name,dataset_path,batch_size,num_rx=None,equalized=0,val_frac=0.1,test_frac=0.1,num_known=10, num_unknown=5):
        compact_dataset = du.load_compact_pkl_dataset(dataset_path, dataset_name)
        known_tx_list = compact_dataset['tx_list'][0:num_known]
        unknown_tx_list = compact_dataset['tx_list'][num_known:num_known+num_unknown]
        rx_list = compact_dataset['rx_list'][0:num_rx]
        capture_date_list = compact_dataset['capture_date_list']
        # 已知类别的数据集构建，train_augset, val_augset将主要用来训练的时候用，计算闭集的准确率
        dataset_known = du.merge_compact_dataset(compact_dataset, capture_date_list, known_tx_list, rx_list, equalized=equalized)
        train_augset_kn, val_augset_kn, _ = du.prepare_dataset(dataset_known, known_tx_list, val_frac, test_frac)
        [sig_train_kn, txidNum_train_kn, _, cls_weights] = train_augset_kn
        [sig_valid_kn, txidNum_valid_kn, _, _] = val_augset_kn
        # 未知类别的数据集构建
        dataset_unknown = du.merge_compact_dataset(compact_dataset, capture_date_list, unknown_tx_list, rx_list, equalized=equalized)
        _, val_augset_un, _ = du.prepare_dataset(dataset_unknown, unknown_tx_list, val_frac, test_frac)
        [sig_valid_un, txidNum_valid_un, _, _] = val_augset_un
        txidNum_valid_un = np.full_like(txidNum_valid_un, fill_value=num_known)
        # 合并已知和未知类别的验证集构成测试集
        sig_test = np.concatenate([sig_valid_kn,sig_valid_un],axis=0)
        label_test = np.concatenate([txidNum_valid_kn,txidNum_valid_un],axis=0)
        # 构建 DataLoader
        train_loader = create_dataloader(sig_train_kn, txidNum_train_kn, batch_size, shuffle=True,drop_last=True)
        valid_loader = create_dataloader(sig_valid_kn, txidNum_valid_kn, batch_size, shuffle=False,drop_last=False)
        test_loader = create_dataloader(sig_test,label_test,batch_size,shuffle=False,drop_last=False)
        return train_loader,valid_loader,test_loader


if __name__ == "__main__":
    dataset_name = 'ManyTx'
    dataset_path = './dataset/'
    train_loader,valid_loader,test_loader = Load_Data_OSR(dataset_name,dataset_path,batch_size=32)
    num = 0
    for sigs,labels in train_loader:
        num += 1
        # print(sigs.shape)
        # print(labels)
        print(sigs.shape)
        break