import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from model.MLRL_Net import Closed_Train
from dataset.dataset_tools import Load_Data_OSR
from components.metric import AccuracyMetric
from components.utilsall import save_checkpoint
from components.utilsall import load_checkpoint
from components.drawing import draw_academic_curves
from components.init_weight import init_weights
from datetime import datetime


# 超参数配置
MODEL_NAME = "Closed_Train"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_BATCH_SIZE = 32
NUM_EPOCHS = 150
NUM_WORKERS = 4
root_path = os.path.abspath('.')
dataset_name = 'ManyTx'
dataset_path = './dataset/'

# 日志文件的创建工具函数
def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

def main():
    def train_fn(loader,model,optimizer_normal,loss_CE,scaler,epoch,device=DEVICE):
        model.train()
        loop = tqdm(loader, desc=f"Train Epoch {epoch+1}", mininterval=0.5)
        acc_metric_train.reset()
        train_losses = []
        for batch_idx,(data,targets) in enumerate(loop):
            # 将数据加载到GPU上
            data = data.to(device)
            targets = targets.to(device)
            # 前向和求损
            with torch.cuda.amp.autocast():
                logits = model(data)
                loss_ce = loss_CE(logits.float(),targets)
            acc_metric_train.update(logits,targets)
            acc = acc_metric_train.compute()
            train_losses.append(loss_ce.item())
            # 梯度清零
            optimizer_normal.zero_grad()
            #反向传播
            scaler.scale(loss_ce).backward()
            scaler.unscale_(optimizer_normal)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 优化
            scaler.step(optimizer_normal)
            scaler.update()

            # 显示进度
            loop.set_postfix(loss=loss_ce.item(),acc=acc)
        return acc_metric_train.compute(),np.mean(train_losses)
    
    def val_fn(loader,model,epoch,device=DEVICE):
        model.eval()
        loop = tqdm(loader, desc=f"Train Epoch {epoch+1}", mininterval=0.5)
        acc_metric_val.reset()
        with torch.no_grad():
            for batch_idx,(data,targets) in enumerate(loop):
                data = data.to(device)
                targets = targets.to(device)
                logits = model(data)
                acc_metric_val.update(logits,targets)
                acc = acc_metric_val.compute()
                loop.set_postfix(acc=acc)
            acc_epoch =  acc_metric_val.compute()
            return acc_epoch
        
        
    #获取数据集
    label_known = 9
    label_unknown = 3
    train_loader,valid_loader,_ = Load_Data_OSR(dataset_name,dataset_path,batch_size=TRAIN_BATCH_SIZE,num_known=label_known,num_unknown=label_unknown)

    #配置
    lr_normal = 0.001
    model = Closed_Train(in_channles=2,mid_channels=16,d_model=128,seq_len=256,out_channels=128,hidden_features=64,num_classes=9).to(DEVICE)
    model.apply(init_weights)
    loss_CE = nn.CrossEntropyLoss()
    optimizer_normal = optim.Adam(model.parameters(),lr_normal)

    # 定义学习率调度器
    # scheduler_normal = MultiStepLR(optimizer_normal,milestones=[30,80],gamma=0.1)
    # scheduler_center = MultiStepLR(optimizer_M,milestones=[30,80],gamma=0.1)


    #评估对象
    acc_metric_train = AccuracyMetric()
    acc_metric_val = AccuracyMetric()
    # 混合精度训练节省显存，提高模型训练效率
    scaler = torch.cuda.amp.GradScaler()

    #评估指标
    best_acc_epoch = 0
    best_acc = 0
    num_epoch = []
    num_train_loss = []
    num_val_acc = []
    num_train_acc = []

    #训练中断恢复
    checkpoint_file = "my_checkpoint.pth.tar"
    start_epoch,best_Acc = load_checkpoint(
        checkpoint_path=checkpoint_file,
        model=model,
        optimizer_model=optimizer_normal,
        device=DEVICE
    )

    #输出
    make_dir('./work_dirs')
    save_model_file_path = os.path.join(root_path,'work_dirs',MODEL_NAME)
    make_dir(save_model_file_path)
    save_file_name = os.path.join(save_model_file_path,MODEL_NAME+'.txt')
    save_best_acc_file_name = os.path.join(save_model_file_path,'best_acc_checkpoint_'+MODEL_NAME+'.pth.tar')
    save_file = open(save_file_name,'a')
    save_file.write(
    '\n---------------------------------------start--------------------------------------------------\n'
    )
    save_file.write(datetime.now().strftime("%Y-%m-%d,%H:%M:%S\n"))


    #开始训练
    for epoch in range(start_epoch,NUM_EPOCHS):
        # 每一轮训练
        train_acc,train_loss = train_fn(train_loader,model,optimizer_normal,loss_CE,scaler,epoch)
        #每一次训练结束，更新一次调度器
        # scheduler_normal.step()
        # scheduler_center.step()
        # 训练完测试检测，当模型有过拟合倾向时及时停止
        val_acc = val_fn(valid_loader,model,epoch,device=DEVICE)
        # 保存中断时模型的所有参数，防止中断而导致要从头重新训练
        checkpoint = {
            # 1. 进度追踪 (用于恢复训练时知道从哪开始)
            'epoch': epoch + 1,
            
            # 2. 核心网络与参数 (用于纯测试/特征提取)
            'model_state_dict': model.state_dict(),                  # 网络权重 \theta
            
            # 3. 双优化器状态 (用于无损恢复训练)
            'optimizer_model_state_dict': optimizer_normal.state_dict(),
            
            # 4. 历史最佳指标 (用于早停逻辑和对比)
            'best_Acc': best_acc, 
            
            # 5. 当前轮次的所有指标 (供记录和排查日志用)
            'current_metrics': {
                'train_loss': train_loss,
                'train_acc': train_acc
            }
        }
        save_checkpoint(checkpoint)

        # 保存一些数据以便后面画图使用
        num_epoch.append(epoch + 1)
        num_train_loss.append(train_loss)
        num_val_acc.append(val_acc)
        num_train_acc.append(train_acc)

        # 保存最精准的一轮训练
        if best_acc<val_acc:
            best_acc = val_acc
            best_acc_epoch = epoch
            # 前面训练的结果一般，不好所以从第40轮才开始
            if epoch + 1 > 40:
                save_dir = os.path.dirname(save_best_acc_file_name)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(checkpoint,save_best_acc_file_name)

        print(f"当前epoch:{epoch + 1}  train_acc:{round(train_acc, 4)}"
              f" val_acc:{round(val_acc, 4)}\n"
              f"best_epoch:{best_acc_epoch + 1}  best_acc:{round(best_acc, 4)}\n")    
           
        save_file.write(f"当前epoch:{epoch + 1}  train_acc:{round(train_acc, 4)}\n")
        save_file.write(
            f"epoch is:{epoch + 1}  val_acc:{round(val_acc, 4)}\n")   
        save_file.flush()

    drawing_acc = {
        "train_acc":num_train_acc,
        "val_acc":num_val_acc
    }

    drawing_loss = {
        "train_loss":num_train_loss
    }
    draw_academic_curves(num_epoch,drawing_acc,save_path="work_dirs/fig_acc.png")
    draw_academic_curves(num_epoch,drawing_loss,save_path="work_dirs/fig_loss.png",label="Loss")

    save_file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S\n"))
    save_file.write('\n---------------------------------------end--------------------------------------------------\n')

if __name__ == "__main__":
    main()
