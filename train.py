'''
Author: Yishuo Wang
Date: 2026-02-04 14:31:53
LastEditors: Yishuo Wang
LastEditTime: 2026-04-11 14:54:50
FilePath: \ocean_front\train_files\train_EMA.py
Description: training script for ocean front prediction model with EMA

Copyright (c) 2026 by Yishuo Wang, All Rights Reserved. 
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

model_name = "EMA"

# 超参数设置
num_epochs = 200
learning_rate = 1e-4
batch_size = 32

# 分辨率，一个格子是9km，要统一量纲
resolution = 9000

# 海水的运动学粘度
kinematic_viscosity = 1e-6  # m^2/s

# 一天的秒数
seconds_per_day = 24 * 60 * 60 

# epsilon and decay for EMA
ema_decay = 0.9
epsilon = 1e-6
warmup_epochs = 1         # 前1个epoch用均匀权重

# average loss for different terms
loss1_avg = 0.0
loss2_avg = 0.0
min_weight_ce = 0.5

# fixed weights for different loss terms
lambda_time = 1e-2
lambda_conv = 1e-3
lambda_diff = 1e-4
lambda_remain = 1.0 - (lambda_time + lambda_conv + lambda_diff)

# early stopping patience
best_f1 = 0.0
patience = 10        # 连续 10 个 epoch 无提升才停
wait = 0
min_delta = 1e-3     # 最小显著提升

region_name = "GSR"
input_path = f".\\train_dataset\\{region_name}\\train_dataset.pt"
train_indices_path = f".\\train_dataset\\train_indices.pt"
val_indices_path = f".\\train_dataset\\val_indices.pt"
txt_save_path = f".\\train_logs\\{region_name}\\"
model_save_path = f".\\saved_models\\{region_name}\\"

class WindowDataset(Dataset):
    def __init__(self, data, indices, input_len=7):
        """
        data: [T,3,H,W]
        indices: list / tensor of target indices t
        """
        self.data = data
        self.indices = indices
        self.input_len = input_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x = self.data[t - self.input_len : t]      # [7,3,H,W]
        y = self.data[t]                            # [3,H,W]
        input_last = self.data[t - 1]               # [3,H,W]

        return x, y, input_last

class SpatialEncoder(nn.Module):
    def __init__(self):
        """处理单帧3xHxW数据的编码器"""
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> [b,16,H/2,W/2]
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # -> [b,32,H/2,W/2]
            nn.GroupNorm(4, 32),
            nn.ReLU(),
        )

    def forward(self, x):
        """输入: [batch,3,H,W] 
           输出: [batch,32,H/2,W/2]"""
        return self.conv_layers(x)

class TemporalDecoder(nn.Module):
    def __init__(self):
        """从编码特征重建HxW图像"""
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # -> [b,16,H,W]
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)            # -> [b,3,H,W]
        )

    def forward(self, x):
        return self.decoder(x)
    
class OceanTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        self.encoder = SpatialEncoder()
        self.temporal_proj = nn.Linear(32, d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=1024,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.channel_restore = nn.Linear(d_model, 32)
        self.decoder = TemporalDecoder()

    def forward(self, x):
        B, T = x.shape[:2]
        spatial_feats = []

        for t in range(T):
            feat = self.encoder(x[:, t])      # [B,32,H',W']
            spatial_feats.append(feat)

        # 用最后一帧做 spatial base（也可 mean）
        feat_base = spatial_feats[-1]

        tokens = torch.stack(
            [f.mean(dim=[2,3]) for f in spatial_feats], dim=1
        )                                      # [B,T,32]

        tokens = self.temporal_proj(tokens)   # [B,T,512]
        context = self.transformer(tokens)
        fused = context.mean(dim=1)            # [B,512]

        gamma = self.channel_restore(fused)    # [B,32]
        gamma = gamma[:, :, None, None]

        restored = feat_base * gamma
        out = self.decoder(restored)
        return out

class PhysicsInformedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def gradient(self, u):
        dy = (u[:, 1:, :] - u[:, :-1, :]) / resolution
        dx = (u[:, :, 1:] - u[:, :, :-1]) / resolution
        return dy, dx

    def laplacian(self, u):
        d2y = (u[:, 2:, :] - 2*u[:, 1:-1, :] + u[:, :-2, :]) / (resolution**2)
        d2x = (u[:, :, 2:] - 2*u[:, :, 1:-1] + u[:, :, :-2]) / (resolution**2)
        return d2y[:, :, :-2] + d2x[:, :-2, :]

    def forward(self, pred, target, input_last):
        """
        pred:        [B,3,H,W]  预测 t+1
        target:      [B,3,H,W]  真值 t+1
        input_last:  [B,3,H,W]    输入序列的最后一帧 t
        """

        # ---------- 1. 分类 ----------
        # 使用带权重的 BCE，解决正负样本不平衡问题
        zone_pred = pred[:,0]
        zone_target = target[:,0]
        pos = (zone_target == 1).sum().item()
        neg = (zone_target == 0).sum().item()
        weight_pos = torch.tensor([neg / pos], dtype=torch.float32, device=pred.device)
        ce_loss = F.binary_cross_entropy_with_logits(zone_pred, zone_target, pos_weight=weight_pos)

        # ---------- 2. 速度 ----------
        u, v = pred[:,1], pred[:,2]
        u_t, v_t = target[:,1], target[:,2]
        vel_loss = F.mse_loss(u, u_t) + F.mse_loss(v, v_t)

        # ---------- 3. 时间变化率 ----------
        u_prev, v_prev = input_last[:,1], input_last[:,2]

        du_dt = (u - u_prev) / seconds_per_day
        dv_dt = (v - v_prev) / seconds_per_day
        du_dt_t = (u_t - u_prev) / seconds_per_day
        dv_dt_t = (v_t - v_prev) / seconds_per_day

        time_loss = F.mse_loss(du_dt, du_dt_t) + F.mse_loss(dv_dt, dv_dt_t)

        # ---------- 4. 对流 ----------
        du_dy, du_dx = self.gradient(u)
        dv_dy, dv_dx = self.gradient(v)
        conv_u = u[:,:-1,:-1]*du_dx[:,:-1,:] + v[:,:-1,:-1]*du_dy[:,:,:-1]
        conv_v = u[:,:-1,:-1]*dv_dx[:,:-1,:] + v[:,:-1,:-1]*dv_dy[:,:,:-1]

        du_dy_t, du_dx_t = self.gradient(u_t)
        dv_dy_t, dv_dx_t = self.gradient(v_t)
        conv_u_t = u_t[:,:-1,:-1]*du_dx_t[:,:-1,:] + v_t[:,:-1,:-1]*du_dy_t[:,:,:-1]
        conv_v_t = u_t[:,:-1,:-1]*dv_dx_t[:,:-1,:] + v_t[:,:-1,:-1]*dv_dy_t[:,:,:-1]

        conv_loss = F.mse_loss(conv_u, conv_u_t) + F.mse_loss(conv_v, conv_v_t)

        # ---------- 5. 扩散 ----------
        lap_u = self.laplacian(u)
        lap_v = self.laplacian(v)
        lap_u_t = self.laplacian(u_t)
        lap_v_t = self.laplacian(v_t)

        diff_loss = F.mse_loss(lap_u, lap_u_t) + F.mse_loss(lap_v, lap_v_t)

        return ce_loss, vel_loss, time_loss, conv_loss, diff_loss

if __name__ == "__main__":
    os.makedirs(txt_save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- 1. 加载数据 --------
    data = torch.load(input_path)                  # [T,3,H,W]
    train_indices = torch.load(train_indices_path)
    val_indices = torch.load(val_indices_path)

    # 保证是 list[int]
    train_indices = train_indices.tolist()
    val_indices = val_indices.tolist()

    # -------- 2. Dataset & DataLoader --------
    train_dataset = WindowDataset(data, train_indices)
    val_dataset = WindowDataset(data, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # -------- 3. 模型 / 损失 / 优化器 --------
    model = OceanTransformer().to(device)
    criterion = PhysicsInformedLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -------- 4. 训练循环 --------
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (x, y, input_last) in enumerate(train_loader):
            x = x.to(device)                   # [B,7,3,H,W]
            y = y.to(device)                   # [B,3,H,W]
            input_last = input_last.to(device) # [B,3,H,W]

            optimizer.zero_grad()
            pred = model(x)
            ce_loss, vel_loss, time_loss, conv_loss, diff_loss = criterion(pred, y, input_last)

            # EMA 更新
            ce_v   = ce_loss.item()
            vel_v  = vel_loss.item()

            # 权重：warmup + 逆损失 + 下限 + 归一化
            if epoch < warmup_epochs:
                w1 = w2 = 0.5 * lambda_remain
            else:
                loss1_avg = ema_decay * loss1_avg + (1 - ema_decay) * ce_v
                loss2_avg = ema_decay * loss2_avg + (1 - ema_decay) * vel_v
                    
                w1 = 1.0 / (loss1_avg + epsilon)
                w2 = 1.0 / (loss2_avg + epsilon)

                # 先归一化
                wsum = w1 + w2
                w1, w2 = w1/wsum, w2/wsum

                if w1 < min_weight_ce:
                    w1 = min_weight_ce
                    w2 = 1.0 - w1
                
                w1 *= lambda_remain
                w2 *= lambda_remain

            total_loss = w1*ce_loss + w2*vel_loss + lambda_time*time_loss + lambda_conv*conv_loss + lambda_diff*diff_loss
    
            with open(os.path.join(txt_save_path, f'loss_weights_{model_name}.txt'), 'a') as f:
                f.write(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}] | "
                        f"W_ce: {w1:.4f}, W_vel: {w2:.4f}\n")
            with open(os.path.join(txt_save_path, f'loss_details_{model_name}.txt'), 'a') as f:
                f.write(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}] | "
                        f"EMA ce:{loss1_avg:.4f} vel:{loss2_avg:.4f} | "
                        f"ce:{ce_v:.4f} vel:{vel_v:.4f} | "
                        f"Total:{total_loss.item():.4f}\n")
                
            total_loss.backward()
            optimizer.step()

        # -------- 5. 验证 --------
        # 计算训练集和验证集的 Precision, Recall, F1 Score
        model.eval()
        train_total_TP, train_total_TN, train_total_FP, train_total_FN = 0, 0, 0, 0
        with torch.no_grad():
            for batch_idx, (x, y, input_last) in enumerate(train_loader):
                x = x.to(device)                   # [B,7,3,H,W]
                y = y.to(device)                   # [B,3,H,W]
                input_last = input_last.to(device) # [B,3,H,W]

                pred = model(x)
                output_flag = pred[:, 0, :, :]
                target_flag = y[:, 0, :, :]

                probs = torch.sigmoid(output_flag)
                pred_labels = (probs >= 0.5).int()
                tgt_labels = target_flag.int()
                TP = torch.sum((pred_labels == 1) & (tgt_labels == 1)).item()
                TN = torch.sum((pred_labels == 0) & (tgt_labels == 0)).item()
                FP = torch.sum((pred_labels == 1) & (tgt_labels == 0)).item()
                FN = torch.sum((pred_labels == 0) & (tgt_labels == 1)).item()

                train_total_TP += TP
                train_total_TN += TN
                train_total_FP += FP
                train_total_FN += FN

        train_precision = train_total_TP / (train_total_TP + train_total_FP) if (train_total_TP + train_total_FP) > 0 else 0
        train_recall = train_total_TP / (train_total_TP + train_total_FN) if (train_total_TP + train_total_FN) > 0 else 0
        train_f1_score = 2 * train_precision * train_recall / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
        print(f"Training after Epoch {epoch+1}: Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1_score:.4f}")
        with open(os.path.join(txt_save_path, f'train_metrics_{model_name}.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1} | Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1_score:.4f}\n")

        val_total_TP, val_total_TN, val_total_FP, val_total_FN = 0, 0, 0, 0
        with torch.no_grad():
            for batch_idx, (x, y, input_last) in enumerate(val_loader):
                x = x.to(device)                   # [B,7,3,H,W]
                y = y.to(device)                   # [B,3,H,W]
                input_last = input_last.to(device) # [B,3,H,W]

                pred = model(x)
                output_flag = pred[:, 0, :, :]
                target_flag = y[:, 0, :, :]

                probs = torch.sigmoid(output_flag)
                pred_labels = (probs >= 0.5).int()
                tgt_labels = target_flag.int()
                TP = torch.sum((pred_labels == 1) & (tgt_labels == 1)).item()
                TN = torch.sum((pred_labels == 0) & (tgt_labels == 0)).item()
                FP = torch.sum((pred_labels == 1) & (tgt_labels == 0)).item()
                FN = torch.sum((pred_labels == 0) & (tgt_labels == 1)).item()

                val_total_TP += TP
                val_total_TN += TN
                val_total_FP += FP
                val_total_FN += FN

        val_precision = val_total_TP / (val_total_TP + val_total_FP) if (val_total_TP + val_total_FP) > 0 else 0
        val_recall = val_total_TP / (val_total_TP + val_total_FN) if (val_total_TP + val_total_FN) > 0 else 0
        val_f1_score = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

        print(f"Validation after Epoch {epoch+1}: Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1_score:.4f}")
        with open(os.path.join(txt_save_path, f'val_metrics_{model_name}.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1} | Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1_score:.4f}\n")   

        # 早停检查
        if val_f1_score - best_f1 > min_delta:
            best_f1 = val_f1_score
            wait = 0
            # 保存当前最优模型
            torch.save(model.state_dict(), os.path.join(model_save_path, f'{model_name}.pt'))
            print(f"New best model saved with F1 Score: {best_f1:.4f}")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break        