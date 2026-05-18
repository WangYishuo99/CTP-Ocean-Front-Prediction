'''
Author: Yishuo Wang
Date: 2026-02-04 22:21:47
LastEditors: Yishuo Wang
LastEditTime: 2026-04-11 14:50:51
FilePath: \ocean_front\test_files\test_EMA.py
Description: test script for EMA model

Copyright (c) 2026 by Yishuo Wang, All Rights Reserved. 
'''
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

region_name = "GSR"
model_name = "EMA"
step_list = [1, 3, 7]  # 预测步数
input_path = f".\\train_dataset\\{region_name}\\train_dataset.pt"
test_indices_path = f".\\train_dataset\\test_indices.pt"
model_save_path = f".\\saved_models\\{region_name}\\"
txt_save_path = f".\\test_logs\\{region_name}\\"

def rolling_forecast(model, input_seq, steps):
    """
    input_seq: [B,7,3,H,W]
    return: [B,steps,3,H,W]
    """
    preds = []
    current_seq = input_seq.clone()

    for _ in range(steps):
        out = model(current_seq)  # [B,3,H,W]
        preds.append(out.unsqueeze(1))  # [B,1,3,H,W]

        # 拼接滚动
        current_seq = torch.cat(
            [current_seq[:, 1:], out.unsqueeze(1)],
            dim=1
        )

    return torch.cat(preds, dim=1)

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

if __name__ == "__main__":
    os.makedirs(txt_save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- 1. 数据准备 --------
    dataset = torch.load(input_path)   # [T,3,H,W]
    test_indices = torch.load(test_indices_path)   # list of indices
    test_indices = test_indices.tolist()

    test_dataset = WindowDataset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # -------- 2. 加载模型 --------
    model = OceanTransformer().to(device)
    state = torch.load(os.path.join(model_save_path, f'{model_name}.pt'), map_location=device)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state.to(device)

    # -------- 3. 测试 --------
    model.eval()
    for steps in step_list:
        # 不要看loss, 直接保留总体的指标
        total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(test_loader):
                inputs = inputs.to(device)               # [B,7,3,H,W]

                base_indices = test_indices[
                    batch_idx * 32 : batch_idx * 32 + inputs.size(0)
                ]
                max_index = len(dataset) - 1
                valid_mask = [idx + steps - 1 <= max_index for idx in base_indices]

                if not any(valid_mask):
                    continue

                valid_positions = [i for i, ok in enumerate(valid_mask) if ok]
                inputs = inputs[valid_positions]
                base_indices = [base_indices[i] for i in valid_positions]

                batch_targets = []
                for i in range(steps):
                    future_indices = [idx + i for idx in base_indices]
                    gt = dataset[future_indices]   # [B,3,H,W]
                    batch_targets.append(gt)

                batch_targets = torch.stack(batch_targets, dim=1).to(device)
                # [B,steps,3,H,W]

                preds = rolling_forecast(model, inputs, steps=steps)
                # [B,steps,3,H,W]

                # 计算分类指标
                output_flag = preds[:, :, 0, :, :]               # [B,steps,H,W]
                target_flag = batch_targets[:, :, 0, :, :]               # [B,steps,H,W]
                probs = torch.sigmoid(output_flag)
                pred = (probs >= 0.5).int()
                tgt = target_flag.int()
                TP = torch.sum((pred == 1) & (tgt == 1)).item()
                TN = torch.sum((pred == 0) & (tgt == 0)).item()
                FP = torch.sum((pred == 1) & (tgt == 0)).item()
                FN = torch.sum((pred == 0) & (tgt == 1)).item()

                total_TP += TP
                total_TN += TN
                total_FP += FP
                total_FN += FN

        # 计算总体指标
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN) if (total_TP + total_TN + total_FP + total_FN) > 0 else 0

        # 保存到txt文件
        with open(os.path.join(txt_save_path, f'test_metrics_{model_name}.txt'), 'a') as f:
            f.write(f"Test Metrics for {steps}-step prediction:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1_score:.4f}\n")