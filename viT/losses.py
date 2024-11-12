import torch
import torch.nn as nn


class FrameContinuityLoss(nn.Module):
    def __init__(self, num_classes):
        super(FrameContinuityLoss, self).__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        batch_size, window_size, num_classes = predictions.size()
        assert num_classes == self.num_classes, "类别数量不匹配"

        # 获取预测的类别序列
        # [batch_size, window_size]
        pred_classes = torch.argmax(predictions, dim=-1)

        # 构建类别的连续帧矩阵
        classes = torch.arange(
            self.num_classes, device=predictions.device).view(1, -1, 1)
        # [batch_size, num_classes, window_size]
        pred_masks = (pred_classes.unsqueeze(1) == classes).int()
        # [batch_size, num_classes, window_size]
        true_masks = (targets.unsqueeze(1) == classes).int()

        # 重塑形状以便批量处理
        pred_masks = pred_masks.view(-1, window_size)
        true_masks = true_masks.view(-1, window_size)

        # 计算所有掩码的连续帧长度
        pred_continuity = self.compute_runs(
            pred_masks).view(batch_size, num_classes)
        true_continuity = self.compute_runs(
            true_masks).view(batch_size, num_classes)

        # 计算 MSE 损失
        continuity_loss = self.mse_loss(
            pred_continuity.float(), true_continuity.float())

        # 打印中间结果
        # print("Predicted Continuity Matrix:\n", pred_continuity)
        # print("True Continuity Matrix:\n", true_continuity)

        return continuity_loss

    def compute_runs(self, x):
        # x 的形状为 [N, window_size]，其中 N = batch_size * num_classes
        N, window_size = x.size()
        # 在两端填充 0
        x_padded = torch.nn.functional.pad(x, (1, 1), mode='constant', value=0)
        # 计算差分
        dx = x_padded[:, 1:] - x_padded[:, :-1]
        # 找到连续段的开始和结束位置
        run_starts = (dx == 1).nonzero(as_tuple=False)
        run_ends = (dx == -1).nonzero(as_tuple=False)
        # 计算连续段的长度
        lengths = run_ends[:, 1] - run_starts[:, 1]
        # 初始化最大连续长度的张量
        max_runs = torch.zeros(N, device=x.device, dtype=lengths.dtype)
        # 计算每个序列的最大连续长度
        for n in range(N):
            idx = run_starts[:, 0] == n
            if idx.any():
                max_runs[n] = lengths[idx].max()
            else:
                max_runs[n] = 0
        return max_runs


# 总损失结合
class TotalLossWithContinuity(nn.Module):
    def __init__(self, class_weights=None, lambda_continuity=0.1, num_classes=17):
        super(TotalLossWithContinuity, self).__init__()
        # # Ensure class_weights is a tensor on the correct device
        # if not isinstance(class_weights, torch.Tensor):
        #     class_weights = torch.tensor(class_weights, dtype=torch.float)
        if class_weights == None:
            self.ce_loss = nn.CrossEntropyLoss()
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.continuity_loss = FrameContinuityLoss(num_classes)
        self.lambda_continuity = lambda_continuity

    def forward(self, logits, targets):
        # 计算交叉熵损失
        ce_loss = self.ce_loss(
            logits.view(-1, logits.size(-1)), targets.view(-1))

        # 计算连续帧损失
        # (logits batch * windows * numclass + 1)
        continuity_loss = self.continuity_loss(logits, targets)

        # 总损失 = 交叉熵损失 + 连续帧损失
        total_loss = ce_loss + self.lambda_continuity * continuity_loss
        return total_loss
