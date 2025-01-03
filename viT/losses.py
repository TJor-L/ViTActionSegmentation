import torch
import torch.nn as nn


class FrameContinuityLoss(nn.Module):
    def __init__(self, num_classes):
        super(FrameContinuityLoss, self).__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        batch_size, window_size, num_classes = predictions.size()
        assert num_classes == self.num_classes, "class_num not match"

        # [batch_size, window_size]
        pred_classes = torch.argmax(predictions, dim=-1)

        classes = torch.arange(
            self.num_classes, device=predictions.device).view(1, -1, 1)
        
        # [batch_size, num_classes, window_size]
        pred_masks = (pred_classes.unsqueeze(1) == classes).int()
        # [batch_size, num_classes, window_size]
        true_masks = (targets.unsqueeze(1) == classes).int()

        # [batch_size * num_classes, window_size]
        pred_masks = pred_masks.view(-1, window_size)
        true_masks = true_masks.view(-1, window_size)

        # [batch_size, num_classes]
        pred_continuity = self.compute_runs(
            pred_masks).view(batch_size, num_classes)
        true_continuity = self.compute_runs(
            true_masks).view(batch_size, num_classes)

        continuity_loss = self.mse_loss(
            pred_continuity.float(), true_continuity.float())

        return continuity_loss

    def compute_runs(self, x):

        # N = batch_size * num_classes, W = window_size
        N, W = x.size()
        device = x.device

        # [N, W+2]
        x_padded = torch.nn.functional.pad(x, (1, 1), mode='constant', value=0)

        # [N, W+1]
        dx = x_padded[:, 1:] - x_padded[:, :-1]

        # [num_runs, 2]
        run_starts = (dx == 1).nonzero(as_tuple=False)
        run_ends = (dx == -1).nonzero(as_tuple=False)

        if run_starts.size(0) == 0:
            return torch.zeros(N, device=device)

        # [num_runs]
        run_start_seq = run_starts[:, 0]
        run_start_time = run_starts[:, 1]
        
        # [num_runs]
        run_end_seq = run_ends[:, 0]
        run_end_time = run_ends[:, 1]

        assert run_start_seq.size(0) == run_end_seq.size(0), "run_starts and run_ends not match"

        # [num_runs]
        run_lengths = run_end_time - run_start_time

        # [N]
        max_runs = torch.full((N,), -1, device=device, dtype=run_lengths.dtype)

        # [N]
        max_runs = max_runs.scatter_reduce(0, run_start_seq, run_lengths, reduce='amax', include_self=False)

        # [N]
        max_runs = torch.where(max_runs < 0, torch.zeros_like(max_runs), max_runs)

        return max_runs



class TotalLossWithContinuity(nn.Module):
    def __init__(self, class_weights=None, lambda_continuity=0.1, num_classes=17):
        super(TotalLossWithContinuity, self).__init__()

        if class_weights == None:
            self.ce_loss = nn.CrossEntropyLoss()
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.continuity_loss = FrameContinuityLoss(num_classes)
        self.lambda_continuity = lambda_continuity

    def forward(self, logits, targets):

        ce_loss = self.ce_loss(
            logits.view(-1, logits.size(-1)), targets.view(-1))

        continuity_loss = self.continuity_loss(logits, targets)

        total_loss = ce_loss + self.lambda_continuity * continuity_loss
        return total_loss
