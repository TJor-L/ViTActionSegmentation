import torch
import torch.nn as nn

class VideoActionSegmentationModel(nn.Module):
    def __init__(self, d_input=1024, d_model=512, nhead=8, num_layers=6, num_classes=17, window_size=30, use_memflow=True, mask=False):
        super(VideoActionSegmentationModel, self).__init__()
        self.use_memflow = use_memflow
        self.mask = mask
        if self.use_memflow:
            self.input_linear = nn.Linear(d_input * 2, d_input)
        self.cnn = nn.Conv1d(in_channels=d_input, out_channels=d_model, kernel_size=3, padding=1)
        self.position_embedding = nn.Parameter(torch.randn(window_size, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, memflow=None):
        if self.use_memflow and memflow is not None:
            x = torch.cat((x, memflow), dim=-1)
            x = self.input_linear(x)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(2, 0, 1)
        x += self.position_embedding.unsqueeze(1)

        # Create a mask to prevent future frames from being accessed
        if self.mask:
            seq_len = x.size(0)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            mask = mask.to(x.device)
        else:
            mask = None

        x = self.transformer_encoder(x, mask=mask)
        x = x.permute(1, 0, 2)
        logits = self.classifier(x)
        return logits
