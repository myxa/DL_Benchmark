from random import uniform, randint

import torch
from torch import nn



class ResidualBlock(nn.Module):
    """Residual block"""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x) + x


class MeanMLP(nn.Module):
    """
    meanMLP model for fMRI data.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]

    Hyperparameters expected in model_cfg:
        dropout: float
        hidden_size: int
        num_layers: int
    Data info expected in model_cfg:
        input_size: int - input_feature_size
        output_size: int - n_classes
    """

    def __init__(self, input_size, output_size, dropout, hidden_size, num_layers):
        super().__init__()


        # input block
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]
        # inter blocks: default HPs model has none of them
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    ResidualBlock(
                        nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.LayerNorm(hidden_size),
                        )
                    ),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ),
            )

        # output block
        layers.append(
            nn.Linear(hidden_size, output_size),
        )

        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, introspection=False):
        bs, tl, fs = x.shape  # [batch_size, time_length, input_feature_size]

        fc_output = self.fc(x.view(-1, fs))
        fc_output = fc_output.view(bs, tl, -1)

        logits = fc_output.mean(1)

        if introspection:
            predictions = torch.argmax(logits, axis=-1)
            return fc_output, predictions

        return logits