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


class AttnMLP(nn.Module):
    """
    MLP model for fMRI data.
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

        #input_size = #model_cfg.input_size
        #output_size = #model_cfg.output_size
        #dropout = #model_cfg.dropout
        #hidden_size = #model_cfg.hidden_size
        #num_layers = #model_cfg.num_layers

        # input block
        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        # inter blocks
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )

        self.fc = nn.Sequential(*layers)

        # attention block
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # output block
        self.clf = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size),
        )

    def get_attention(self, outputs):
        bs, tl, fs = outputs.shape  # [batch_size, time_length, hidden_feature_size]

        # get average-over-time mean outputs and prepare them for concatenation
        outputs_mean = outputs.mean(1).unsqueeze(1).expand(-1, tl, -1)

        attn_input = torch.cat((outputs, outputs_mean), dim=2)
        weights = self.attn(attn_input.view(-1, 2 * fs))
        weights = weights.view(bs, tl)
        return weights

    def forward(self, x: torch.Tensor):
        bs, tl, fs = x.shape  # [batch_size, time_length, input_feature_size]

        # pass the data through the MLP
        fc_output = self.fc(x.view(-1, fs))
        fc_output = fc_output.view(bs, tl, -1)

        # get attention weights
        weights = self.get_attention(fc_output)

        # apply attention and pass the data to the calssifier
        weights = weights.unsqueeze(-1)
        weighted_fc_output = torch.sum(fc_output * weights, dim=1)

        logits = self.clf(weighted_fc_output)
        return logits