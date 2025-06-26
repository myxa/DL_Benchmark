from random import uniform, randint
from torch import nn



class Transformer(nn.Module):
    """
    Transformer model for fMRI data.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]

    Hyperparameters expected in model_cfg:
        dropout: float
        head_hidden_size: int
        num_heads: int
        num_layers: int
    Data info expected in model_cfg:
        input_size: int - input_feature_size
        output_size: int - n_classes
    """

    def __init__(self, input_size, output_size, dropout, head_hidden_size, num_heads, num_layers):
        super().__init__()

        #input_size = model_cfg.input_size
        #output_size = model_cfg.output_size

        #dropout = model_cfg.dropout
        #head_hidden_size = model_cfg.head_hidden_size
        #num_heads = model_cfg.num_heads
        #num_layers = model_cfg.num_layers

        hidden_size = head_hidden_size * num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.input_embed = nn.Linear(input_size, hidden_size)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        input_embed = self.input_embed(x)

        tf_output = self.transformer(input_embed)
        tf_output = tf_output[:, 0, :]

        fc_output = self.fc(tf_output)
        return fc_output