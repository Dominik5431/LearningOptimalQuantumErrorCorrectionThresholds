import math
import time
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from positional_encodings.torch_encodings import PositionalEncoding2D, PositionalEncoding1D

"""
Defines the model used for estimating the QEC threshold.
"""


class Net(nn.Module):
    """
    Base class for a neural network model.
    Implements custom save(), load() and load_smaller_d(name),
    where some of the weights from a smaller distance are reused.
    """

    def __init__(self, name: str, cluster: bool = False):
        super(Net, self).__init__()
        self.name = name
        self.cluster = cluster
        self.pretraining = None

    def save(self):
        torch.save(self.state_dict(), "data/net_{0}.pt".format(self.name))
        return self

    def load(self):
        self.load_state_dict(torch.load("data/net_{0}.pt".format(self.name), map_location=torch.device('cpu')))
        return self

    def load_smaller_d(self, name: str):
        self.load_state_dict(torch.load("data/net_{0}.pt".format(name)))


class BiasMultiheadAttention(nn.MultiheadAttention):
    """
    Custom Multihead Attention that allows to add a bias to the attention weights.
    """

    def __init__(self, embed_dim, num_heads, batch_first, **kwargs):
        super().__init__(embed_dim, num_heads, batch_first=batch_first, **kwargs)
        self.attention_bias = None

    def set_attention_bias(self, matrix):
        """
        Set the relative positional encoding matrix.
        Args:
            matrix (torch.Tensor): Shape [seq_len, seq_len]
        """
        self.attention_bias = matrix

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Tensor | None = None,
                need_weights: bool = True,
                attn_mask: Tensor | None = None,
                average_attn_weights: bool = True,
                is_causal: bool = False) -> tuple[Tensor, Tensor | None]:
        # Standard attention computation
        attn_output, attn_weights = super().forward(
            query, key, value,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights,
            is_causal=is_causal
        )

        if self.attention_bias is not None:
            seq_len = query.size(0)
            # Use only relevant portion of the relative position matrix
            attention_bias_seq = self.attention_bias[:seq_len, :seq_len]
            attention_bias_seq = attention_bias_seq[attn_mask[:seq_len, :seq_len]]
            attention_bias_seq = attention_bias_seq.unsqueeze(0)  # Add batch dimension

            # Add relative positional matrix to the attention weights
            attn_weights = attn_weights + attention_bias_seq

            # Re-normalize attention weights
            attn_weights = F.softmax(attn_weights, dim=-1)

            # Recompute attention output using updated weights
            attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


class QecTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Custom Transformer Encoder Layer.
    Includes BiasMultiheadAttention, diluted convolutions and a learnable padding when scattering the data to 2d.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout: float, batch_first, scatter_indices,
                 convolutions: bool, **kwargs):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                         dropout=dropout, batch_first=batch_first, **kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.batch_first = batch_first

        self.self_attn = BiasMultiheadAttention(d_model, nhead, batch_first=self.batch_first, **kwargs)

        # Learnable padding vector for the positions where there is no stabilizer
        self.no_stab = nn.Parameter(torch.randn(d_model), requires_grad=True)

        # dilated convolutions

        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=True, dilation=1,
                               groups=self.self_attn.num_heads)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, bias=True, dilation=2,
                               groups=self.self_attn.num_heads)
        self.conv3 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=3, bias=True, dilation=3,
                               groups=self.self_attn.num_heads)
        self.scatter_indices = scatter_indices

        # Boolean whether convolutional layers should be applied or not
        self.convolutions = convolutions

    def set_attention_bias(self, matrix):
        """
        Pass the attention bias matrix to the attention layer.
        """
        self.self_attn.set_attention_bias(matrix)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tensor:
        # torch.mps.synchronize()
        # t1 = time.time()
        src = super().forward(src, src_mask, src_key_padding_mask, is_causal)
        # torch.mps.synchronize()
        # t2 = time.time()
        # print(f'Attention time: {t2 - t1}.6f s')
        if self.convolutions:
            src = self._conv_block(src)
        return src

    def _conv_block(self, src: Tensor) -> Tensor:
        d = int(math.sqrt(src.size(1) + 1))

        # torch.mps.synchronize()
        # t0 = time.time()
        src = scatter_to_2d(src, scatter_positions=self.scatter_indices, padding=self.no_stab, d=d, device=src.device)

        src = src.permute(0, 3, 1, 2)

        c1 = F.relu(self.conv1(src))
        c2 = F.relu(self.conv2(src))
        c3 = F.relu(self.conv3(src))

        src = 0.5 * (src + c1 + c2 + c3)

        src = src.permute(0, 2, 3, 1)

        src = collect_from_2d(src, d=d, device=src.device, scatter_positions=self.scatter_indices)

        # torch.mps.synchronize()
        # total_conv = time.time() - t0
        # print(f'Total conv time: {total_conv}')
        return src


class QecTransformerEncoder(nn.Module):
    """
        Work-around since PyTorch doesn't support deepcopy for custom parameters as the no-stab padding value.
        Create the nn.ModuleList manually by avoiding to use deepcopy
    """

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # Create new instances of encoder_layer instead of deepcopy
        self.layers = nn.ModuleList([encoder_layer.__class__(d_model=encoder_layer.d_model,
                                                             nhead=encoder_layer.nhead,
                                                             dim_feedforward=encoder_layer.dim_feedforward,
                                                             dropout=encoder_layer.dropout.p,
                                                             batch_first=encoder_layer.batch_first,
                                                             scatter_indices=encoder_layer.scatter_indices,
                                                             convolutions=encoder_layer.convolutions) for _ in
                                     range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Pass input through each encoder layer
        for i, layer in enumerate(self.layers):
            # torch.mps.synchronize()
            # start = time.time()
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
            # torch.mps.synchronize()
            # end = time.time() - start
            # print(f'layer {i} time: {end}.6f s')
        return src


class LearnablePositionalEncoding(nn.Module):
    """
    1D positional encoding. Deprecated since not well scalable for higher distances.
    """

    def __init__(self, n, d_model):
        # put n here as max_seq_len
        super().__init__()
        self.positional_embedding = nn.Embedding(n, d_model)

    def forward(self, x):
        return self.positional_embedding(torch.arange(x.size(1), device=x.device))


class LearnablePositionalQecEncoding2D(nn.Module):
    """
    Positional Encoding that respects the 2D structure of the input data.
    """

    def __init__(self, d, d_model, device):  # have also k here, in case for codes with more than 1 logical qubits
        super().__init__()
        self.pos_x = nn.Embedding(2 * d + 1, d_model).to(device)
        self.pos_y = nn.Embedding(2 * d + 1, d_model).to(device)
        self.stab = nn.Embedding(2, d_model).to(device)

        self.token_to_x = torch.zeros(d ** 2 - 1, dtype=torch.long).to(device)
        self.token_to_y = torch.zeros(d ** 2 - 1, dtype=torch.long).to(device)
        self.token_to_stab = torch.zeros(d ** 2 - 1, dtype=torch.long).to(device)

        z_idx = (d ** 2 - 1) // 2
        x_idx = (d ** 2 - 1) - 1

        self.token_to_stab[:z_idx] = 0
        self.token_to_stab[z_idx:] = 1

        for x in range(2, 2 * d, 4):
            self.token_to_x[x_idx] = x
            self.token_to_y[x_idx] = 0
            x_idx -= 1

        for y in range(2, 2 * d, 2):
            yi = y % 4
            xs = range(yi, 2 * d + yi // 2, 2)
            for i, x in enumerate(xs):
                if i % 2 == 0:
                    self.token_to_x[z_idx] = x
                    self.token_to_y[z_idx] = y
                    z_idx -= 1
                elif i % 2 == 1:
                    self.token_to_x[x_idx] = x
                    self.token_to_y[x_idx] = y
                    x_idx -= 1

        for x in range(4, 2 * d, 4):
            self.token_to_x[x_idx] = x
            self.token_to_y[x_idx] = 2 * d
            x_idx -= 1

        '''
        self.token_to_x[-2] = d
        self.token_to_y[-2] = d
        self.token_to_stab[-2] = 1  # X denoted as 1, Z as 0!

        self.token_to_x[-1] = d
        self.token_to_y[-1] = d
        self.token_to_stab[-1] = 0
        '''

    def forward(self, x):
        return (self.pos_x(self.token_to_x[:x.size(1)]) +
                self.pos_y(self.token_to_y[:x.size(1)]) +
                self.stab(self.token_to_stab[:x.size(1)]))


def scatter_to_2d(flat_tokens, scatter_positions, d, padding, device: Union[torch.device, str]):
    """
    Scatter flat tensor of syndrome measurements to 2d according to the positions of the stabilizers.
    """
    if padding is None:
        padding = -1 * torch.ones(1, device=device, dtype=torch.float)
    if len(flat_tokens.shape) != 3:
        flat_tokens = flat_tokens.unsqueeze(-1)
    assert flat_tokens.size(-1) == padding.size(-1)

    # Create the 2D result tensor
    length, height = d + 1, d + 1
    result = padding * flat_tokens.new_ones(flat_tokens.shape[0], height, length, flat_tokens.size(-1), device=device)

    # Batch indices
    batch_indices = torch.arange(flat_tokens.shape[0], device=flat_tokens.device).view(-1, 1)
    # Scatter the values into the result tensor
    flat_tokens = flat_tokens.to(dtype=torch.float)
    result[batch_indices, scatter_positions[:, 0], scatter_positions[:, 1]] = flat_tokens
    return result


def collect_from_2d(grid, d: int, device: Union[torch.device, str], dtype=torch.float32,
                    scatter_positions: torch.Tensor = None) -> torch.Tensor:
    """
    Collect the scattered syndromes from 2d grid according to the positions of the stabilizers to a 1d tesor.
    """
    # Batch indices
    batch_indices = torch.arange(grid.shape[0], device=grid.device).view(-1, 1)
    # Gather the values from the grid
    result = grid[batch_indices, scatter_positions[:, 0], scatter_positions[:, 1]]
    return result


class QecTransformer(Net):
    """
    Model to estimate the probability of a logical operator given the syndromes. Effectively realizes a neural network decoder.
    """

    def __init__(self, name, distance,
                 cluster=False,
                 pretraining=False,
                 rounds=2,
                 readout='conv',
                 biased_attention=None,
                 convolutions=None,
                 measurement_input=None,
                 input_residual=None,
                 penc_type='learnable',
                 **kwargs):
        super(QecTransformer, self).__init__(name, cluster)

        ''' Current changes '''
        if biased_attention is None:
            biased_attention = False
        if convolutions is None:
            convolutions = True
        if measurement_input is None:
            self.measurement_input = False
        else:
            self.measurement_input = measurement_input
        if input_residual is None:
            self.input_residual = True
        else:
            self.input_residual = input_residual

        ''' Hyperparameters '''
        self.n = kwargs['n']
        self.k = kwargs['k']
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']
        self.n_layers = kwargs['n_layers']
        self.n_heads = kwargs['n_heads']
        self.dropout = kwargs['dropout']
        self.device = kwargs['device']
        self.noise_model = kwargs['noise_model']
        self.rounds = rounds

        self.penc_type = penc_type
        self.readout = readout
        self.distance = distance

        self.pretraining = pretraining

        # Building the input representation  ->  binary input + start_token
        self.event_embedding = nn.Embedding(3,
                                            self.d_model)  # +1 for start token embedding, used when readout=='transformer-decoder'
        self.final_event_embedding = nn.Embedding(2, self.d_model)
        self.msmt_embedding = nn.Embedding(4 if readout == 'conv' else 5, self.d_model)
        if self.penc_type == 'learnable':
            self.positional_encoding = LearnablePositionalQecEncoding2D(self.distance, self.d_model, device=self.device)
        else:
            self.positional_encoding = PositionalEncoding2D(self.d_model)

        # Residual network to map to the input representation
        self.linear1 = nn.Linear(self.d_model, self.d_model)
        self.norm1 = nn.LayerNorm(self.d_model, eps=1e-5, bias=True, device=self.device)
        self.linear2 = nn.Linear(self.d_model, self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model, eps=1e-5, bias=True, device=self.device)

        # Attention bias
        self._precompute_scatter_indices()

        encoder_layer = QecTransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=self.dropout,
                                                   batch_first=True,
                                                   scatter_indices=self.scatter_indices,
                                                   convolutions=convolutions)

        '''
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=self.dropout,
                                                   batch_first=True)
        '''

        self.g_embedding = nn.Embedding(5, 1, device=self.device)
        self.stab_type_embedding = nn.Embedding(3, 1, device=self.device)
        self.manhattan_embedding = nn.Embedding(2 * 2 * self.distance, 1, device=self.device)

        # Attention bias embedding

        self.encoder = QecTransformerEncoder(encoder_layer, num_layers=self.n_layers)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        self.pretraining_token = nn.Parameter(torch.zeros(1, 1, self.d_model), requires_grad=True)

        # Readout network. Current standard: Convolutional readout network.
        if readout == 'transformer-decoder':
            if penc_type == 'learnable':
                self.log_pos = LearnablePositionalEncoding(2, self.d_model)  # logical positional encoding
            else:
                self.log_pos = PositionalEncoding1D(self.d_model)
            decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                                       nhead=self.n_heads,
                                                       dim_feedforward=self.d_ff,
                                                       dropout=self.dropout,
                                                       batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)
            self.fc_out = nn.Linear(self.d_model, 1)
            self.fc_out = nn.Linear(self.d_model, 1)
        elif readout == 'conv':
            self.conv_to_data = nn.Conv2d(self.d_model, self.d_model, kernel_size=2, stride=1, padding=0, bias=False,
                                          groups=self.n_heads)
            self.to_log = nn.Conv2d(self.d_model, self.d_model, kernel_size=self.distance, stride=1, padding=0,
                                    bias=True)
            self.linear_readout = nn.Linear(self.d_model, self.d_model)

            self.loss = nn.CrossEntropyLoss(reduction='none')
            self.fc_out = nn.Linear(self.d_model, 4)
        else:
            raise NotImplementedError(f"Readout network {readout} not implemented")

    def _precompute_scatter_indices(self):
        """
        Precomputes the positions of the stabilizers they get scattered to.
        :return: scatter positions
        """
        scatter = torch.zeros(self.distance ** 2 - 1, 2, device=self.device, dtype=torch.int)

        z_idx = (self.distance ** 2 - 1) // 2 - 1
        x_idx = (self.distance ** 2 - 1) - 1

        for x in range(1, self.distance, 2):
            scatter[x_idx, 0] = x
            scatter[x_idx, 1] = 0
            x_idx -= 1

        for y in range(1, self.distance):
            yi = y % 2
            xs = range(yi, self.distance + yi)
            for i, x in enumerate(xs):
                if i % 2 == 0:
                    scatter[z_idx, 0] = x
                    scatter[z_idx, 1] = y
                    z_idx -= 1
                elif i % 2 == 1:
                    scatter[x_idx, 0] = x
                    scatter[x_idx, 1] = y
                    x_idx -= 1

        for x in range(2, self.distance, 2):
            scatter[x_idx, 0] = x
            scatter[x_idx, 1] = self.distance
            x_idx -= 1

        self.scatter_indices = scatter

    '''
    def set_pretrain(self, pretrain):
        """
        Sets pretraining mode.
        """
        self.pretraining = pretrain
        if pretrain:
            self.con
    '''

    def set_pretrain(self, pretrain):
        """
        Sets pretraining mode.
        """
        self.pretraining = pretrain
        for layer in self.encoder.layers:
            layer.convolutions = not pretrain

    def stab_relation(self, i, j):
        """
        Returns relation between two stabilizers i and j.
        :return: 0 if ZZ, 1 if XZ or ZX, 2 if XX
        """
        n_stab_single = (self.distance ** 2 - 1) // 2
        if i < n_stab_single and j < n_stab_single:
            return torch.tensor(0, device=self.device)
        elif i < n_stab_single <= j or j < n_stab_single <= i:
            return torch.tensor(1, device=self.device)
        elif i >= n_stab_single and j >= n_stab_single:
            return torch.tensor(2, device=self.device)

    def input_repr(self, x, final_round=False):
        if self.measurement_input:
            event = self.event_embedding(x[:, :, 0])
            msmt = self.msmt_embedding(x[:, :, 1] * 2 + x[:, :, 2])  # converting binary measurements to decimal values
            position = self.positional_encoding(x[:, :, 0])
            x = event + msmt + position
        else:
            if final_round:
                event = self.final_event_embedding(x)
            else:
                event = self.event_embedding(x)

            if self.penc_type == 'learnable':
                position = self.positional_encoding(x)
                x = event + position
            else:
                event = scatter_to_2d(event, scatter_positions=self.scatter_indices, padding=torch.zeros(self.d_model),
                                      d=self.distance,
                                      device=self.device)
                position = self.positional_encoding(event)
                x = event + position
                x = collect_from_2d(x, d=self.distance, device=self.device, scatter_positions=self.scatter_indices)
        return x

    def logical_input_repr(self, x):
        if self.measurement_input:
            event = self.event_embedding(x[:, :, 0])
            msmt = self.msmt_embedding(x[:, :, 1] * 2 + x[:, :, 2])  # converting binary measurements to decimal values
            position = self.log_pos(x[:, :, 0])
            x = event + msmt + position
        else:
            event = self.event_embedding(x)
            position = self.log_pos(event)
            x = event + position
        return x

    def res_net(self, x, scaling_factor=1 / math.sqrt(2)):
        identity = x

        out = self.norm1(F.relu(self.linear1(x)))
        out = self.norm2(F.relu(self.linear2(out)))

        out += identity
        out = F.relu(out * scaling_factor)

        return out

    def decode(self, x: torch.Tensor, *args) -> torch.Tensor:
        if self.readout == 'conv':
            x = scatter_to_2d(x, scatter_positions=self.scatter_indices, d=self.distance,
                              padding=torch.zeros(self.d_model),
                              device=self.device)

            x = self.conv_to_data(x.permute(0, 3, 1, 2))

            proj = self.to_log(x).permute(0, 2, 3, 1)

            identity = proj
            out = identity + F.relu(self.linear_readout(proj))
        else:
            log = args[0]
            seq_len = log.size(1)
            mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0)
            mask = mask.masked_fill(mask == 0, float('-inf'))
            mask = mask.masked_fill(mask == 1, 0.0)
            mask = mask.to(self.device)

            out = self.decoder(tgt=log, memory=x, tgt_mask=mask)
        return out

    def forward(self, x, *args):
        # Prepare syndrome input
        if self.input_residual:
            x = self.input_repr(x)
            x = self.res_net(x)
        else:
            if self.measurement_input:
                x = F.relu(self.event_embedding(x[:, :, 0])) + F.relu(self.event_embedding(x[:, :, 1] * 2 + x[:, :, 2]))
                x = x + self.positional_encoding(x[:, :, 0])
            else:
                x = F.relu(self.event_embedding(x))
                x = x + self.positional_encoding(x)

        # No mask, all-to-all attention between syndromes
        # No decoding history --> solely x as input, no decoding state being added

        # torch.mps.synchronize()
        # t0 = time.time()
        x = self.encoder(x)
        # encoder_t = time.time() - t0
        # print(f'Encoder time: {encoder_t:.6f}s')

        if args:
            log = args[0]

            # Prepare logical input
            if self.input_residual:
                log = self.logical_input_repr(log)
                log = self.res_net(log)
            else:
                log = F.relu(self.event_embedding(log))
                log = log + self.log_pos(log)

            x = self.decode(x, log)
            x = self.fc_out(x)
            x = torch.sigmoid(x)
            x = x.squeeze(2)
        else:
            x = self.decode(x)
            x = x.squeeze(2)
            x = self.fc_out(x)
            x = x.squeeze(1)  # nn.cross_entropy_loss expects raw inputs, applies softmax internally
        return x

    def encoder_forward(self, x, mask=None):
        if self.input_residual:
            x = self.input_repr(x)
            x = self.res_net(x)
        else:
            if self.measurement_input:
                x = F.relu(self.event_embedding(x[:, :, 0])) + F.relu(self.event_embedding(x[:, :, 1] * 2 + x[:, :, 2]))
                x = x + self.positional_encoding(x[:, :, 0])
            else:
                x = F.relu(self.event_embedding(x))
                x = x + self.positional_encoding(x)

        # No mask, all-to-all attention between syndromes
        # No decoding history --> solely x as input, no decoding state being added

        # torch.mps.synchronize()
        # t0 = time.time()
        x = self.encoder(x, src_mask=mask)
        return x

    def pretraining_forward(self, x):
        # Prepare syndrome input
        if self.input_residual:
            x = self.input_repr(x)
            x = self.res_net(x)
        else:
            if self.measurement_input:
                x = F.relu(self.event_embedding(x[:, :, 0])) + F.relu(self.event_embedding(x[:, :, 1] * 2 + x[:, :, 2]))
                x = x + self.positional_encoding(x[:, :, 0])
            else:
                x = F.relu(self.event_embedding(x))
                x = x + self.positional_encoding(x)

        start_token = self.pretraining_token.expand(x.size(0), -1, -1)
        x = torch.cat((start_token, x[:, :-1]), dim=1)

        # Here use mask! Next stabilizer prediction
        seq_len = (x.size(1))
        mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        mask = mask.to(self.device)

        x = self.encoder(x, src_mask=mask)
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        return x.squeeze(2)

    def log_prob(self, x):
        """
        Using the log probability as loss function. It is calculated here.
        :param x: syndromes + logicals
        :return: log_prob

        Allows for pretraining in form of next stabilizer prediction.
        """
        epsilon = 1e-9

        if not self.measurement_input:
            try:
                x = x[:, :, 0]
            except IndexError:
                pass

        if self.measurement_input:
            syndrome = x[:, :self.distance ** 2 - 1, :]
            logical = x[:, self.distance ** 2 - 1:, 0]  # target tokens
        else:
            syndrome = x[:, :self.distance ** 2 - 1]
            logical = x[:, self.distance ** 2 - 1:]

        if self.pretraining:
            # Next stabilizer prediction
            # don't use start token, start next stabilizer prediction from second stabilizer on
            # loss only calculated on prediction of stabilizers 2 to n-k.
            x_hat = self.pretraining_forward(syndrome)
            if self.measurement_input:
                syndrome = syndrome[:, :, 0]
            log_prob = torch.log(x_hat + epsilon) * syndrome + torch.log(1 - x_hat + epsilon) * (
                    1 - syndrome)
            log_prob = log_prob.sum(dim=1)
            sequence_length = syndrome.size(1)
        else:
            if self.readout == 'transformer-decoder':
                # Start token for transformer-decoder necessary:
                start_token_value = 2
                if self.measurement_input:
                    start_token = torch.full((x.size(0), 1, x.size(2)), start_token_value, dtype=torch.long,
                                             device=self.device)
                    logical_in = torch.cat((start_token, x[:, self.distance ** 2 - 1:-1, :]), dim=1).to(self.device)
                else:
                    start_token = torch.full((x.size(0), 1), start_token_value, dtype=torch.long,
                                             device=self.device)
                    logical_in = torch.cat((start_token, x[:, self.distance ** 2 - 1:-1]), dim=1).to(self.device)
                x_hat = self.forward(syndrome, logical_in)

                log_prob = torch.log(x_hat + epsilon) * logical + torch.log(1 - x_hat + epsilon) * (
                        1 - logical)

                sequence_length = logical.size(1)

                log_prob = log_prob.sum(dim=1)
            else:
                x_hat = self.forward(syndrome)

                powers_of_two = torch.tensor([2, 1], device=self.device, dtype=torch.float)
                target = torch.matmul(logical.to(torch.float), powers_of_two)
                log_prob = - self.loss(x_hat, target)

                sequence_length = 1

        return log_prob / sequence_length

    def predict_logical(self, syndrome):
        """
        Used during inference.
        :param syndrome: Measurement syndromes
        :return: Probability of logical operators
        """
        with torch.no_grad():
            if self.readout == 'transformer-decoder':
                logical = torch.zeros(syndrome.size(0), 1).to(self.device)

                if self.measurement_input:
                    start_token_value = 2
                    start_token = torch.full((syndrome.size(0), 1, syndrome.size(2)), start_token_value,
                                             dtype=torch.long,
                                             device=self.device)
                    logical_in = torch.cat((start_token, syndrome[:, self.distance ** 2 - 1:, :]), dim=1).to(
                        self.device)
                    syndrome_in = syndrome[:, :self.distance ** 2 - 1, :]
                else:
                    syndrome = syndrome[:, :, 0]
                    start_token_value = 2
                    start_token = torch.full((syndrome.size(0), 1), start_token_value, dtype=torch.long,
                                             device=self.device)
                    syndrome = torch.cat((syndrome, start_token), dim=1).to(self.device)

                    syndrome_in = syndrome[:, :self.distance ** 2 - 1]
                    logical_in = syndrome[:, self.distance ** 2 - 1:]
                for i in range(1):
                    conditional = self.forward(syndrome_in, logical_in)

                    # conditional = torch.sigmoid(logits)
                    if len(conditional.shape) < 2:
                        conditional = conditional.unsqueeze(0)
                    # r = torch.as_tensor(np.random.rand(syndrome.size(0)), dtype=torch.float32, device=self.device)
                    # syndrome = torch.cat((syndrome, 1*(r < logical[:, i]).unsqueeze(1)), dim=1)
                    logical[:, i] = conditional[:, -1]
                    # x[:, s + i] = torch.floor(2 * conditional[:, s + i])
                    # x[:, s + i] = conditional[:, s + i]
            elif self.readout == 'conv':
                if not self.measurement_input:
                    syndrome = syndrome[:, :, 0]

                logical = self.forward(syndrome).to(self.device)
                logical = torch.softmax(logical, dim=1)
        return logical.squeeze()


if __name__ == '__main__':
    # For testing
    distance = 3
    model_dict = {
        'n': distance ** 2,
        'k': 1,
        'distance': distance,
        'd_model': 32,
        'd_ff': 32,
        'n_layers': 3,
        'n_heads': 8,
        'device': 'cpu',
        'dropout': 0.2,
        'max_seq_len': distance ** 2 - 1 + 2 * distance,
        'noise_model': 'depolarizing'
    }
    model = QecTransformer(name='test-transformer', **model_dict)
    s = torch.tensor(
        [[0.1, 0.2, 0.3], [1.1, 0.2, 3.3], [0.1, 5.2, 0.3], [3.1, 0.2, 0.3], [0.4, 0.2, 9.3], [0.1, 0.2, 0.03],
         [0.11, 0.2, 0.3], [0.51, 0.28, 0.3]])
    s = torch.unsqueeze(s, 0)
    s_2d = scatter_to_2d(s, scatter_positions=model.scatter_indices, padding=torch.tensor([0., 0., 0.]), d=3,
                         device='cpu')
    print(s_2d)
    s_recon = collect_from_2d(s_2d, d=3, device='cpu', scatter_positions=model.scatter_indices)
    print(s_recon)
    assert torch.allclose(s, s_recon)
