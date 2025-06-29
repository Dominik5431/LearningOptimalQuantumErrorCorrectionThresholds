import math
from typing import Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# rom functorch.einops import rearrange
from positional_encodings.torch_encodings import PositionalEncoding2D, PositionalEncoding1D, PositionalEncoding3D
from torch import Tensor
from torch.nn import MultiheadAttention

from .qectransformer import Net, QecTransformer, scatter_to_2d, collect_from_2d


class QecSpaceTimeEncoderLayer(nn.TransformerEncoderLayer):
    """
    Custom Transformer Encoder Layer.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward, dropout: float, batch_first, scatter_indices,
                 convolutions: bool, rounds, layer_norm_eps, bias, device, **kwargs):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                         dropout=dropout, batch_first=batch_first, **kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.batch_first = batch_first
        self.rounds = rounds
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias
        self.device = device

        self.ff_im = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.time_norm = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)
        self.space_norm = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)
        self.norm1 = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)
        self.norm2 = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tensor:
        assert not self.norm_first

        batch_size, num_stab, _ = src.size()
        num_stab = num_stab // self.rounds

        xt = src
        # xt = rearrange(xt, 'b (r n) d -> (b n) r d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        xt = xt.view(batch_size, self.rounds, num_stab, self.d_model)  # Shape: (b, r, n, d)
        xt = xt.permute(0, 2, 1, 3).contiguous()  # Shape: (b, n, r, d)
        xt = xt.view(batch_size * num_stab, self.rounds, self.d_model)  # Shape: (b*n, r, d)
        res_t = self.time_norm(self._sa_block(xt, src_mask, src_key_padding_mask, is_causal=is_causal) + xt)
        # res_t = rearrange(res_t, '(b n) r d -> b (r n) d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        res_t = res_t.view(batch_size, num_stab, self.rounds, self.d_model)  # Shape: (b, n, r, d)
        res_t = res_t.permute(0, 2, 1, 3).contiguous()  # .view(batch_size, self.rounds * num_stab, self.d_model)
        res_t = self.norm1(res_t + self.ff_im(res_t))
        # Shape: (b, r, n, d)

        # Spatial expert
        xs = res_t  # syndrome
        # xs = rearrange(xs, 'b (r n) d -> (b r) n d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        # xs = xs.view(batch_size, self.rounds, num_stab, self.d_model)  # Shape: (b, r, n, d)
        xs = xs.view(batch_size * self.rounds, num_stab, self.d_model)  # Shape: (b*r, n, d)
        res_s = self.space_norm(self._sa_block(xs, src_mask, src_key_padding_mask, is_causal=is_causal) + xs)
        # res_s = rearrange(xs, '(b r) n d -> b (r n) d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        res_s = res_s.view(batch_size, self.rounds, num_stab, self.d_model)  # Shape: (b, r, n, d)
        memory = res_s.view(batch_size, self.rounds * num_stab, self.d_model)  # Shape: (b, r*n, d)
        memory = self.norm2(memory + self._ff_block(memory))

        # if self.convolutions:
        #     src = self._conv_block(src)
        return memory

    '''
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
        '''


class QecSpaceTimeEncoder(nn.Module):
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
                                                             scatter_indices=None,
                                                             convolutions=False,
                                                             rounds=encoder_layer.rounds,
                                                             layer_norm_eps=encoder_layer.layer_norm_eps,
                                                             bias=encoder_layer.bias,
                                                             device=encoder_layer.device) for _ in
                                     range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Pass input through each encoder layer
        for i, layer in enumerate(self.layers):
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        return src


def scatter_to_3d(flat_tokens, scatter_positions, d, padding, device: Union[torch.device, str]):
    """
    Scatter flat tensor of syndrome measurements to dd according to the positions of the stabilizers.
    """
    if padding is None:
        padding = -1 * torch.ones(1, device=device, dtype=torch.float)
    if len(flat_tokens.shape) != 3:
        flat_tokens = flat_tokens.unsqueeze(-1)
    assert flat_tokens.size(-1) == padding.size(-1)

    # Create the 3D result tensor
    length, height, rounds = d + 1, d + 1, d
    result = padding * flat_tokens.new_ones(flat_tokens.shape[0], rounds, height, length, flat_tokens.size(-1), device=device)

    # Batch indices
    batch_indices = torch.arange(flat_tokens.shape[0], device=flat_tokens.device).view(-1, 1)
    
    flat_tokens = flat_tokens.to(dtype=torch.float)
    # Scatter the values into the result tensor
    result[batch_indices, scatter_positions[:, 2], scatter_positions[:, 0], scatter_positions[:, 1]] = flat_tokens
    return result


def collect_from_3d(grid, d: int, device: Union[torch.device, str], dtype=torch.float32,
                    scatter_positions: torch.Tensor = None) -> torch.Tensor:
    """
    Collect the scattered syndromes from 3d grid according to the positions of the stabilizers to a 1d tesor.
    """
    # Batch indices
    batch_indices = torch.arange(grid.shape[0], device=grid.device).view(-1, 1)
    
    # Gather the values from the grid
    result = grid[batch_indices, scatter_positions[:, 2], scatter_positions[:, 0], scatter_positions[:, 1]]
    return result


class RQecTransformer(Net):
    def __init__(self, name, distance,
                 cluster=False,
                 pretraining=False,
                 readout='transformer-decoder',
                 penc_type='fixed',
                 every_round=False,
                 dropout: float = 0.0,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True,
                 **kwargs):
        super().__init__(name, cluster)

        ''' Hyperparameters '''
        self.n = kwargs['n']
        self.k = kwargs['k']
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']
        self.n_layers = kwargs['n_layers']
        self.n_heads = kwargs['n_heads']
        self.device = kwargs['device']
        self.noise_model = kwargs['noise_model']

        self.penc_type = penc_type
        self.readout = readout
        self.every_round = every_round
        self.distance = distance
        self.rounds = self.distance
        self.pretraining = pretraining

        # Input representation
        # self.event_embedding = nn.Embedding(3, self.d_model)
        # self.final_event_embedding = nn.Embedding(2, self.d_model)
        self.positional_encoding = PositionalEncoding3D(self.d_model)

        # Use convolutional embedding instead
        # '''
        self.event_embedding = nn.Conv2d(1, self.d_model, kernel_size=3, stride=1, padding=0,
                                         bias=True, device=self.device)

        self.final_event_embedding = nn.Conv2d(1, self.d_model, kernel_size=3, stride=1, padding=0,  # padding with constant value later manually
                                               bias=True, device=self.device)
        # '''
        # Input Res Net
        self.linear1 = nn.Linear(self.d_model, self.d_model)
        self.norm = nn.LayerNorm(self.d_model, eps=1e-5, bias=True, device=self.device)
        self.linear2 = nn.Linear(self.d_model, self.d_model)

        self.encoder_layer = QecSpaceTimeEncoderLayer(d_model=self.d_model,
                                                      nhead=self.n_heads,
                                                      dim_feedforward=self.d_ff,
                                                      dropout=dropout,
                                                      batch_first=True,
                                                      scatter_indices=None,
                                                      convolutions=False,
                                                      rounds=self.rounds,
                                                      layer_norm_eps=layer_norm_eps,
                                                      bias=bias,
                                                      device=self.device)
        self.encoder = QecSpaceTimeEncoder(self.encoder_layer, self.n_layers)

        '''  this for ph15!!
        # Spatial and temporal experts
        self.time_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                             nhead=self.n_heads,
                                                             dropout=dropout,
                                                             dim_feedforward=self.d_ff,
                                                             batch_first=True)
        self.time_encoder = nn.TransformerEncoder(self.time_encoder_layer,
                                                  num_layers=self.n_layers)

        self.space_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                              nhead=self.n_heads,
                                                              dropout=dropout,
                                                              dim_feedforward=self.d_ff,
                                                              batch_first=True)
        self.space_encoder = nn.TransformerEncoder(self.space_encoder_layer,
                                                   num_layers=self.n_layers)

        self.time_norm = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)
        self.space_norm = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)
        '''
        '''
        # Space-time cross attention
        self.multihead_attn = MultiheadAttention(self.d_model, self.n_heads, dropout=dropout, batch_first=batch_first,
                                                 bias=bias, device=self.device)
        # Implementation of Feedforward model
        self.activation = activation
        self.linear1_s = nn.Linear(self.d_model, self.d_ff, bias=bias, device=self.device)
        self.dropout2_s = nn.Dropout(dropout)
        self.linear2_s = nn.Linear(self.d_model, self.d_ff, bias=bias, device=self.device)
        self.dropout3_s = nn.Dropout(dropout)

        self.linear1_t = nn.Linear(self.d_model, self.d_ff, bias=bias, device=self.device)
        self.dropout2_t = nn.Dropout(dropout)
        self.linear2_t = nn.Linear(self.d_model, self.d_ff, bias=bias, device=self.device)
        self.dropout3_t = nn.Dropout(dropout)

        self.norm_first = norm_first
        self.norm2_s = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)
        self.norm3_s = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)
        self.norm2_t = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)
        self.norm3_t = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)

        self.dropout_s = nn.Dropout(dropout)
        self.dropout_t = nn.Dropout(dropout)

        # Merging space-time representation for decoder
        self.linear_st = nn.Linear(2 * self.d_model, self.d_model, bias=bias, device=self.device)
        '''
        # Decoder part
        self.log_pos = PositionalEncoding1D(self.d_model)
        self.log_embedding = nn.Embedding(3, self.d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)
        self.fc_out = nn.Linear(self.d_model, 1)

        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._precompute_scatter_indices()

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

        repeated_scatter = scatter.repeat((self.rounds, 1))
        indices = torch.arange(self.rounds, device=self.device).repeat_interleave(self.distance ** 2 - 1).unsqueeze(1)
        expanded_scatter = torch.hstack((repeated_scatter, indices))
        self.scatter_indices = expanded_scatter
        # self.scatter_indices = scatter

    '''
    # multi-head attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                   is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return x

    # feed forward block
    def _ff_block_t(self, x: Tensor) -> Tensor:
        x = self.linear2_t(self.dropout2_t(self.activation(self.linear1_t(x))))
        return self.dropout3_t(x)

    def _ff_block_s(self, x: Tensor) -> Tensor:
        x = self.linear2_s(self.dropout2_s(self.activation(self.linear1_s(x))))
        return self.dropout3_s(x)
    '''

    def _input_repr(self, x):
        batch_size, num_stab = x.size()
        # xr = rearrange(x, 'b (r n) -> b r n', b=batch_size, r=self.rounds, n=num_stab//self.rounds)
        events = scatter_to_3d(x, self.scatter_indices, d=self.distance, padding=None, device=self.device).to(torch.float)
        events_embedded = torch.empty(batch_size, self.rounds, events.size(2), events.size(3), self.d_model, device=self.device, dtype=torch.float)

        for i in range(self.rounds):
            r = events[:, i]
            r = r.permute(0, 3, 1, 2)
            if i == self.rounds - 1:
                r = F.pad(r, pad=(1, 1, 1, 1), mode="constant", value=-1)
                event = self.final_event_embedding(r)
            else:
                r = F.pad(r, pad=(1, 1, 1, 1), mode="constant", value=-1)
                event = self.event_embedding(r)
            event = event.permute(0, 2, 3, 1)
            events_embedded[:, i, :, :, :] = event

        position = self.positional_encoding(events_embedded)
        embedded = events + position

        x_in = collect_from_3d(embedded, d=self.distance, device=self.device,
                               scatter_positions=self.scatter_indices)
        x_in = x_in.view(batch_size * self.rounds, (num_stab // self.rounds), self.d_model)
        x_in = self._res_net(x_in)
        x_in = x_in.view(batch_size, self.rounds * (num_stab // self.rounds), self.d_model)
        return x_in

    def _logical_input_repr(self, x):
        event = self.log_embedding(x)
        position = self.log_pos(event)
        x = event + position
        return x

    def _res_net(self, x):
        identity = x

        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))

        out = self.norm(out + identity)
        out = F.relu(out)

        return out

    def forward(self, syndrome, log):
        """
        syndrome: Shape (B, R * N), R: rounds, N: num syndromes
        """
        batch_size, num_stab = syndrome.size()
        num_stab = num_stab // self.rounds

        # Embedding
        syndrome = self._input_repr(syndrome)

        memory = self.encoder(syndrome)
        ### ph15 - first spatial attention, than temporal attention ###
        '''
        # Temporal expert
        xt = syndrome
        # xt = rearrange(xt, 'b (r n) d -> (b n) r d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        xt = xt.view(batch_size, self.rounds, num_stab, self.d_model)  # Shape: (b, r, n, d)
        xt = xt.permute(0, 2, 1, 3).contiguous()  # Shape: (b, n, r, d)
        xt = xt.view(batch_size * num_stab, self.rounds, self.d_model)  # Shape: (b*n, r, d)
        res_t = self.time_norm(self.time_encoder(xt) + xt)
        # res_t = rearrange(res_t, '(b n) r d -> b (r n) d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        res_t = res_t.view(batch_size, num_stab, self.rounds, self.d_model)  # Shape: (b, n, r, d)
        res_t = res_t.permute(0, 2, 1, 3).contiguous()  # .view(batch_size, self.rounds * num_stab, self.d_model)
        # Shape: (b, r*n, d)

        # Spatial expert
        xs = res_t  # syndrome
        # xs = rearrange(xs, 'b (r n) d -> (b r) n d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        # xs = xs.view(batch_size, self.rounds, num_stab, self.d_model)  # Shape: (b, r, n, d)
        xs = xs.view(batch_size * self.rounds, num_stab, self.d_model)  # Shape: (b*r, n, d)
        res_s = self.space_norm(self.space_encoder(xs) + xs)
        # res_s = rearrange(xs, '(b r) n d -> b (r n) d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        res_s = res_s.view(batch_size, self.rounds, num_stab, self.d_model)  # Shape: (b, r, n, d)
        memory = res_s.view(batch_size, self.rounds * num_stab, self.d_model)  # Shape: (b, r*n, d)
        '''
        '''
        # cross-attention space-time exchange
        # space information to temporal expert
        # s2t = rearrange(res_t, '(b n) r d -> (b r) n d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        s2t = res_t.view(batch_size, num_stab, self.rounds, self.d_model)  # Shape: (b, n, r, d)
        s2t = s2t.permute(0, 2, 1, 3).contiguous()  # Shape: (b, r, n, d)
        s2t = s2t.view(batch_size * self.rounds, num_stab, self.d_model)  # Shape: (b*r, n, d)
        s2t = self.norm2_t(s2t + self.dropout_t(self._mha_block(s2t, res_s)))
        s2t = self.norm3_t(s2t + self._ff_block_t(s2t))
        # s2t = rearrange(s2t, '(b r) n d -> b (r n) d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        s2t = s2t.reshape(batch_size, self.rounds * num_stab, self.d_model)

        # time information to spatial expert
        # t2s = rearrange(res_s, '(b r) n d -> (b n) r d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        t2s = res_s.view(batch_size, self.rounds, num_stab, self.d_model)  # Shape: (b, n, r, d)
        t2s = t2s.permute(0, 2, 1, 3).contiguous()  # Shape: (b, n, r, d)
        t2s = t2s.view(batch_size * num_stab, self.rounds, self.d_model)  # Shape: (b*r, n, d)

        t2s = self.norm2_s(t2s + self.dropout_s(self._mha_block(t2s, res_t)))
        t2s = self.norm3_s(t2s + self._ff_block_s(t2s))
        # t2s = rearrange(t2s, '(b n) r d -> b (r n) d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        t2s = t2s.reshape(batch_size, self.rounds * num_stab, self.d_model)

        memory = self.linear_st(torch.cat([t2s, s2t], dim=-1))
        '''
        log = self._logical_input_repr(log)
        log = self._res_net(log)

        seq_len = log.size(1)
        mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0)
        mask = mask.masked_fill(mask == 0, -1e12)
        mask = mask.masked_fill(mask == 1, 0.0)
        mask = mask.to(self.device)

        out = self.decoder(tgt=log, memory=memory, tgt_mask=mask)
        out = self.fc_out(out)
        # out = torch.sigmoid(out)  # shape (B, 2, 1)
        out = out.squeeze(2)

        return out

    def log_prob(self, x, include_mi=False):
        """
            Using the log probability as loss function. It is calculated here.
            :param x: syndromes + logicals
            :param include_mi: include mutual information loss
            :return: log_prob

            Allows for pretraining in form of next stabilizer prediction.
        """
        n_syndromes = (self.distance ** 2 - 1) * self.rounds

        syndrome = x[:, :n_syndromes]
        logical = x[:, n_syndromes:]

        # syndrome_rounds = torch.reshape(syndrome, (x.size(0), self.rounds, -1))

        # Start token for transformer-decoder necessary:
        start_token_value = 2
        start_token = torch.full((x.size(0), 1), start_token_value, dtype=torch.int,
                                 device=self.device)

        logical_in = torch.cat((start_token, logical[:, :-1]), dim=1).to(self.device)

        #print(syndrome_rounds)
        #print(logical_rounds)

        x_hat = self.forward(syndrome, logical_in)  # outputs logits

        # log_prob = torch.log(x_hat + epsilon) * logical_rounds + torch.log(1 - x_hat + epsilon) * (
        #         1 - logical_rounds)

        logical = logical.to(x_hat.dtype)
        log_prob = - self.loss(x_hat, logical)

        sequence_length = logical.size(1)

        # alpha = 0.25
        # gamma = 2
        # probs = torch.exp(log_prob)
        # log_prob = alpha * (1 - probs) ** gamma * log_prob
        # print(torch.mean(log_prob.sum(dim=1), dim=0) / logical_rounds.size(1))

        log_prob = log_prob.sum(dim=1)

        if include_mi:
            return log_prob / sequence_length, self.mi_loss(x_hat)
        return log_prob / sequence_length

    def mi_loss(self, p_out):  # p(y | x) - decoder predictions
        """Encourage decoder output to be different from unconditional probability distribution."""
        q_out = torch.mean(p_out, dim=0, keepdim=True).detach()  # p(y) - empirical marginal
        loss = torch.mean(torch.sum(p_out * torch.log(p_out / q_out), dim=1), dim=0)  # Reverse KL
        return -1 * loss  # We maximize MI by making the loss negative

    def predict_logical(self, syndrome):
        """
        Used during inference.
        :param syndrome: Measurement syndromes
        :return: Probability of logical operators
        """
        n_syndromes = (self.distance ** 2 - 1) * (self.rounds)
        previous_every_round = self.every_round
        self.every_round = False

        with torch.no_grad():
            logical = torch.zeros(syndrome.size(0), 1).to(self.device)

            # try:
            #     syndrome = syndrome[:, :, 0]
            # except IndexError:
            #     pass

            start_token_value = 2
            start_token = torch.full((syndrome.size(0), 1), start_token_value, dtype=torch.int,
                                     device=self.device)

            syndrome_in = syndrome[:, :n_syndromes]
            logical_in = syndrome[:, n_syndromes:]

            logical_in = torch.cat((start_token, logical_in), dim=1).to(self.device)

            for i in range(1):
                conditional = torch.sigmoid(self.forward(syndrome_in, logical_in))

                # conditional = torch.sigmoid(logits)
                if len(conditional.shape) < 2:
                    conditional = conditional.unsqueeze(0)
                # r = torch.as_tensor(np.random.rand(syndrome.size(0)), dtype=torch.float32, device=self.device)
                # syndrome = torch.cat((syndrome, 1*(r < logical[:, i]).unsqueeze(1)), dim=1)
                logical[:, i] = conditional[:, -1]
                # x[:, s + i] = torch.floor(2 * conditional[:, s + i])
                # x[:, s + i] = conditional[:, s + i]

        self.every_round = previous_every_round
        return logical.squeeze()


class RQecTransformer2(Net):
    def __init__(self, name, distance,
                 cluster=False,
                 pretraining=False,
                 readout='conv',
                 penc_type='fixed',
                 every_round=False,
                 dropout: float = 0.,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True,
                 **kwargs):
        super().__init__(name, cluster)

        ''' Hyperparameters '''
        self.n = kwargs['n']
        self.k = kwargs['k']
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']
        self.n_layers = kwargs['n_layers']
        self.n_heads = kwargs['n_heads']
        self.device = kwargs['device']
        self.noise_model = kwargs['noise_model']

        self.penc_type = penc_type
        self.readout = readout
        self.every_round = every_round
        self.distance = distance
        self.rounds = self.distance
        self.pretraining = pretraining

        # Input representation
        self.event_embedding = nn.Embedding(3, self.d_model)
        self.final_event_embedding = nn.Embedding(2, self.d_model)
        self.positional_encoding = PositionalEncoding3D(self.d_model)

        # Input Res Net
        self.linear1 = nn.Linear(self.d_model, self.d_model)
        self.norm = nn.LayerNorm(self.d_model, eps=1e-5, bias=True, device=self.device)
        self.linear2 = nn.Linear(self.d_model, self.d_model)

        # Spatial and temporal experts
        self.time_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                             nhead=self.n_heads,
                                                             dropout=dropout,
                                                             dim_feedforward=self.d_ff,
                                                             batch_first=True)
        self.time_encoder = nn.TransformerEncoder(self.time_encoder_layer,
                                                  num_layers=self.n_layers)

        self.space_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                              nhead=self.n_heads,
                                                              dropout=dropout,
                                                              dim_feedforward=self.d_ff,
                                                              batch_first=True)
        self.space_encoder = nn.TransformerEncoder(self.space_encoder_layer,
                                                   num_layers=self.n_layers)

        self.time_norm = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)
        self.space_norm = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)

        # Space-time cross attention
        self.multihead_attn = MultiheadAttention(self.d_model, self.n_heads, dropout=dropout, batch_first=True,
                                                 bias=bias, device=self.device)
        # Implementation of Feedforward model
        self.activation = activation
        self.linear1_s = nn.Linear(self.d_model, self.d_ff, bias=bias, device=self.device)
        self.dropout2_s = nn.Dropout(dropout)
        self.linear2_s = nn.Linear(self.d_model, self.d_ff, bias=bias, device=self.device)
        self.dropout3_s = nn.Dropout(dropout)

        self.linear1_t = nn.Linear(self.d_model, self.d_ff, bias=bias, device=self.device)
        self.dropout2_t = nn.Dropout(dropout)
        self.linear2_t = nn.Linear(self.d_model, self.d_ff, bias=bias, device=self.device)
        self.dropout3_t = nn.Dropout(dropout)

        self.norm_first = norm_first
        self.norm2_s = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)
        self.norm3_s = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)
        self.norm2_t = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)
        self.norm3_t = nn.LayerNorm(self.d_model, eps=layer_norm_eps, bias=bias, device=self.device)

        self.dropout_s = nn.Dropout(dropout)
        self.dropout_t = nn.Dropout(dropout)

        # Merging space-time representation for decoder
        self.linear_st = nn.Linear(2 * self.d_model, self.d_model, bias=bias, device=self.device)

        # Decoder part
        self.log_pos = PositionalEncoding1D(self.d_model)
        self.log_embedding = nn.Embedding(3, self.d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)
        self.fc_out = nn.Linear(self.d_model, 1)

        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._precompute_scatter_indices()

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

    # multi-head attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                   is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return x

    # feed forward block
    def _ff_block_t(self, x: Tensor) -> Tensor:
        x = self.linear2_t(self.dropout2_t(self.activation(self.linear1_t(x))))
        return self.dropout3_t(x)

    def _ff_block_s(self, x: Tensor) -> Tensor:
        x = self.linear2_s(self.dropout2_s(self.activation(self.linear1_s(x))))
        return self.dropout3_s(x)

    def _input_repr(self, x):
        batch_size, num_stab = x.size()
        # xr = rearrange(x, 'b (r n) -> b r n', b=batch_size, r=self.rounds, n=num_stab//self.rounds)
        xr = x.view(batch_size, self.rounds, num_stab // self.rounds)
        events = torch.empty(xr.size(0), xr.size(1), self.distance + 1, self.distance + 1, self.d_model,
                             device=self.device)
        x_in = torch.empty(xr.size(0), xr.size(1), xr.size(2), self.d_model, device=self.device)

        for i in range(self.rounds):
            r = xr[:, i]
            if i == self.rounds - 1:
                event = self.final_event_embedding(r)
            else:
                event = self.event_embedding(r)
            event = scatter_to_2d(event, scatter_positions=self.scatter_indices, padding=torch.zeros(self.d_model),
                                  d=self.distance,
                                  device=self.device)
            events[:, i, :, :, :] = event

        position = self.positional_encoding(events)
        embedded = events + position

        for i in range(self.rounds):
            r = embedded[:, i]
            r = collect_from_2d(r, d=self.distance, device=self.device, scatter_positions=self.scatter_indices)
            x_in[:, i] = self._res_net(r)

        # x_in = rearrange(x_in, 'b r n d -> b (r n) d', b=batch_size, r=self.rounds, n=num_stab//self.rounds, d=self.d_model)
        x_in = x_in.view(batch_size, self.rounds * (num_stab // self.rounds), self.d_model)

        return x_in

    def _logical_input_repr(self, x):
        event = self.log_embedding(x)
        position = self.log_pos(event)
        x = event + position
        return x

    def _res_net(self, x):
        identity = x

        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))

        out = self.norm(out + identity)
        out = F.relu(out)

        return out

    def forward(self, syndrome, log):
        """
        syndrome: Shape (B, R * N), R: rounds, N: num syndromes
        """
        batch_size, num_stab = syndrome.size()
        num_stab = num_stab // self.rounds

        # Embedding
        syndrome = self._input_repr(syndrome)

        # Temporal expert
        xt = syndrome
        # xt = rearrange(xt, 'b (r n) d -> (b n) r d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        xt = xt.view(batch_size, self.rounds, num_stab, self.d_model)  # Shape: (b, r, n, d)
        xt = xt.permute(0, 2, 1, 3).contiguous()  # Shape: (b, n, r, d)
        xt = xt.view(batch_size * num_stab, self.rounds, self.d_model)  # Shape: (b*n, r, d)
        res_t = self.time_norm(self.time_encoder(xt) + xt)
        # res_t = rearrange(res_t, '(b n) r d -> b (r n) d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        # res_t = res_t.view(batch_size, num_stab, self.rounds, self.d_model)  # Shape: (b, n, r, d)
        # res_t = res_t.permute(0, 2, 1, 3).contiguous()  # .view(batch_size, self.rounds * num_stab, self.d_model)
        # Shape: (b, r*n, d)

        # Spatial expert
        xs = syndrome  # syndrome
        # xs = rearrange(xs, 'b (r n) d -> (b r) n d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        # xs = xs.view(batch_size, self.rounds, num_stab, self.d_model)  # Shape: (b, r, n, d)
        xs = xs.view(batch_size * self.rounds, num_stab, self.d_model)  # Shape: (b*r, n, d)
        res_s = self.space_norm(self.space_encoder(xs) + xs)
        # res_s = rearrange(xs, '(b r) n d -> b (r n) d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        # res_s = res_s.view(batch_size, self.rounds, num_stab, self.d_model)  # Shape: (b, r, n, d)
        # memory = res_s.view(batch_size, self.rounds * num_stab, self.d_model)  # Shape: (b, r*n, d)

        # cross-attention space-time exchange
        # space information to temporal expert
        # s2t = rearrange(res_t, '(b n) r d -> (b r) n d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        s2t = res_t.view(batch_size, num_stab, self.rounds, self.d_model)  # Shape: (b, n, r, d)
        s2t = s2t.permute(0, 2, 1, 3).contiguous()  # Shape: (b, r, n, d)
        s2t = s2t.view(batch_size * self.rounds, num_stab, self.d_model)  # Shape: (b*r, n, d)
        s2t = self.norm2_t(s2t + self.dropout_t(self._mha_block(s2t, res_s)))
        s2t = self.norm3_t(s2t + self._ff_block_t(s2t))
        # s2t = rearrange(s2t, '(b r) n d -> b (r n) d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        s2t = s2t.reshape(batch_size, self.rounds * num_stab, self.d_model)

        # time information to spatial expert
        # t2s = rearrange(res_s, '(b r) n d -> (b n) r d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        t2s = res_s.view(batch_size, self.rounds, num_stab, self.d_model)  # Shape: (b, r, n, d)
        t2s = t2s.permute(0, 2, 1, 3).contiguous()  # Shape: (b, n, r, d)
        t2s = t2s.view(batch_size * num_stab, self.rounds, self.d_model)  # Shape: (b*n, r, d)

        t2s = self.norm2_s(t2s + self.dropout_s(self._mha_block(t2s, res_t)))
        t2s = self.norm3_s(t2s + self._ff_block_s(t2s))
        # t2s = rearrange(t2s, '(b n) r d -> b (r n) d', b=batch_size, n=num_stab, r=self.rounds, d=self.d_model)
        t2s = t2s.reshape(batch_size, self.rounds * num_stab, self.d_model)

        memory = self.linear_st(torch.cat([t2s, s2t], dim=-1))

        log = self._logical_input_repr(log)
        log = self._res_net(log)

        seq_len = log.size(1)
        mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0)
        mask = mask.masked_fill(mask == 0, -1e12)
        mask = mask.masked_fill(mask == 1, 0.0)
        mask = mask.to(self.device)

        out = self.decoder(tgt=log, memory=memory, tgt_mask=mask)
        out = self.fc_out(out)
        # out = torch.sigmoid(out)  # shape (B, 2, 1)
        out = out.squeeze(2)

        return out

    def log_prob(self, x, include_mi=False):
        """
            Using the log probability as loss function. It is calculated here.
            :param x: syndromes + logicals
            :param include_mi: include mutual information loss
            :return: log_prob

            Allows for pretraining in form of next stabilizer prediction.
        """
        n_syndromes = (self.distance ** 2 - 1) * self.rounds

        syndrome = x[:, :n_syndromes]
        logical = x[:, n_syndromes:]

        # syndrome_rounds = torch.reshape(syndrome, (x.size(0), self.rounds, -1))

        # Start token for transformer-decoder necessary:
        start_token_value = 2
        start_token = torch.full((x.size(0), 1), start_token_value, dtype=torch.int,
                                 device=self.device)

        logical_in = torch.cat((start_token, logical[:, :-1]), dim=1).to(self.device)

        #print(syndrome_rounds)
        #print(logical_rounds)

        x_hat = self.forward(syndrome, logical_in)  # outputs logits

        # log_prob = torch.log(x_hat + epsilon) * logical_rounds + torch.log(1 - x_hat + epsilon) * (
        #         1 - logical_rounds)

        logical = logical.to(x_hat.dtype)
        log_prob = - self.loss(x_hat, logical)

        sequence_length = logical.size(1)

        # alpha = 0.25
        # gamma = 2
        # probs = torch.exp(log_prob)
        # log_prob = alpha * (1 - probs) ** gamma * log_prob
        # print(torch.mean(log_prob.sum(dim=1), dim=0) / logical_rounds.size(1))

        log_prob = log_prob.sum(dim=1)

        if include_mi:
            return log_prob / sequence_length, self.mi_loss(x_hat)
        return log_prob / sequence_length

    def mi_loss(self, p_out):  # p(y | x) - decoder predictions
        """Encourage decoder output to be different from unconditional probability distribution."""
        q_out = torch.mean(p_out, dim=0, keepdim=True).detach()  # p(y) - empirical marginal
        loss = torch.mean(torch.sum(p_out * torch.log(p_out / q_out), dim=1), dim=0)  # Reverse KL
        return -1 * loss  # We maximize MI by making the loss negative

    def predict_logical(self, syndrome):
        """
        Used during inference.
        :param syndrome: Measurement syndromes
        :return: Probability of logical operators
        """
        n_syndromes = (self.distance ** 2 - 1) * (self.rounds)
        previous_every_round = self.every_round
        self.every_round = False

        with torch.no_grad():
            logical = torch.zeros(syndrome.size(0), 1).to(self.device)

            # try:
            #     syndrome = syndrome[:, :, 0]
            # except IndexError:
            #     pass

            start_token_value = 2
            start_token = torch.full((syndrome.size(0), 1), start_token_value, dtype=torch.int,
                                     device=self.device)

            syndrome_in = syndrome[:, :n_syndromes].reshape(syndrome.size(0), self.rounds, -1)
            logical_in = syndrome[:, n_syndromes:]

            logical_in = torch.cat((start_token, logical_in), dim=1).to(self.device)
            logical_in = logical_in.unsqueeze(1)

            for i in range(1):
                conditional = torch.sigmoid(self.forward(syndrome_in, logical_in, predicting=True))

                # conditional = torch.sigmoid(logits)
                if len(conditional.shape) < 2:
                    conditional = conditional.unsqueeze(0)
                # r = torch.as_tensor(np.random.rand(syndrome.size(0)), dtype=torch.float32, device=self.device)
                # syndrome = torch.cat((syndrome, 1*(r < logical[:, i]).unsqueeze(1)), dim=1)
                logical[:, i] = conditional[:, -1]
                # x[:, s + i] = torch.floor(2 * conditional[:, s + i])
                # x[:, s + i] = conditional[:, s + i]

        self.every_round = previous_every_round
        return logical.squeeze()
