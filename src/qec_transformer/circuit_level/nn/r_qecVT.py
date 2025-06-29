import math
from typing import Optional, Union, Callable

from positional_encodings.torch_encodings import PositionalEncoding3D, PositionalEncoding1D
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from .qectransformer import Net, QecTransformer, scatter_to_2d


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


class RQecVT(Net):
    def __init__(self, name, distance,
                 cluster=False,
                 pretraining=False,
                 readout='transformer-decoder',
                 penc_type='fixed',
                 every_round=False,
                 dropout: float = 0.0,
                 patch_distance=3,
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
        # self.rounds = 2
        self.pretraining = pretraining
        self.patch_distance = patch_distance

        # Input representation
        self.positional_encoding = PositionalEncoding3D(self.d_model)

        self.patch_encoder = nn.Sequential(
            nn.Conv2d(1, self.d_model // 4, kernel_size=2, stride=2, padding=0, bias=False, device=self.device),
            nn.Conv2d(self.d_model // 4, self.d_model, kernel_size=2, stride=2, padding=0, bias=False,
                      device=self.device)
        )
        self.patch_encoder_final_round = nn.Sequential(
            nn.Conv2d(1, self.d_model // 4, kernel_size=2, stride=2, padding=0, bias=False, device=self.device),
            nn.Conv2d(self.d_model // 4, self.d_model, kernel_size=2, stride=2, padding=0, bias=False,
                      device=self.device)
        )

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

        self.no_stab = nn.Parameter(torch.randn(1), requires_grad=True)

        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._precompute_scatter_indices()
        self._precompute_blocks()

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

        scatter = torch.zeros(self.patch_distance ** 2 - 1, 2, device=self.device, dtype=torch.int)

        z_idx = (self.patch_distance ** 2 - 1) // 2 - 1
        x_idx = (self.patch_distance ** 2 - 1) - 1

        for x in range(1, self.patch_distance, 2):
            scatter[x_idx, 0] = x
            scatter[x_idx, 1] = 0
            x_idx -= 1

        for y in range(1, self.patch_distance):
            yi = y % 2
            xs = range(yi, self.patch_distance + yi)
            for i, x in enumerate(xs):
                if i % 2 == 0:
                    scatter[z_idx, 0] = x
                    scatter[z_idx, 1] = y
                    z_idx -= 1
                elif i % 2 == 1:
                    scatter[x_idx, 0] = x
                    scatter[x_idx, 1] = y
                    x_idx -= 1

        for x in range(2, self.patch_distance, 2):
            scatter[x_idx, 0] = x
            scatter[x_idx, 1] = self.patch_distance
            x_idx -= 1

        self.scatter_indices_2d = scatter

    def _precompute_blocks(self):
        num_patches = (self.distance // self.patch_distance + 1) ** 2
        patched = torch.zeros(num_patches, self.patch_distance ** 2 - 1, device=self.device, dtype=torch.int64)
        num_p_syndromes = self.patch_distance ** 2 - 1
        num_syndromes = self.distance ** 2 - 1

        p0 = torch.zeros(num_p_syndromes)
        for i in range(self.patch_distance - 1):
            p0[(self.patch_distance // 2 + 1) * i: (self.patch_distance // 2 + 1) * (i + 1)] = (
                    torch.arange(0, self.patch_distance // 2 + 1) + i * (self.distance // 2 + 1))
        for i in range(self.patch_distance + 1):
            p0[
            num_p_syndromes // 2 + (self.patch_distance // 2) * i: num_p_syndromes // 2 + (self.patch_distance // 2) * (
                        i + 1)] = torch.arange(num_syndromes // 2,
                                               num_syndromes // 2 + self.patch_distance // 2) + i * (self.distance // 2)

        offset = torch.zeros(num_p_syndromes, device=p0.device)

        for i, p in enumerate(torch.flip(torch.arange(0, num_patches), dims=[0])):
            patched[p] = p0.clone() + offset
            if (i + 1) % (self.distance // self.patch_distance + 1) == 0:
                # += number stabilizers per row * rows according to patch distance - number of moves before jump
                offset[:num_p_syndromes//2] += (self.distance // 2 + 1) * (
                            self.patch_distance - 1) - self.distance // self.patch_distance * (self.patch_distance // 2)
                offset[num_p_syndromes // 2:] += (self.distance // self.patch_distance + 1) * (
                        self.patch_distance - 1) - self.distance // self.patch_distance * (self.patch_distance // 2)

            else:
                offset += self.patch_distance // 2

        self.patched = patched

    def _input_repr(self, x):
        batch_size, num_stab = x.size()
        num_stab = num_stab // self.rounds
        num_patches, patch_length = self.patched.size()
        x = x.reshape(batch_size * self.rounds, num_stab)
        x = self.patch_input(x, batch_size * self.rounds, num_patches, patch_length)
        # Here shape (B*R, P, 8)
        x = x.view(batch_size * self.rounds * num_patches, patch_length)

        x = scatter_to_2d(x, self.scatter_indices_2d, d=self.patch_distance, padding=None, device=self.device).to(torch.float)
        x = x.permute(0, 3, 1, 2)
        stab_embedding = self.patch_encoder(x[:-batch_size * num_patches])
        final_embedding = self.patch_encoder_final_round(x[-batch_size * num_patches:])
        events_embedded = torch.cat((stab_embedding, final_embedding), dim=0).squeeze()

        events_embedded = events_embedded.unsqueeze(1).unsqueeze(1)
        events_embedded = events_embedded.view(batch_size, self.rounds, int(math.sqrt(num_patches)), int(math.sqrt(num_patches)), self.d_model)
        position = self.positional_encoding(events_embedded)
        embedded = events_embedded + position

        x_in = embedded.flatten(2, 3)
        x_in = x_in.view(batch_size * self.rounds, num_patches, self.d_model)
        x_in = self._res_net(x_in)
        x_in = x_in.view(batch_size, self.rounds * num_patches, self.d_model)
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

    def patch_input(self, syndrome, batch_size, num_patches, patch_length):
        # Indices to gather patches
        indices = self.patched.unsqueeze(0).expand(batch_size, -1, -1)
        indices = indices.flatten(1)

        # Gather patches
        syndrome = torch.gather(syndrome, 1, indices)
        syndrome = syndrome.reshape((batch_size, num_patches, patch_length))

        return syndrome

    def forward(self, syndrome, log):
        """
        syndrome: Shape (B, R * N), R: rounds, N: num syndromes
        """
        batch_size, num_stab = syndrome.size()
        num_stab = num_stab // self.rounds

        # Embedding
        syndrome = self._input_repr(syndrome)

        memory = self.encoder(syndrome)

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
        epsilon = 1e-9

        n_syndromes = (self.distance ** 2 - 1) * self.rounds

        syndrome = x[:, :n_syndromes]
        logical = x[:, n_syndromes:]

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

        return logical.squeeze()
