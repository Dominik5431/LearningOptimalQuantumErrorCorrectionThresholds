import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .qectransformer import QecTransformer, Net, scatter_to_2d, collect_from_2d
from positional_encodings.torch_encodings import PositionalEncoding2D, PositionalEncoding1D


class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, x, y, d_model, device):
        super().__init__()
        self.x = x
        self.y = y

        self.x_position = torch.zeros(self.x * self.y, dtype=torch.long, device=device)
        self.y_position = torch.zeros(self.x * self.y, dtype=torch.long, device=device)
        self.x_embedding = nn.Embedding(x, d_model, device=device)
        self.y_embedding = nn.Embedding(y, d_model, device=device)
        self.d_model = d_model
        self.device = device

        self.x_position = torch.arange(0, self.x * self.y) % x
        self.y_position = torch.arange(0, self.x * self.y) // y

        self.x_position = self.x_position.to(self.device)
        self.y_position = self.y_position.to(self.device)

    def forward(self, x):
        return self.x_embedding(self.x_position[:x.size(1)]) + self.y_embedding(self.y_position[:x.size(1)])


class QecVT(Net):
    def __init__(self, name, distance,
                 cluster=False,
                 pretraining=False,
                 rounds=2,
                 pretrained_qec_name=None,
                 readout='conv',
                 biased_attention=False,
                 convolutions=False,
                 measurement_input=False,
                 input_residual=False,
                 patch_distance=3,
                 **kwargs):
        super(QecVT, self).__init__(name, cluster)

        self.measurement_input = measurement_input

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
        self.readout = readout
        self.patch_distance = patch_distance
        self.distance = distance

        self.pretraining = pretraining

        '''
        if pretrained_qec_name is None:
            pretrained_qec_name = self.name
            self.patch_encoder = QecTransformer(name=pretrained_qec_name, distance=self.patch_distance,
                                                cluster=self.cluster,
                                                pretraining=self.pretraining, rounds=self.rounds, readout=self.readout,
                                                biased_attention=False, convolutions=True,
                                                measurement_input=False,
                                                input_residual=True, **kwargs)
        
        self.patch_encoder = QecTransformer(name=pretrained_qec_name, distance=self.patch_distance,
                                            cluster=self.cluster, pretraining=self.pretraining, rounds=self.rounds,
                                            readout=self.readout,
                                            biased_attention=False, convolutions=True,
                                            measurement_input=False, penc_type='fixed',
                                            input_residual=True, **kwargs)# .load()
        '''
        if self.patch_distance == 3:
            self.patch_encoder = nn.Sequential(
                nn.Conv2d(1, self.d_model//4, kernel_size=2, stride=2, padding=0, bias=False, device=self.device),
                nn.Conv2d(self.d_model // 4, self.d_model, kernel_size=2, stride=2, padding=0, bias=False, device=self.device)
            )
        else:
            self.patch_encoder = nn.Sequential(
                nn.Conv2d(1, self.d_model//4, kernel_size=2, stride=2, padding=0, bias=False, device=self.device),
                nn.Conv2d(self.d_model // 4, self.d_model//2, kernel_size=2, stride=1, padding=0, bias=False, device=self.device),
                nn.Conv2d(self.d_model//2, self.d_model, kernel_size=2, stride=2, padding=0, bias=False, device=self.device)
            )

        # Residual network to map to the input representation
        self.linear1 = nn.Linear(self.d_model, self.d_model)
        self.norm1 = nn.LayerNorm(self.d_model, eps=1e-5, bias=True, device=self.device)
        self.linear2 = nn.Linear(self.d_model, self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model, eps=1e-5, bias=True, device=self.device)

        self._precompute_blocks()
        # Learnable padding vector for the positions where there is no stabilizer
        self.no_stab = nn.Parameter(torch.randn(1), requires_grad=True)


        # self.patch_position_encoding = LearnablePositionalEncoding2D(self.distance // self.patch_distance + 1,
        #                                                              self.distance // self.patch_distance + 1,
        #                                                              self.d_model, self.device)

        self.patch_position_encoding = PositionalEncoding2D(self.d_model)
        self.patch_linear = nn.Linear(self.d_model * (self.patch_distance ** 2 - 1), self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=self.dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        self.event_embedding = nn.Embedding(3, self.d_model)
        self.msmt_embedding = nn.Embedding(5, self.d_model)
        self.log_pos = PositionalEncoding1D(self.d_model)

        # self.res_net = self.patch_encoder.res_net

        # for param in self.patch_encoder.parameters():
        #     param.requires_grad = False

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=self.dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)
        self.fc_out = nn.Linear(self.d_model, 1)

        self._precompute_scatter_indices()

    def _precompute_scatter_indices(self):
        """
        Precomputes the positions of the stabilizers they get scattered to.
        :return: scatter positions
        """
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

        self.scatter_indices = scatter

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

    def patch_input(self, syndrome, batch_size, num_patches, patch_length):
        # Indices to gather patches
        indices = self.patched.unsqueeze(0).expand(batch_size, -1, -1)
        indices = indices.flatten(1)

        # Gather patches
        syndrome = torch.gather(syndrome, 1, indices)
        syndrome = syndrome.reshape((batch_size, num_patches, patch_length))

        return syndrome

    def res_net(self, x, scaling_factor=1 / math.sqrt(2)):
        identity = x

        out = self.norm1(F.relu(self.linear1(x)))
        out = self.norm2(F.relu(self.linear2(out)))

        out += identity
        out = F.relu(out * scaling_factor)

        return out

    def forward(self, syndrome: Tensor, logical: Tensor) -> Tensor:
        num_patches, patch_length = self.patched.size()
        batch_size = syndrome.size(0)

        syndrome = self.patch_input(syndrome, batch_size, num_patches, patch_length)
        # Encode the patches
        syndrome = syndrome.view(-1, patch_length)

        # Equivalent to encoding part in self.patch_encoder network
        # syndrome = self.patch_encoder.input_repr(syndrome)
        # syndrome = self.patch_encoder.res_net(syndrome)
        # syndrome = self.patch_encoder.encoder(syndrome)

        # Only for convolutional patch encoding
        syndrome = scatter_to_2d(syndrome.unsqueeze(2), self.scatter_indices, self.patch_distance, torch.zeros(1), self.device)
        syndrome = syndrome.to(torch.float)
        syndrome = self.patch_encoder(syndrome.permute(0, 3, 1, 2))
        syndrome = syndrome.squeeze()
        syndrome = self.res_net(syndrome)

        # Here additional operation to get from (patch_length, d_model) to d_model: only for encoder patch encoding
        # syndrome = syndrome.flatten(1)
        # syndrome = self.patch_linear(syndrome)
        # syndrome = syndrome.reshape(batch_size, num_patches, self.d_model)

        # Patch positional encoding
        syndrome = syndrome.view(batch_size, self.distance // self.patch_distance + 1,
                                 self.distance // self.patch_distance + 1, self.d_model)
        syndrome = syndrome + self.patch_position_encoding(syndrome)
        syndrome = syndrome.view(batch_size, num_patches, self.d_model)

        # Transformer encoding
        syndrome = self.encoder(syndrome)

        # Prepare logical observable, encoding equal as in patch_encoder
        log = self.logical_input_repr(logical)
        # log = self.res_net(log)

        seq_len = log.size(1)
        mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0)
        mask = mask.masked_fill(mask == 0, -1e12)
        mask = mask.masked_fill(mask == 1, 0.0)
        # mask = mask == 0
        mask = mask.to(self.device)

        # Masked decoding
        out = self.decoder(tgt=log, memory=syndrome, tgt_mask=mask)
        out = self.fc_out(out)
        out = F.sigmoid(out)

        return out.squeeze(2)

    def pretraining_forward(self, syndrome: Tensor):
        num_patches, patch_length = self.patched.size()
        batch_size = syndrome.size(0)

        syndrome_patched = self.patch_input(syndrome, batch_size, num_patches,
                                            patch_length)  # (batch_size, num_patches, patch_length)

        out = []

        for n in range(num_patches):
            s = syndrome_patched[:, n, :]
            encoded = self.patch_encoder.encoder_forward(s)
            out.append(encoded)

        # After the loop, concatenate all the encoded results along the appropriate dimension
        out = torch.stack(out, dim=1)

        return out, syndrome_patched

    def log_prob(self, x, include_mi=False):
        """
        Using the log probability as loss function. It is calculated here.
        :param include_mi:
        :param x: syndromes + logicals
        :return: log_prob

        Allows for pretraining in form of next stabilizer prediction.
        """
        epsilon = 1e-9

        if not self.measurement_input:
            try:
                x = x[:, :, 0]
            except IndexError:
                print("Problem with measurement input.")
                pass

        syndrome = x[:, :self.distance ** 2 - 1]
        logical = x[:, self.distance ** 2 - 1:]

        start_token_value = 2
        start_token = torch.full((x.size(0), 1), start_token_value, dtype=torch.long,
                                 device=self.device)

        if self.pretraining:
            next_stab, syndrome = self.pretraining_forward(syndrome)

            next_stab = next_stab.flatten(1)
            syndrome = syndrome.flatten(1)

            log_prob = torch.log(next_stab + epsilon) * syndrome + torch.log(1 - next_stab + epsilon) * (
                    1 - syndrome)
            sequence_length = syndrome.size(1)
        else:
            # Start token for transformer-decoder necessary:
            logical_in = torch.cat((start_token, x[:, self.distance ** 2 - 1:-1]), dim=1).to(self.device)

            x_hat = self.forward(syndrome, logical_in)

            log_prob = torch.log(x_hat + epsilon) * logical + torch.log(1 - x_hat + epsilon) * (
                    1 - logical)

            sequence_length = logical.size(1)

        if include_mi:
            return log_prob.sum(dim=1) / sequence_length, self.mi_loss(x_hat)

        return log_prob.sum(dim=1) / sequence_length

    def mi_loss(self, p_out):  # p(y | x) - decoder predictions
        """Encourage decoder output to be different from unconditional probability distribution."""
        q_out = torch.mean(p_out, dim=0, keepdim=True).detach()  # p(y) - empirical marginal
        loss = torch.mean(torch.sum(p_out * torch.log(p_out/q_out), dim=1), dim=0)  # Reverse KL
        return -1 * loss  # We maximize MI by making the loss negative

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

    def predict_logical(self, syndrome):
        """
        Used during inference.
        :param syndrome: Measurement syndromes
        :return: Probability of logical operators
        """
        with torch.no_grad():
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
                try:
                    syndrome = syndrome[:, :, 0]
                except IndexError:
                    pass
                start_token_value = 2
                start_token = torch.full((syndrome.size(0), 1), start_token_value, dtype=torch.long, device=self.device)
                syndrome = torch.cat((syndrome[:, :self.distance ** 2 - 1], start_token, syndrome[:, self.distance ** 2 - 1:]), dim=1).to(self.device)

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

        return logical.squeeze()


if __name__ == '__main__':
    # For testing
    distance = 7
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
    model = QecVT(name='test-transformer', **model_dict)
