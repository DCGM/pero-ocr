import json
import math
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules import TransformerDecoder, TransformerDecoderLayer, ModuleList, MultiheadAttention

from typing import Optional, Tuple, List, Union


def build_net(net, input_height, input_channels, nb_output_symbols, max_seq_len=2000):
    config = json.loads(net) if type(net) == str else net
    dim_model = config['dim_model']
    dim_ff = config['dim_ff']
    heads = config['heads']
    dropout_rate = 0.0
    encoder_layers = config['encoder_layers']
    decoder_layers = config['decoder_layers']
    conv_subsampling = config['conv_subsampling']

    conv_frontend = ConvolutionalEncoder(
        in_height=input_height,
        in_channels=input_channels,
        conv_subsampling=conv_subsampling,
        out_channels=dim_model
    )
    encoder = LineSelfAttentionEncoder(
        dropout=dropout_rate,
        max_seq_len=512,
        dim_model=dim_model,
        dim_ff=dim_ff,
        nb_layers=encoder_layers,
        nb_heads=heads
    )
    model = TransformerOCR(
        encoder_frontend=conv_frontend,
        encoder=encoder,
        num_classes=nb_output_symbols + 2,
        dropout=dropout_rate,
        nb_layers=decoder_layers,
        dim_model=dim_model,
        dim_ff=dim_ff,
        max_seq_len=512,
        nb_heads=heads,
    )

    return model


def create_vgg_block_2d(in_channels, out_channels, stride=(2,2), layer_count=2, norm='bn'):
    layers = []
    for i in range(layer_count):
        if norm == 'bn':
            layers += [
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.LeakyReLU(),
            ]
        elif norm == 'none':
            layers += [
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
            ]
        else:
            print(f'ERROR: Normalization "f{norm}" is not implemented')
            raise "Unknown norm"

        in_channels = out_channels

    layers += [torch.nn.MaxPool2d(kernel_size=stride, stride=stride)]
    return torch.nn.Sequential(*layers)


class VGG_conv_module(torch.nn.Module):
    def __init__(self, base_channels=16, conv_blocks=4, subsampling=(8, 4), in_channels=3, layers_2d=None, dropout_rate=0.0):
        super(VGG_conv_module, self).__init__()
        if layers_2d is None:
            layers_2d = 16

        if type(layers_2d) is int:
            import torchvision
            vgg = torchvision.models.vgg16(pretrained=True)
            layers_2d = list(vgg.features[:layers_2d])

        start_level = 0
        self.blocks_2d = []
        current_subsampling_h = 1
        current_subsampling_v = 1

        for layer in layers_2d:
            if type(layer) == torch.nn.modules.pooling.MaxPool2d:
                if subsampling[0] is None or current_subsampling_v < subsampling[0]:
                    stride_v = 2
                else:
                    stride_v = 1

                if current_subsampling_h < subsampling[1]:
                    stride_h = 2
                else:
                    stride_h = 1

                stride = (stride_v, stride_h)

                self.blocks_2d += [torch.nn.MaxPool2d(kernel_size=stride, stride=stride)]
                self.blocks_2d += [torch.nn.Dropout(p=dropout_rate, inplace=True)]
                current_subsampling_h *= stride[1]
                current_subsampling_v *= stride[0]
                start_level += 1
            else:
                self.blocks_2d.append(layer)
                if type(layer) == torch.nn.modules.conv.Conv2d:
                    in_channels = layer.bias.shape[0]

        print('Pretrained layers')
        print(self.blocks_2d)

        out_channels = in_channels
        for i in range(start_level, conv_blocks):
            out_channels = base_channels*(2**i)
            if subsampling[0] is None or current_subsampling_v < subsampling[0]:
                stride_v = 2
            else:
                stride_v = 1

            if current_subsampling_h < subsampling[1]:
                stride_h = 2
            else:
                stride_h = 1

            stride = (stride_v, stride_h)

            current_subsampling_h *= stride[1]
            current_subsampling_v *= stride[0]

            self.blocks_2d += [
                create_vgg_block_2d(in_channels, out_channels, stride=stride, norm='none'),
                torch.nn.BatchNorm2d(out_channels),
                ]

            self.blocks_2d += [torch.nn.Dropout(p=dropout_rate, inplace=True)]
            in_channels = out_channels

        self.blocks_2d = torch.nn.Sequential(*self.blocks_2d)
        self.out_channels = out_channels

    def forward(self, x):
        return self.blocks_2d(x)


class SequenceTooLongException(Exception):
    pass


class CustomMultiheadAttention(MultiheadAttention):
    """
    This is the original MHA with added cached_forward method, which is used
    for more effective inference using caching and cache_index_select method
    used during beam search.
    """
    def __init__(self, embedding_len, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, is_self_attention=False, max_seq_len=500):
        super(CustomMultiheadAttention, self).__init__(embedding_len, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim,
                                                       vdim)
        self.max_seq_len = max_seq_len
        self.is_self_attention = is_self_attention
        self.linear_cache = None

    def reallocate_caches(self, sequence_len):
        if self.linear_cache is None:
            raise Exception("Caches not allocated yet!")

        self.max_seq_len = sequence_len

        batch_size, embedding_len = self.memory_tgt.shape[1:]
        self.linear_cache = torch.empty((sequence_len, batch_size, embedding_len), device=self.linear_cache.device)

    def infer(self, query: Tensor, seq_len:int, key: Tensor, value: Tensor, need_weights: bool = True,
              return_attention=False)\
            -> Tuple[Tensor, Optional[Tensor]]:
        return self.cached_forward(
            query, seq_len, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias, self.out_proj.weight,
            self.out_proj.bias, need_weights=need_weights, return_attention=return_attention)

    def cached_forward(self,
                       query,  # type: Tensor
                       seq_len,
                       key,  # type: Tensor
                       value,  # type: Tensor
                       embed_dim_to_check,  # type: int
                       num_heads,  # type: int
                       in_proj_weight,  # type: Tensor
                       in_proj_bias,  # type: Tensor
                       out_proj_weight,  # type: Tensor
                       out_proj_bias,  # type: Tensor
                       need_weights=True,  # type: bool
                       return_attention=False,  # type: bool
                       ):
        # type: (...) -> Tuple[Tensor, Optional[Tensor]]
        r"""
        Method which is used for inference and utilizes cache.

        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
            need_weights: output attn_output_weights.

        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.

            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """

        _, batch_size, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        _, batch_size, embedding_len = query.shape
        query = query[-1:]

        assert seq_len < self.max_seq_len, f"MHA: Sequence longer than {self.max_seq_len} logits"

        if self.linear_cache is None or seq_len == 1:
            # shape: [seq, batch, embedding_len * 3]
            self.linear_cache = torch.empty((self.max_seq_len, batch_size, embedding_len * 3), device=query.device)

            if not self.is_self_attention:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]

                self.linear_cache[:key.shape[0], :, embedding_len:] = F.linear(key, _w, _b)

        if self.is_self_attention:
            self.linear_cache[seq_len - 1] = F.linear(query, in_proj_weight, in_proj_bias)
            q = self.linear_cache[seq_len - 1:seq_len, :, :embedding_len]
            k, v = self.linear_cache[:seq_len, :, embedding_len:].chunk(2, axis=-1)

        else:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]

            q = F.linear(query, _w, _b)
            self.linear_cache[seq_len - 1:seq_len, :, :embedding_len] = q

            k, v = self.linear_cache[:key.shape[0], :, embedding_len:].chunk(2, dim=-1)

        q = q * scaling

        q = q.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [batch_size * num_heads, 1, src_len]

        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)

        attn_output = torch.bmm(attn_output_weights, v)

        assert list(attn_output.size()) == [batch_size * num_heads, 1, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, batch_size, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

        if return_attention:
            return attn_output, attn_output_weights.clone().detach()

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(batch_size, num_heads, -1, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None

    def cache_index_select(self, indices, seq_len):
        """
        Method used during beam search to "shuffle" cache using given indices. Length
        of shuffled cache is limited by current length of sequence.
        """
        self.linear_cache[:seq_len] = self.linear_cache[:seq_len].index_select(1, indices)


class PositionalEncoding(torch.nn.Module):
    """
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ConvolutionalEncoder(torch.nn.Module):
    def __init__(self, in_height, in_channels, out_channels, conv_subsampling=(8, 8)):
        super().__init__()
        self.base_channels = 64
        self.conv_blocks = 4
        self.conv_subsampling = conv_subsampling
        self.layers_2d = 17
        self.dropout_rate = 0.0
        self.blocks_2d = VGG_conv_module(base_channels=self.base_channels, conv_blocks=self.conv_blocks,
                                         subsampling=self.conv_subsampling,
                                         in_channels=in_channels, layers_2d=self.layers_2d,
                                         dropout_rate=self.dropout_rate)

        aggregation_height = in_height // conv_subsampling[0]
        print('Aggregation height', aggregation_height)

        self.aggregation_conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.blocks_2d.out_channels, out_channels, kernel_size=(aggregation_height, 1), stride=1,
                            padding=0),
            torch.nn.LeakyReLU()
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, data):
        image_enc = self.blocks_2d(data)
        image_enc = self.aggregation_conv(image_enc)
        image_enc = torch.squeeze(image_enc)
        return image_enc


class LineSelfAttentionEncoder(torch.nn.Module):
    def __init__(self, dropout, max_seq_len=1000, dim_model=512, dim_ff=2048, nb_heads=8, nb_layers=2):
        super().__init__()
        self.dim_model = dim_model

        encoder_layer = torch.nn.TransformerEncoderLayer(self.dim_model, nb_heads, dim_feedforward=dim_ff,
                                                         dropout=dropout)
        self.trans_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=nb_layers)
        self.pos_encoder = PositionalEncoding(self.dim_model, max_len=max_seq_len)

        self.input_norm = torch.nn.LayerNorm(dim_model, eps=1e-05)

    def forward(self, X):  # [batch, channels, time]
        image_enc_tbc = X.permute(2, 0, 1)  # [time, batch, channels]
        image_enc_tbc = self.input_norm(image_enc_tbc)
        encoder_output = self.trans_encoder(self.pos_encoder(image_enc_tbc))
        return encoder_output

    def change_max_seq_len(self, seq_len):
        self.pos_encoder = PositionalEncoding(self.dim_model, max_len=seq_len)


class DecoderLayer(TransformerDecoderLayer):
    def __init__(self, dim_model, nb_heads, dim_ff=2048, dropout=0.0, activation='relu', max_seq_len=500, norm=None):
        super(DecoderLayer, self).__init__(dim_model, nb_heads, dim_ff, dropout, activation)
        max_seq_len = 220
        self.self_attn = CustomMultiheadAttention(
            dim_model,
            nb_heads,
            dropout=dropout,
            is_self_attention=True,
            max_seq_len=512,
        )
        self.multihead_attn = CustomMultiheadAttention(
            dim_model,
            nb_heads,
            dropout=dropout,
            max_seq_len=512,
        )

        self.memory_tgt: Optional[Tensor] = None
        self.max_seq_len = max_seq_len

    def reallocate_caches(self, sequence_len):
        if self.memory_tgt is None:
            raise Exception("Memory not allocated yet.")

        self.max_seq_len = sequence_len

        batch, emb = self.memory_tgt.shape[1:]
        self.memory_tgt = torch.empty((self.max_seq_len, batch, emb), device=self.memory_tgt.device)

        self.self_attn.reallocate_caches(sequence_len)
        self.multihead_attn.reallocate_caches(sequence_len)

    def infer(self, tgt: Tensor, memory: Tensor, is_cached: bool = False, return_attention: bool = False) -> \
            Union[Tensor, Tuple[Tensor, Tensor]]:
        seq_len = tgt.shape[0]

        if seq_len >= self.max_seq_len:
            raise SequenceTooLongException()

        if is_cached:
            tgt_single = tgt[-1:]
        else:
            tgt_single = tgt

        if is_cached:
            tgt_single = tgt_single + self.self_attn.infer(tgt_single, seq_len, tgt, tgt, need_weights=False)[0]
        else:
            tgt_single = tgt_single + self.self_attn(tgt_single, tgt, tgt, need_weights=False)[0]
        tgt_single = self.norm1(tgt_single)

        if is_cached:
            if return_attention:
                tmp, attention = self.multihead_attn.infer(tgt_single, seq_len, memory, memory,
                                                           return_attention=return_attention, need_weights=False)
                tgt_single += tmp
            else:
                tgt_single += self.multihead_attn.infer(tgt_single, seq_len, memory, memory, need_weights=False)[0]
        else:
            tgt_single += self.multihead_attn(tgt_single, memory, memory, need_weights=False)[0]
        tgt_single = self.norm2(tgt_single)

        tgt_single += self.linear2(self.activation(self.linear1(tgt_single)))
        tgt_single = self.norm3(tgt_single)

        # different batch sizes -> reset memory_tgt
        if self.memory_tgt is not None and self.memory_tgt.shape[1] != tgt.shape[1]:
            self.memory_tgt = None

        # sequence with 1 element => rewrite memory (seq, batch, embedding)
        if self.memory_tgt is None:
            batch, emb = tgt.shape[1:]
            self.memory_tgt = torch.empty((self.max_seq_len, batch, emb), device=tgt.device)

        self.memory_tgt[seq_len - 1, :, :] = tgt_single[-1]

        if return_attention:
            return self.memory_tgt[:seq_len, :, :], attention

        return self.memory_tgt[:seq_len, :, :]

    def cache_index_select(self, indices, seq_len):
        self.memory_tgt[:seq_len] = self.memory_tgt[:seq_len].index_select(1, indices)

        self.self_attn.cache_index_select(indices, seq_len)
        self.multihead_attn.cache_index_select(indices, seq_len)


class Decoder(TransformerDecoder):
    def __init__(self, nb_layers, dim_model, nb_heads, expansion_dim, dropout, norm=None,
                 max_seq_len=500):
        super().__init__(None, 0)
        layer_constructor = lambda: DecoderLayer(dim_model, nb_heads, expansion_dim, dropout, max_seq_len=max_seq_len)
        self.layers = ModuleList([layer_constructor() for _ in range(nb_layers)])
        self.norm = norm

    def infer(self, tgt: Tensor, memory: Tensor, is_cached: bool = False, return_attention=False) -> \
            Union[Tensor, Tuple[Tensor, Tensor]]:
        attention = None

        for layer_idx, layer in enumerate(self.layers):
            if return_attention and layer_idx == len(self.layers) - 1:
                tgt, attention = layer.infer(tgt, memory, is_cached, return_attention=True)
            else:
                tgt = layer.infer(tgt, memory, is_cached)

        if self.norm is not None:
            tgt[-1, :, :] = self.norm(tgt[-1, :, :])

        if return_attention:
            return tgt[-1, :, :], attention

        return tgt[-1, :, :]

    def cache_index_select(self, indices, seq_len):
        for layer in self.layers:
            layer.cache_index_select(indices, seq_len)

    def reallocate_caches(self, seq_len):
        for layer in self.layers:
            layer.reallocate_caches(seq_len)


class TransformerOCR(torch.nn.Module):
    def __init__(self, encoder_frontend, encoder, num_classes, dropout, nb_layers=4, dim_model=512, dim_ff=2048,
                 max_seq_len=500, nb_heads=8):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.dim_model = dim_model
        self.num_classes = num_classes

        self.trans_decoder = Decoder(nb_layers, dim_model, nb_heads, dim_ff, dropout=dropout,
                                     max_seq_len=max_seq_len)

        mask = torch.triu(torch.full((max_seq_len, max_seq_len), - float("inf")), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

        self.pos_encoder = PositionalEncoding(dim_model, max_len=max_seq_len)

        self.dec_embeder = torch.nn.Embedding(num_classes, dim_model)
        self.dec_out_proj = torch.nn.Linear(dim_model, num_classes)

    def forward(self, X, labels):
        ''' Both X and labels are expected as batch-first !
        '''

        encoder_output = self.encode(X)

        dec_mask = self.get_mask(labels.shape[1])
        label_embs = self.dec_embeder(labels.permute(1, 0))
        transformed = self.trans_decoder(self.pos_encoder(label_embs), encoder_output, tgt_mask=dec_mask)

        return self.dec_out_proj(transformed)

    def get_mask(self, length):
        return self.mask[:length, :length]

    def encode(self, X):
        enc = self.encoder_frontend(X)

        if len(enc.shape) == 2:
            enc = enc.unsqueeze(0)

        enc = self.encoder(enc)
        return enc

    def change_max_seq_len(self, seq_len):
        self.max_seq_len = seq_len

        self.mask = torch.triu(torch.full((seq_len, seq_len), - float("inf")), diagonal=1)
        self.trans_decoder.reallocate_caches(seq_len)
        self.pos_encoder = PositionalEncoding(self.dim_model, max_len=seq_len)

        self.encoder.change_max_seq_len(seq_len)
