# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
from torch import nn
import numpy as np
from functools import partial
from .line_ocr_engine import BaseEngineLineOCR


# scores_probs should be N,C,T, blank is last class
def greedy_decode_ctc(scores_probs, chars):
    if len(scores_probs.shape) == 2:
        scores_probs = torch.cat((scores_probs[:, 0:1], scores_probs), axis=1)
        scores_probs[:, 0] = -1000
        scores_probs[-1, 1] = 1000
    else:
        scores_probs = torch.cat((scores_probs[:, :, 0:1], scores_probs), axis=2)
        scores_probs[:, :, 0] = -1000
        scores_probs[:, -1, 0] = 1000

    best = torch.argmax(scores_probs, 1) + 1
    mask = best[:, :-1] == best[:, 1:]
    best = best[:, 1:]
    best[mask] = 0
    best[best == scores_probs.shape[1]] = 0
    best = best.cpu().numpy() - 1

    outputs = []
    for line in best:
        line = line[np.nonzero(line >= 0)]
        outputs.append(''.join([chars[c] for c in line]))
    return outputs


class PytorchEngineLineOCR(BaseEngineLineOCR):
    def __init__(self, json_def, gpu_id=0, batch_size=8):
        super(PytorchEngineLineOCR, self).__init__(json_def, gpu_id=gpu_id, batch_size=8)

        self.net_subsampling = 4
        self.characters = list(self.characters) + [u'\u200B']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = PYTORCH_NETS[self.net_name]
        self.model = net(num_classes=len(self.characters), in_height=self.line_px_height, num_embeddings=self.embed_num)
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        if self.embed_id is not None and self.embed_id == "mean":
            embeddings = self.model.embeddings_layer(torch.LongTensor(list(range(self.model.embeddings_layer.num_embeddings))).to(self.device))
            mean_embedding = torch.mean(embeddings, 0, keepdim=True)
            self.model.embeddings_layer = torch.nn.Embedding.from_pretrained(mean_embedding)
            self.embed_id = 0

    def run_ocr(self, batch_data):
        with torch.no_grad():
            batch_data = torch.from_numpy(batch_data).to(self.device).float() / 255.0
            if self.embed_id is not None:
                ids_embedding = torch.LongTensor([self.embed_id] * batch_data.shape[0]).to(self.device)
                self.model.embeddings['data'] = self.model.embeddings_layer(ids_embedding)
            logits = self.model(batch_data)
            decoded = greedy_decode_ctc(logits, self.characters)
            logits = logits.permute(0, 2, 1).cpu().numpy()

        return decoded, logits


def create_vgg_block_2d(in_channels, out_channels, stride=(2,2), layer_count=2, norm='bn'):
    layers = []
    for i in range(layer_count):
        if norm == 'bn':
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.LeakyReLU(),
            ]
        elif norm == 'none':
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
            ]
        else:
            print(f'ERROR: Normalization "f{norm}" is not implemented')
            raise "Unknown norm"

        in_channels = out_channels

    layers += [nn.MaxPool2d(kernel_size=stride, stride=stride)]
    return nn.Sequential(*layers)


def create_vgg_block_1d(in_channels, out_channels, stride=(2,2), layer_count=2, norm='bn'):
    layers = []
    for i in range(layer_count):
        if norm == 'bn':
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm1d(out_channels),
                torch.nn.LeakyReLU(),
            ]
        elif norm == 'none':
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
            ]
        else:
            print(f'ERROR: Normalization "f{norm}" is not implemented')
            raise "Unknown norm"
        in_channels = out_channels
    return nn.Sequential(*layers)


class NET_VGG(nn.Module):
    def __init__(self, num_classes, in_height=32, base_channels=16, conv_blocks=4, subsampling=4, in_channels=3,
                 layers_2d=None, **kwargs):
        super(NET_VGG, self).__init__()
        if layers_2d is None:
            layers_2d = 16

        if type(layers_2d) is int:
            import torchvision
            vgg = torchvision.models.vgg16(pretrained=True)
            layers_2d = list(vgg.features[:layers_2d])

        start_level = 0
        self.blocks_2d = []
        actual_subsampling_h = 1
        actual_subsampling_v = 1
        for layer in layers_2d:
            if type(layer) == torch.nn.modules.pooling.MaxPool2d:
                if actual_subsampling_h < subsampling:
                    stride = (2, 2)
                else:
                    stride = (2, 1)
                self.blocks_2d += [nn.MaxPool2d(kernel_size=stride, stride=stride)]
                actual_subsampling_h *= stride[1]
                actual_subsampling_v *= stride[0]
                start_level += 1
            else:
                self.blocks_2d.append(layer)
                if type(layer) == torch.nn.modules.conv.Conv2d:
                    in_channels = layer.bias.shape[0]

        out_channels = in_channels
        for i in range(start_level, conv_blocks):
            out_channels = base_channels*(2**i)
            if actual_subsampling_h < subsampling:
                stride=(2, 2)
            else:
                stride=(2, 1)
            actual_subsampling_h *= stride[1]
            actual_subsampling_v *= stride[0]
            self.blocks_2d += [
                create_vgg_block_2d(in_channels, out_channels, stride=stride, norm='none'),
                torch.nn.BatchNorm2d(out_channels),
                ]
            in_channels = out_channels

        self.blocks_2d = nn.Sequential(*self.blocks_2d)
        self.block_1d = create_vgg_block_1d(in_channels , out_channels)
        self.gru = torch.nn.LSTM(out_channels, out_channels // 2, num_layers=2, bidirectional=True)
        self.output_layer = nn.Conv1d(out_channels, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.blocks_2d(x)
        out = torch.mean(out, 2)
        out = self.block_1d(out)
        out, _ = self.gru(out.permute(2, 0, 1))
        out = out.permute(1, 2, 0)
        out = self.output_layer(out)
        return out


class VGG_conv_module(nn.Module):
    def __init__(self, base_channels=16, conv_blocks=4, subsampling=4, in_channels=3, layers_2d=None):
        super(VGG_conv_module, self).__init__()
        if layers_2d is None:
            layers_2d = 16

        if type(layers_2d) is int:
            import torchvision
            vgg = torchvision.models.vgg16(pretrained=True)
            layers_2d = list(vgg.features[:layers_2d])

        start_level = 0
        self.blocks_2d = []
        actual_subsampling_h = 1
        actual_subsampling_v = 1
        for layer in layers_2d:
            if type(layer) == torch.nn.modules.pooling.MaxPool2d:
                if actual_subsampling_h < subsampling:
                    stride = (2, 2)
                else:
                    stride = (2, 1)
                self.blocks_2d += [nn.MaxPool2d(kernel_size=stride, stride=stride)]
                actual_subsampling_h *= stride[1]
                actual_subsampling_v *= stride[0]
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
            if actual_subsampling_h < subsampling:
                stride = (2, 2)
            else:
                stride = (2, 1)
            actual_subsampling_h *= stride[1]
            actual_subsampling_v *= stride[0]
            self.blocks_2d += [
                create_vgg_block_2d(in_channels, out_channels, stride=stride, norm='none'),
                torch.nn.BatchNorm2d(out_channels),
                ]
            in_channels = out_channels

        self.blocks_2d = nn.Sequential(*self.blocks_2d)
        self.out_channels = out_channels

    def forward(self, x):
        return self.blocks_2d(x.contiguous())


class MultiscaleRecurrentBlock(nn.Module):
    def __init__(self, channels, layers_per_scale=2, scales=4):
        super(MultiscaleRecurrentBlock, self).__init__()

        self.layers = nn.ModuleList([torch.nn.LSTM(channels, channels // 2, num_layers=layers_per_scale, bidirectional=True)
                  for scale in range(scales)])

        self.final_layer = torch.nn.LSTM(channels, channels // 2, num_layers=1, bidirectional=True)

    def forward(self, x):
        outputs = []
        for depth, layer in enumerate(self.layers):
            if depth == 0:
                scaled_data = x
            else:
                scaled_data = torch.nn.functional.max_pool1d(scaled_data, kernel_size=2, stride=2)

            out, _ = layer(scaled_data.permute(2, 0, 1))
            out = out.permute(1, 2, 0)
            if depth != 0:
                out = torch.nn.functional.interpolate(out, scale_factor=2**depth, mode='nearest')
            outputs.append(out)

        out = outputs[0]
        for output in outputs[1:]:
            out = out + output

        out, _ = self.final_layer(out.permute(2, 0, 1))

        return out.permute(1, 2, 0)


class NET_VGG_LSTM(nn.Module):
    def __init__(self, num_classes, in_height=32, in_channels=3, dropout_rate=0.0, base_channels=16, conv_blocks=4,
                 subsampling=4, layers_2d=None, **kwargs):
        super(NET_VGG_LSTM, self).__init__()
        self.output_subsampling = subsampling

        self.blocks_2d = VGG_conv_module(base_channels=base_channels, conv_blocks=conv_blocks, subsampling=subsampling,
                                         in_channels=in_channels, layers_2d=layers_2d)
        rnn_channels = self.blocks_2d.out_channels
        self.recurrent_block = MultiscaleRecurrentBlock(rnn_channels, layers_per_scale=2, scales=3)
        self.output_layer = nn.Conv1d(rnn_channels, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.blocks_2d(x)
        out, _ = torch.max(out, 2)
        out = self.recurrent_block(out)
        out = self.output_layer(out)
        return out


class NormalizationBlock(nn.Module):
    def __init__(self, in_channels, dim, normalization_type, embeddings=None, scale_std=0.1):
        super(NormalizationBlock, self).__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.normalization_type = normalization_type
        if self.normalization_type == 'batch_norm':
            if self.dim == 1:
                self.normalization = nn.BatchNorm1d(self.in_channels)
            elif self.dim == 2:
                self.normalization = nn.BatchNorm2d(self.in_channels)
        elif self.normalization_type == 'insta_norm':
            if self.dim == 1:
                self.normalization = nn.InstanceNorm1d(self.in_channels)
            elif self.dim == 2:
                self.normalization = nn.InstanceNorm2d(self.in_channels)
        elif self.normalization_type == 'filter_response_norm':
            if self.dim == 1:
                self.normalization = FilterResponseNorm1d(in_channels=self.in_channels)
            if self.dim == 2:
                self.normalization = FilterResponseNorm2d(in_channels=self.in_channels)
        elif self.normalization_type == 'embed_norm':
            if self.dim == 1:
                self.normalization = EmbedNorm1d(in_channels=self.in_channels, embeddings=embeddings,
                                                 scale_std=scale_std)
            if self.dim == 2:
                self.normalization = EmbedNorm2d(in_channels=self.in_channels, embeddings=embeddings,
                                                 scale_std=scale_std)
        else:
            raise Exception(f'Not implemented normalization: "{self.normalization_type}"')

    def forward(self, x):
        x = self.normalization(x)
        return x


class FilterResponseNorm1d(nn.Module):
    def __init__(self, in_channels):
        super(FilterResponseNorm1d, self).__init__()
        self.in_channels = in_channels
        self.eps = 0.00001
        self.scale = torch.nn.Parameter(torch.Tensor(1, self.in_channels, 1))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.in_channels, 1))
        self.tau = torch.nn.Parameter(torch.Tensor(1, self.in_channels, 1))
        nn.init.zeros_(self.tau)
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.scale)

    def forward(self, x):
        nu2 = torch.mean(torch.pow(x, 2), dim=2, keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        x = x * self.scale + self.bias
        return torch.max(x, self.tau)


class FilterResponseNorm2d(nn.Module):
    def __init__(self, in_channels):
        super(FilterResponseNorm2d, self).__init__()
        self.in_channels = in_channels
        self.eps = 0.00001
        self.scale = torch.nn.Parameter(torch.Tensor(1, self.in_channels, 1, 1))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.in_channels, 1, 1))
        self.tau = torch.nn.Parameter(torch.Tensor(1, self.in_channels, 1, 1))
        nn.init.zeros_(self.tau)
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.scale)

    def forward(self, x):
        nu2 = torch.mean(torch.pow(x, 2), dim=(2, 3), keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        x = x * self.scale + self.bias
        return torch.max(x, self.tau)


class EmbedNorm1d(nn.Module):
    def __init__(self, in_channels, embeddings, scale_std=0.1):
        super(EmbedNorm1d, self).__init__()
        self.in_channels = in_channels
        self.embeddings = embeddings
        self.embed_to_param_net = nn.Conv1d(self.embeddings['dim'], int(self.in_channels * 2), kernel_size=1,
                                            stride=1, padding=0)
        self.embed_to_param_net.weight = nn.Parameter(self.embed_to_param_net.weight * (scale_std / 0.6))
        self.embed_to_param_net.bias = nn.Parameter(torch.zeros_like(self.embed_to_param_net.bias))
        self.std_in = None
        self.mean_in = None
        self.scale = None
        self.bias = None

    def forward(self, x):
        self.std_in, self.mean_in = torch.std_mean(x, dim=2, keepdim=True)
        embed_to_param_net_out = self.embed_to_param_net(self.embeddings['data'].unsqueeze(2))
        self.scale, self.bias = torch.chunk(embed_to_param_net_out, 2, dim=1)
        self.scale = self.scale.contiguous()
        self.bias = self.bias.contiguous()
        self.scale += 1
        out = (x - self.mean_in) / (self.std_in + 0.00001)
        return out * self.scale + self.bias


class EmbedNorm2d(nn.Module):
    def __init__(self, in_channels, embeddings, scale_std=0.1):
        super(EmbedNorm2d, self).__init__()
        self.in_channels = in_channels
        self.embeddings = embeddings
        self.embed_to_param_net = nn.Conv2d(self.embeddings['dim'], int(self.in_channels * 2), kernel_size=1,
                                            stride=1, padding=0)
        self.embed_to_param_net.weight = nn.Parameter(self.embed_to_param_net.weight * (scale_std / 0.6))
        self.embed_to_param_net.bias = nn.Parameter(torch.zeros_like(self.embed_to_param_net.bias))
        self.std_in = None
        self.mean_in = None
        self.scale = None
        self.bias = None

    def forward(self, x):
        self.std_in, self.mean_in = torch.std_mean(x, dim=(2, 3), keepdim=True)
        embed_to_param_net_out = self.embed_to_param_net(self.embeddings['data'].unsqueeze(2).unsqueeze(3))
        self.scale, self.bias = torch.chunk(embed_to_param_net_out, 2, dim=1)
        self.scale = self.scale.contiguous()
        self.bias = self.bias.contiguous()
        self.scale += 1
        out = (x - self.mean_in) / (self.std_in + 0.00001)
        return out * self.scale + self.bias


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim=2, kernel_size=3, stride=1, padding=1,
                 normalization_type=None, activation=None, embeddings=None, normalization_scale_std=0.1):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.normalization_type = normalization_type
        self.activation = None
        if self.dim == 1:
            self.conv = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride,
                                  padding=self.padding)
        elif self.dim == 2:
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride,
                                  padding=self.padding)
        else:
            raise Exception(f'Not implemented conv. dim.: "{self.dim}"')
        if normalization_type is not None:
            self.normalization = NormalizationBlock(in_channels=self.out_channels,
                                                    dim=self.dim,
                                                    normalization_type=self.normalization_type,
                                                    embeddings=embeddings, scale_std=normalization_scale_std)
        else:
            self.normalization = None
        if activation is not None:
            self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        if self.normalization is not None:
            x = self.normalization(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SequentialConvBlock(nn.Module):
    def __init__(self, conv_count, in_channels, out_channels, dim=2, kernel_size=3, stride=1, padding=1,
                 normalization_type=None, activation=None, embeddings=None, normalization_scale_std=0.1):
        super(SequentialConvBlock, self).__init__()
        self.conv_count = conv_count
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if normalization_type is not None:
            self.normalization_type = normalization_type
        else:
            self.normalization_type = [None] * self.conv_count
        if activation is not None:
            self.activation = activation
        else:
            self.activation = [None] * self.conv_count
        self.conv_blocks = []
        actual_in_channels = in_channels
        for i in range(conv_count):
            self.conv_blocks.append(ConvBlock(actual_in_channels, out_channels, dim=self.dim,
                                              kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                              normalization_type=self.normalization_type[i], activation=self.activation[i],
                                              embeddings=embeddings, normalization_scale_std=normalization_scale_std))
            actual_in_channels = out_channels
        self.conv_blocks = nn.Sequential(*self.conv_blocks)

    def forward(self, x):
        return self.conv_blocks(x)


class DownSampleBlock(nn.Module):
    def __init__(self, inner_block, kernel_size=(2, 2), stride=2, dropout_rate=0.0, normalization_type=None,
                 embeddings=None, normalization_scale_std=0.1):
        super(DownSampleBlock, self).__init__()
        self.dim = inner_block.dim
        self.inner_block = inner_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        if self.dim == 1:
            self.pooling = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride)
        elif self.dim == 2:
            self.pooling = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)
        else:
            raise Exception(f'Not implemented down sample for dim.: "{self.dim}"')
        if normalization_type is not None:
            self.normalization = NormalizationBlock(in_channels=self.inner_block.out_channels,
                                                    dim=self.dim,
                                                    normalization_type=normalization_type,
                                                    embeddings=embeddings,
                                                    scale_std=normalization_scale_std)
        else:
            self.normalization = None
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        x = self.inner_block(x)
        x = self.pooling(x)
        if self.normalization is not None:
            x = self.normalization(x)
        x = self.dropout(x)
        return x


class AggregationBlock(nn.Module):
    def __init__(self, kernel_height, in_channels, out_channels, normalization_type='batch_norm', activation=None,
                 embeddings=None, normalization_scale_std=0.1):
        super(AggregationBlock, self).__init__()
        self.kernel_height = kernel_height
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization_type = normalization_type
        self.conv_block = ConvBlock(self.in_channels, self.out_channels, dim=2, kernel_size=(kernel_height, 1),
                                    stride=1, padding=0, normalization_type=self.normalization_type, activation=activation,
                                    embeddings=embeddings, normalization_scale_std=normalization_scale_std)

    def forward(self, x):
        x = self.conv_block(x)
        return x[:, :, 0, :]


class NetRecurrent(nn.Module):
    def __init__(self, num_classes, in_height, in_channels=3, dropout_rate=0.0, down_sample_block_count=2,
                 down_sample_block_layer_count=1, down_sample_base_filter_count=6, recurrent_layer_count=1,
                 output_subsampling=4, recurrent_type="LSTM", normalization_type='batch_norm', activation=nn.ReLU,
                 num_embeddings=0, embedding_dim=0, normalization_scale_std=0.6, **kwargs):
        super(NetRecurrent, self).__init__()
        self.num_classes = num_classes
        self.in_height = in_height
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.down_sample_block_count = down_sample_block_count
        self.down_sample_block_layer_count = down_sample_block_layer_count
        self.down_sample_base_filter_count = down_sample_base_filter_count
        self.recurrent_layer_count = recurrent_layer_count
        self.output_subsampling = output_subsampling
        self.recurrent_type = recurrent_type
        self.normalization_type = normalization_type
        self.normalization_counter = 0
        self.increase_normalization_counter = False
        self.embeddings_layer = None
        self.embeddings = {'dim': embedding_dim}
        self.activation = activation
        if type(self.normalization_type) is list:
            self.increase_normalization_counter = True
            if 'embed_norm' in normalization_type:
                self.embeddings_layer = nn.Embedding(num_embeddings, embedding_dim, scale_grad_by_freq=True)
        else:
            self.normalization_type = [self.normalization_type]
            if normalization_type == 'embed_norm':
                self.embeddings_layer = nn.Embedding(num_embeddings, embedding_dim, scale_grad_by_freq=True)

        # DOWN SAMPLING
        self.down_sample_blocks = []
        actual_in_channels = self.in_channels
        actual_in_height = self.in_height
        for i in range(self.down_sample_block_count):
            actual_out_channels = self.down_sample_base_filter_count * (2 ** i)
            normalization_type = self.normalization_type[self.normalization_counter]
            activation = []
            if normalization_type == 'filter_response_norm' or normalization_type == 'filter_response_adaptive_norm':
                activation += [self.activation] * (self.down_sample_block_layer_count - 1)
                activation.append(None)
            else:
                activation += [self.activation] * self.down_sample_block_layer_count
            sequential_conv_block = SequentialConvBlock(conv_count=self.down_sample_block_layer_count,
                                                        in_channels=actual_in_channels,
                                                        out_channels=actual_out_channels,
                                                        dim=2,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=1,
                                                        activation=activation)
            if 2 ** i < output_subsampling:
                down_sample_kernel = (2, 2)
            else:
                down_sample_kernel = (2, 1)
            down_sample_block = DownSampleBlock(inner_block=sequential_conv_block,
                                                kernel_size=down_sample_kernel,
                                                stride=down_sample_kernel,
                                                dropout_rate=self.dropout_rate,
                                                normalization_type=normalization_type,
                                                embeddings=self.embeddings,
                                                normalization_scale_std=normalization_scale_std)
            self.down_sample_blocks.append(down_sample_block)
            actual_in_channels = actual_out_channels
            actual_in_height = actual_in_height // 2

            if self.increase_normalization_counter:
                self.normalization_counter += 1

        actual_out_channels = self.down_sample_base_filter_count * (2 ** self.down_sample_block_count)
        self.down_sample_blocks.append(SequentialConvBlock(conv_count=1,
                                                           in_channels=actual_in_channels,
                                                           out_channels=actual_out_channels,
                                                           dim=2,
                                                           kernel_size=3,
                                                           stride=1,
                                                           padding=1,
                                                           activation=[self.activation],
                                                           embeddings=self.embeddings,
                                                           normalization_scale_std=normalization_scale_std))
        actual_in_channels = actual_out_channels
        self.down_sample_blocks = nn.Sequential(*self.down_sample_blocks)

        # AGGREGATION
        if normalization_type == 'filter_response_norm' or normalization_type == 'filter_response_adaptive_norm':
            activation = None
        else:
            activation = self.activation
        self.aggregation_block = []
        self.aggregation_block.append(AggregationBlock(kernel_height=actual_in_height,
                                                       in_channels=actual_in_channels,
                                                       out_channels=actual_out_channels,
                                                       normalization_type=self.normalization_type[self.normalization_counter],
                                                       activation=activation,
                                                       embeddings=self.embeddings,
                                                       normalization_scale_std=normalization_scale_std))
        if self.increase_normalization_counter:
            self.normalization_counter += 1
        self.aggregation_block = nn.Sequential(*self.aggregation_block)

        # RECURRENT
        self.recurrent_block = []
        self.recurrent_block.append(MultiscaleRecurrentBlock(channels=actual_in_channels,
                                                             layers_per_scale=self.recurrent_layer_count,
                                                             scales=3))
        self.recurrent_block.append(NormalizationBlock(in_channels=actual_out_channels,
                                                       dim=1,
                                                       normalization_type=self.normalization_type[self.normalization_counter],
                                                       embeddings=self.embeddings,
                                                       scale_std=normalization_scale_std))
        self.recurrent_block = nn.Sequential(*self.recurrent_block)

        # OUTPUT
        self.output = nn.Conv1d(actual_in_channels, self.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, ids_embedding=None, **kwargs):
        if self.embeddings_layer is not None and ids_embedding is not None:
            self.embeddings['data'] = self.embeddings_layer(ids_embedding)
        x = x.permute(0, 3, 1, 2)
        x = self.down_sample_blocks(x)
        x = self.aggregation_block(x)
        x = self.recurrent_block(x)
        x = self.output(x)
        return x


PYTORCH_NETS = {
    "VGG_B32_L16_S4_CB4": partial(NET_VGG, in_channels=3, base_channels=32, conv_blocks=4, subsampling=4, layers_2d=6),
    "VGG_LSTM_B64_L17_S4_CB4": partial(NET_VGG_LSTM, in_channels=3, base_channels=64, conv_blocks=4, subsampling=4,
                                       layers_2d=17),
    "LSTM_BC_3_BLC_2_BFC_64_EMBED_32": partial(NetRecurrent, down_sample_block_count=3, down_sample_block_layer_count=2,
                                               down_sample_base_filter_count=64, recurrent_layer_count=2,
                                               output_subsampling=4, recurrent_type="LSTM", embedding_dim=32,
                                               normalization_type=['filter_response_norm',
                                                                   'filter_response_norm',
                                                                   'filter_response_norm',
                                                                   'filter_response_norm',
                                                                   'embed_norm'])
}

