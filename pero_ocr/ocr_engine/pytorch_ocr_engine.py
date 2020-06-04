# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
from torch import nn
import numpy as np
from .line_ocr_engine import BaseEngineLineOCR


# scores_probs should be N,C,T, blank is last class
def greedy_decode_ctc(scores_probs, chars):
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
        super(PytorchEngineLineOCR, self).__init__(json_def, gpu_id=0, batch_size=8)

        self.net_subsampling = 4
        self.characters = list(self.characters) + ['|']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = PYTORCH_NETS[self.net_name]
        self.model = net[0](num_classes=len(self.characters), in_height=self.line_px_height, **net[1])
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def run_ocr(self, batch_data):
        with torch.no_grad():
            batch_data = torch.from_numpy(batch_data).to(self.device).float() / 255.0
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
    def __init__(self, num_classes, in_height=32, base_channels=16, conv_blocks=4, subsampling=4, in_channels=3, layers_2d=None):
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
                 subsampling=4, layers_2d=None):
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


PYTORCH_NETS = {
    "VGG_B32_L16_S4_CB4": (NET_VGG, {'in_channels': 3, 'base_channels': 32, 'conv_blocks': 4, 'subsampling': 4, 'layers_2d': 16}),
    "VGG_LSTM_B64_L17_S4_CB4": (NET_VGG_LSTM, {'in_channels': 3, 'base_channels': 64, 'conv_blocks': 4, 'subsampling': 4, 'layers_2d': 17})
}
