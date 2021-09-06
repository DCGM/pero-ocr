import unittest
import torch
import numpy as np
import os

from pero_ocr.decoding.lm_wrapper import LMWrapper, HiddenState


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._model_i = torch.nn.Embedding(4, 1)
        self._model_i.weight[0, 0] = 0
        self._model_i.weight[1, 0] = 1
        self._model_i.weight[2, 0] = 2
        self._model_i.weight[3, 0] = 3

        self._model_r = torch.nn.Linear(1, 1)
        self._model_r.weight[0, 0] = 2
        self._model_r.bias[0] = -1

    def forward(self, xs, hs):
        out_h = "rubbish, never to be accessed"
        last_h = self._model_i(xs)[:, :, 0] + self._model_r(hs)
        return out_h, last_h

    def init_hidden(self, bsz):
        ref = self._model_r.weight
        return torch.ones((1, bsz, 1), dtype=ref.dtype, device=ref.device) * 10

    def process_seq(self, seq, h0):
        h = h0
        hs = []
        for x in seq:
            _, h = self.forward(x.unsqueeze(0), h)
            hs.append(h)

        return torch.cat(hs, dim=1)


class DummyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._model_o = torch.nn.Linear(1, 4)
        self._model_o.weight[0, 0] = -100
        self._model_o.weight[1, 0] = 2
        self._model_o.weight[2, 0] = 0
        self._model_o.weight[3, 0] = 5
        self._model_o.bias[0] = -100
        self._model_o.bias[1] = 1
        self._model_o.bias[2] = 3
        self._model_o.bias[3] = -4

    def forward(self, hs):
        return self._model_o(hs)

    def neg_log_prob(self, hs, targets):
        preds = self.forward(hs)
        targets_flat = targets.view(-1)
        preds_flat = preds.view(-1, preds.size(-1))

        return preds_flat[torch.arange(0, preds_flat.shape[0]), targets_flat].sum(), preds_flat.size(0)


class DummyLm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyModel()
        self.decoder = DummyDecoder()
        self.vocab = {
            '</s>': 0,
            'a': 1,
            'b': 2,
            'c': 3,
        }

        self._unused_prefix_len = 1

    def single_sentence_nll(self, sentence, prefix):
        sentence_ids = [self.vocab[c] for c in sentence]

        if prefix:
            prefix_id = self.vocab[prefix]
            tensor = torch.tensor([prefix_id] + sentence_ids).view(-1, 1)
        else:
            tensor = torch.tensor(sentence_ids).view(-1, 1)

        h0 = self.model.init_hidden(1)
        o = self.model.process_seq(tensor[:-1], h0).detach()

        if prefix:
            nll, _ = self.decoder.neg_log_prob(o, tensor[1:])
        else:
            nll, _ = self.decoder.neg_log_prob(torch.cat([h0[0][0].unsqueeze(1), o]), tensor)

        return nll.item()


class HiddenStateTests(unittest.TestCase):
    def test_keeps_value(self):
        in_h = torch.from_numpy(np.asarray([-2.0])).to(dtype=torch.float32)
        h0 = HiddenState(in_h)
        self.assertEqual(h0.prepare_for_torch(), in_h)

    def test_keeps_value_array(self):
        in_h = torch.from_numpy(np.asarray([-2.0, 3.0])).to(dtype=torch.float32)
        h0 = HiddenState(in_h)
        self.assertTrue(torch.equal(h0.prepare_for_torch(), in_h))

    def test_keeps_value_lstm_double_state(self):
        in_h = torch.from_numpy(np.asarray([[-1.0, 3.0], [-2.0, 4.0]])).to(dtype=torch.float32)
        in_c = torch.from_numpy(np.asarray([[-1.0, 3.0], [-2.0, 4.0]])).to(dtype=torch.float32)
        h0 = HiddenState((in_h, in_c))
        self.assertTrue(torch.equal(h0.prepare_for_torch()[0], in_h))
        self.assertTrue(torch.equal(h0.prepare_for_torch()[1], in_c))


class LMWrapperTemplate:
    '''Set of tests that any LM wrapper needs to pass.

    Inheriting class needs to:
    - instantiate an LM wrapper in self._wrapper
    - prepare proper PyTorch device in self._device
    '''
    def test_single_advance(self):
        h0 = HiddenState(torch.tensor([-2.0]).to(self._device))
        h1 = self._wrapper.advance_h0(np.asarray([2]), h0)
        self.assertTrue((h1._h == torch.tensor([-2.0]).to(self._device)).all())

    def test_advance_batch(self):
        h0 = HiddenState(torch.tensor([[-2.0], [2.0]]).to(self._device))
        h1 = self._wrapper.advance_h0(np.asarray([2, 0]), h0)
        expected = torch.tensor([[-2.0], [4.0]]).to(self._device)
        self.assertTrue((h1._h == expected).all())

    def test_single_output(self):
        h = HiddenState(torch.tensor([[[1.0]]]).to(self._device))
        y = self._wrapper.log_probs(h)
        self.assertTrue((y == np.asarray([3, 3, 1])).all())

    def test_batch_output(self):
        h = HiddenState(torch.tensor([[[1.0], [-2.0]]]).to(self._device))
        y = self._wrapper.log_probs(h)
        self.assertTrue((y == np.asarray([[3, 3, 1], [-3, 3, -14]])).all())

    def test_initial_state(self):
        h_init = self._wrapper.initial_h(1)
        self.assertTrue((h_init._h == torch.tensor([[19.0]]).to(self._device)).all())

    def test_single_translation(self):
        translation = self._wrapper.translate(np.asarray([0]))
        self.assertTrue((translation == np.asarray([1])).all())

    def test_batch_translation(self):
        translation = self._wrapper.translate(np.asarray([0, 1]))
        self.assertTrue((translation == np.asarray([1, 2])).all())

    def test_prefix_cutting_translation(self):
        translation = self._wrapper.translate(np.asarray([0, 1]))
        self.assertTrue((translation == np.asarray([1, 2])).all())

    def test_eos_score(self):
        h = HiddenState(torch.tensor([[[0.0], [0.2]]]).to(self._device))
        eos_scores = self._wrapper.eos_scores(h)
        self.assertTrue((eos_scores == np.asarray([-100.0, -120.0])).all())


class TorchCPULmWrapperTests(LMWrapperTemplate, unittest.TestCase):
    def setUp(self):
        self._wrapper = LMWrapper(DummyLm(), ['a', 'b', 'c'])
        self._device = torch.device('cpu')


@unittest.skipIf(os.environ.get('TEST_CUDA') != 'yes', "For GPU tests, set TEST_CUDA='yes'")
class GPULMWrapperTests(LMWrapperTemplate, unittest.TestCase):
    def setUp(self):
        self._wrapper = LMWrapper(DummyLm(), ['a', 'b', 'c'], lm_on_gpu=True)
        self._device = torch.device('cuda')
