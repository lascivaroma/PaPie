import unittest

import torch
from torch.nn.utils.rnn import pad_packed_sequence

from pie import torch_utils


class TestTorchUtils(unittest.TestCase):
    def test_pack_sort_roundtrip(self):
        inp = torch.tensor([[1, 3], [2, 4], [0, 5]], dtype=torch.float).unsqueeze(-1)
        packed, unsort = torch_utils.pack_sort(inp, torch.tensor([2, 3]))
        out, lengths = pad_packed_sequence(packed)
        self.assertEqual(lengths.tolist(), [3, 2])  # sorted by length
        self.assertTrue(torch.equal(out[:, unsort], inp))

    def test_get_last_token(self):
        t = torch.tensor([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]])
        last = torch_utils.get_last_token(t, [3, 2, 1])
        self.assertEqual(last.tolist(), [[3], [3], [3]])

    def test_pad_and_flatten(self):
        emb = torch.arange(6).unsqueeze(-1)
        nwords = torch.tensor([3, 1, 2])
        padded = torch_utils.pad_flat_batch(emb, nwords)
        self.assertEqual(tuple(padded.shape), (3, 3, 1))
        flat = torch_utils.flatten_padded_batch(padded, nwords)
        self.assertTrue(torch.equal(flat, emb))
        with self.assertRaises(ValueError):
            torch_utils.pad_flat_batch(emb, torch.tensor([1, 1]))

    def test_init_hidden(self):
        inp = torch.zeros(5, 4, 8)
        h, c = torch_utils.init_hidden_for(inp, 2, 1, 16, 'LSTM')
        self.assertEqual(tuple(h.shape), (2, 4, 16))
        self.assertEqual(tuple(c.shape), (2, 4, 16))
        h = torch_utils.init_hidden_for(inp, 1, 2, 16, 'GRU')
        self.assertEqual(tuple(h.shape), (2, 4, 16))
