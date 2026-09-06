"""Synthetic invariants only; no dataset loading or training."""
import unittest

import torch

from model.gwnet_att import gwnet, nconv


class LagGraphTests(unittest.TestCase):
    def model(self, **kwargs):
        kwargs.setdefault('use_lag_graph', True)
        return gwnet(3, 8, blocks=1, layers=2, residual_channels=4,
                     dilation_channels=4, skip_channels=8, end_channels=8,
                     **kwargs)

    def test_delayed_source_and_no_wrap(self):
        x = torch.zeros(1, 1, 2, 6)
        x[0, 0, 0] = torch.arange(1., 7.)
        weights = torch.zeros(1, 2, 2, 2)
        weights[0, 1, 0, 1] = 1
        out = nconv()(x, (weights, (0, 2)))
        torch.testing.assert_close(out[0, 0, 1], torch.tensor([0., 0., 1., 2., 3., 4.]))
        x[..., -1] = 1000
        torch.testing.assert_close(nconv()(x, (weights, (0, 2))), out)

    def test_normalization_shapes_gradients_and_batch_independence(self):
        torch.manual_seed(7)
        m = self.model(out_dim=3).eval()
        x = torch.randn(2, 3, 8, 1)
        out, lag_att = m(x, return_lag_att=True)
        self.assertEqual(tuple(out.shape), (2, 3, 3))
        self.assertEqual(tuple(lag_att.shape), (2, 4, 3, 3))
        torch.testing.assert_close(lag_att.sum(dim=(1, 2)), torch.ones(2, 3))
        self.assertEqual(lag_att[:, 1:].diagonal(dim1=-2, dim2=-1).count_nonzero().item(), 0)
        _, summed = m(x, return_att=True)
        torch.testing.assert_close(summed, lag_att.sum(dim=1))
        torch.testing.assert_close(m(x[:1]), out[:1], atol=1e-6, rtol=1e-5)
        out[:, 0].square().mean().backward()
        for p in (m.weight_key, m.weight_query):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_zero_lag_matches_original_path(self):
        # With only lag 0, preserving the temporal axis must give the same
        # forecast as the original valid-convolution path at evaluation.
        torch.manual_seed(9)
        old = self.model(use_lag_graph=False).eval()
        new = self.model(graph_lags=(0,)).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        x = torch.randn(2, 3, 8, 1)
        torch.testing.assert_close(old(x), new(x), atol=1e-6, rtol=1e-5)

    def test_invalid_lags(self):
        for lags in ((), (1, 2), (0, -1), (0, 8), (0, 1, 1)):
            with self.assertRaises(ValueError):
                self.model(graph_lags=lags)


if __name__ == '__main__':
    unittest.main()
