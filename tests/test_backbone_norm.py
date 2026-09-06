import unittest

import torch

from model.gwnet_att import ChannelLayerNorm, gwnet


class BackboneNormTests(unittest.TestCase):
    def test_channel_norm_is_local_and_train_eval_consistent(self):
        torch.manual_seed(1)
        norm = ChannelLayerNorm(4)
        x = torch.randn(2, 4, 3, 5)
        expected = norm(x)
        norm.eval()
        torch.testing.assert_close(expected, norm(x))
        changed = x.clone()
        changed[1] *= 100
        changed[0, :, 1:, :] += 20
        changed[0, :, 0, 1:] -= 20
        torch.testing.assert_close(expected[0, :, 0, 0], norm(changed)[0, :, 0, 0])
        torch.testing.assert_close(expected.mean(dim=1), torch.zeros(2, 3, 5), atol=1e-6, rtol=0)

    def test_forecast_and_gradients(self):
        torch.manual_seed(2)
        m = gwnet(3, 8, out_dim=2, blocks=1, layers=2,
                  residual_channels=4, dilation_channels=4, skip_channels=8,
                  end_channels=8, backbone_norm='channel').eval()
        x = torch.randn(2, 3, 8, 1)
        y = m(x)
        self.assertEqual(tuple(y.shape), (2, 3, 2))
        torch.testing.assert_close(m(x[:1]), y[:1], atol=1e-6, rtol=1e-5)
        y[:, 0].square().mean().backward()
        for p in (m.bn[0].weight, m.weight_key):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
