"""Test-only entry point: reloads a checkpoint written by RADL.py and re-emits
its result table.  The checkpoint path is derived from the same run tag, so the
flags here must match the ones the run was trained with.
"""
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from load_data import load_EOD_data, prepare_dataloaders
from tool import test
from config import get_args, set_split_indices
from set_model import get_model


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    eod_data, a, b = load_EOD_data(args)
    _, _, test_loader = prepare_dataloaders(eod_data, args)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model = get_model(device, args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return test(model, test_loader, optimizer, device, args, a, b)


if __name__ == '__main__':
    args = get_args()
    set_split_indices(args)
    all_result = main(args)
    print(all_result.to_string(index=False))
