"""Dataloader helpers for full-window attention diagnostics (historical path)."""
import torch
import torch.utils.data as Data
import numpy as np



def merge_adj(pred):
    arr = 0
    for i, val in enumerate(pred):
        if i == 0:
            arr = val
        else:
            arr = np.concatenate((arr, val), axis=0)
    return arr




def get_batch(eod_data, offset, seq_len, out_len):

    return eod_data[:, offset:offset + seq_len, :], \
           eod_data[:, offset + seq_len:offset + seq_len + out_len, 0]




def prepare_all_dataloaders(eod_data, args):
    # Full-sample sliding windows for attention diagnostics
    EOD, GT = [], []

    for i in range(eod_data.shape[1] - args.length - args.forecast_horizon + 1):
        eod, gt = get_batch(eod_data, i, args.length, args.forecast_horizon)
        EOD.append(eod)

        gt = gt.reshape(args.code_num, -1)
        GT.append(gt)

    GT = merge_adj(GT).reshape(-1, args.code_num, args.forecast_horizon)
    EOD, GT = np.array(EOD), np.array(GT)  # , np.array(CORR)

    train_eod, train_gt = EOD, GT   #, ID[:args.train_index]

    train_eod = torch.FloatTensor(train_eod)
    train_gt = torch.FloatTensor(train_gt)

    train_dataset = Data.TensorDataset(train_eod, train_gt)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader