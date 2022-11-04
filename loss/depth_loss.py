from __future__ import division
import torch
import numpy as np


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def get_depth_grad_loss(depth_pred, depth_gt, valid, inverse_depth_loss=True):
    # default is on inverse depth
    # both: [B, H, W]
    assert depth_pred.dim() == 3 and depth_gt.dim() == 3 and valid.dim() == 3

    valid = valid > 0.5
    valid_x = valid[:, :, :-1] & valid[:, :, 1:]
    valid_y = valid[:, :-1, :] & valid[:, 1:, :]

    if valid_x.max() < 0.5 or valid_y.max() < 0.5:  # no valid pixel
        return 0.

    if inverse_depth_loss:
        grad_pred_x = torch.abs(1. / depth_pred[:, :, :-1][valid_x] - 1. / depth_pred[:, :, 1:][valid_x])
        grad_pred_y = torch.abs(1. / depth_pred[:, :-1, :][valid_y] - 1. / depth_pred[:, 1:, :][valid_y])

        grad_gt_x = torch.abs(1. / depth_gt[:, :, :-1][valid_x] - 1. / depth_gt[:, :, 1:][valid_x])
        grad_gt_y = torch.abs(1. / depth_gt[:, :-1, :][valid_y] - 1. / depth_gt[:, 1:, :][valid_y])
    else:
        grad_pred_x = torch.abs((depth_pred[:, :, :-1] - depth_pred[:, :, 1:])[valid_x])
        grad_pred_y = torch.abs((depth_pred[:, :-1, :] - depth_pred[:, 1:, :])[valid_y])

        grad_gt_x = torch.abs((depth_gt[:, :, :-1] - depth_gt[:, :, 1:])[valid_x])
        grad_gt_y = torch.abs((depth_gt[:, :-1, :] - depth_gt[:, 1:, :])[valid_y])

    loss_grad_x = torch.abs(grad_pred_x - grad_gt_x).mean()
    loss_grad_y = torch.abs(grad_pred_y - grad_gt_y).mean()

    return loss_grad_x + loss_grad_y


def depth_grad_loss_func(depth_preds, depth_gt, valid,
                         inverse_depth_loss=True,
                         gamma=0.9):
    num = len(depth_preds)
    loss = 0.

    for i in range(num):
        weight = gamma ** (num - i - 1)
        loss += weight * get_depth_grad_loss(depth_preds[i], depth_gt, valid,
                                             inverse_depth_loss=inverse_depth_loss)

    return loss


def depth_loss_func(depth_preds, depth_gt, valid, gamma=0.9,
                    ):
    """ loss function defined over multiple depth predictions """

    n_predictions = len(depth_preds)
    depth_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        # inverse depth loss
        valid_bool = valid > 0.5
        if valid_bool.max() < 0.5:  # no valid pixel
            i_loss = 0.
        else:
            i_loss = (1. / depth_preds[i][valid_bool] - 1. / depth_gt[valid_bool]).abs().mean()

        depth_loss += i_weight * i_loss

    return depth_loss
