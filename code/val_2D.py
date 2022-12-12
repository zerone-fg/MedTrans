import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[224, 224]):
    slice, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    # for ind in range(image.shape[0]):
    #     slice = image[ind, :, :]
    x, y = slice.shape[1], slice.shape[2]
        # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
            0).float().cuda()
    net.eval()
    with torch.no_grad():
        output_cnn, output_trans, _, _ = net(input)
        out_cnn = torch.softmax(output_cnn, dim=1)
        out_trans = torch.softmax(output_trans, dim=1)
        out = (out_cnn + out_trans) / 2
        # out = net(input)
        #
        out = torch.argmax(out, dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        # prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            pred == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[224, 224]):
    # image, label = image.squeeze(0).cpu().detach(
    # ).numpy(), label.squeeze(0).cpu().detach().numpy()
    # prediction = np.zeros_like(label)
    # for ind in range(image.shape[0]):
    #     slice = image[ind, :, :]
    #     x, y = slice.shape[0], slice.shape[1]
    #     slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    #     input = torch.from_numpy(slice).unsqueeze(
    #         0).unsqueeze(0).float().cuda()
    #     net.eval()
    #     with torch.no_grad():
    #         output_main, _, _, _ = net(input)
    #         out = torch.argmax(torch.softmax(
    #             output_main, dim=1), dim=1).squeeze(0)
    #         out = out.cpu().detach().numpy()
    #         pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
    #         prediction[ind] = pred
    # metric_list = []
    # for i in range(1, classes):
    #     metric_list.append(calculate_metric_percase(
    #         prediction == i, label == i))
    # return metric_list
    slice, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    # for ind in range(image.shape[0]):
    #     slice = image[ind, :, :]
    x, y = slice.shape[1], slice.shape[2]
    # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        # output_cnn, output_trans, _, _ = net(input)
        # out_cnn = torch.softmax(output_cnn, dim=1)
        # out_trans = torch.softmax(output_trans, dim=1)
        # out = (out_cnn + out_trans) / 2
        out, _, _, _ = net(input)

        out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        # prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            pred == i, label == i))
    return metric_list
