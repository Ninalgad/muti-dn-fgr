# %%writefile /content/src/eval.py
import gc
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from utils import load_mask, load_ultra, dice, select_fetal_abdomen_mask_and_frame


class TestImageDataset(Dataset):
    def __init__(self, x):
        self.x = x
        self.n = len(self.x)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = load_ultra(self.x[idx])
        return image


def predict(images, model, device, batch_size=16, verbose=False):
    model.eval()
    test_loader = DataLoader(TestImageDataset(images), batch_size=batch_size,
                             shuffle=False)
    seg_pred, dist_pred = [], []
    if verbose:
        test_loader = tqdm(test_loader)

    for x in test_loader:
        x = x.to(device)
        s, d = model(x)
        s = F.sigmoid(s).detach().cpu().numpy()
        d = F.sigmoid(d).detach().cpu().numpy()

        seg_pred.append(s)
        dist_pred.append(d)
    return np.concatenate(seg_pred, axis=0), np.concatenate(dist_pred, axis=0)


def optimize_thresh(y_true, y_pred, debug=False):
    if np.all(y_true == 0):
        return 0, .5

    assert y_pred.shape == y_true.shape

    sup_ = min([x.max() for x in y_pred])
    inf_ = max([x.min() for x in y_pred])

    best_dice, best_thr = 0, 0
    for t in np.linspace(sup_, inf_, num=200, endpoint=True):
        pt = (y_pred > t).astype('float32')

        d = dice(pt, y_true)
        if d > best_dice:
            best_dice = d
            best_thr = t

        if debug:
            break

    return best_dice, best_thr


def get_largest_frame(frame_probabilities, segmentation_map):
    n = int(np.argmax(frame_probabilities))

    if segmentation_map[n].max() == 0:
        return -1, np.zeros_like(segmentation_map[0])

    return n, segmentation_map[n]


def evaluate_metric(test_df, model, device, debug=False):
    tar_mask_frame, tar_frame_num = [], []
    tar_mask_of_pred_fn = []
    pred_prob_frame, pred_frame_num = [], []

    wfss = []

    for p, p_df in tqdm(test_df.groupby("patient"), total=len(test_df["patient"].unique())):

        if debug:
            p_df = p_df.iloc[:2]

        mask_true = np.array([load_mask(p) for p in p_df.label.values], 'float32')
        mask_pred, dist_pred = predict(p_df.image.values, model, device,
                                       batch_size=16, verbose=False)

        yf, yn = select_fetal_abdomen_mask_and_frame(mask_true)

        pn = np.argmax(dist_pred)
        pf = mask_pred[pn]

        yf_pn = mask_true[pn]

        tar_mask_frame.append(yf)
        tar_frame_num.append(yn)
        pred_prob_frame.append(pf)
        pred_frame_num.append(pn)
        tar_mask_of_pred_fn.append(yf_pn)

        s = 0
        if yn == pn:
            s = 1
        elif np.any(yf_pn != 0):
            s = .6
        wfss.append(s)

        del mask_true, mask_pred, dist_pred, yf, pf, p_df
        gc.collect()

    tar_mask_frame = np.array(tar_mask_frame, "uint8")
    pred_prob_frame = np.array(pred_prob_frame, "float32")
    tar_mask_of_pred_fn = np.array(tar_mask_of_pred_fn, "uint8")

    dsc, thr = optimize_thresh(tar_mask_of_pred_fn, pred_prob_frame, debug=debug)
    pred_mask_frame = (pred_prob_frame > thr).astype("uint8")
    del pred_prob_frame

    # nae = np.abs(tar_mask_frame - pred_mask_frame).sum() / max([1e-5, tar_mask_frame.sum(), pred_mask_frame.sum()])
    nae = dice(pred_mask_frame, tar_mask_frame)  # not nae
    wfss = np.mean(wfss)

    print(f"NAE: {nae:.4f}, DSC: {dsc:.4f}, WFSS: {wfss:.4f}")

    return .5 * nae + .25 * dsc + .25 * wfss, thr
