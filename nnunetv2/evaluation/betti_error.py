import sys
import os
current_working_dir = os.getcwd()
sys.path.append(current_working_dir)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from skimage.measure import euler_number, label

def betti_numbers(img: np.array):
    """
    calculates the Betti number B0, B1, and B2 for a 3D img
    from the Euler characteristic number

    code prototyped by
    - Martin Menten (Imperial College)
    - Suprosanna Shit (Technical University Munich)
    - Johannes C. Paetzold (Imperial College)
    """
    img = img.squeeze()
    assert img.ndim == 3

    N6 = 1
    N26 = 3

    padded = np.pad(img, pad_width=1)

    assert set(np.unique(padded)).issubset({0, 1})

    _, b0 = label(
        padded,
        # return the number of assigned labels
        return_num=True,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    euler_char_num = euler_number(
        padded,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    _, b2 = label(
        1 - padded,
        # return the number of assigned labels
        return_num=True,
        # 6 neighborhoods for background
        connectivity=N6,
    )

    b2 -= 1

    b1 = b0 + b2 - euler_char_num  # Euler number = Betti:0 - Bett:1 + Betti:2

    return [b0, b1, b2]

def betti_error(betti_1, betti_2):
    if not isinstance(betti_1,np.ndarray):
        betti_1 = np.array(betti_1)
    if not isinstance(betti_2,np.ndarray):
        betti_2 = np.array(betti_2)
    return np.sum(np.abs(betti_1 - betti_2))

def compute_betti_error(pred, gt):
    if len(pred.shape) > 3:
        pred = pred.squeeze()
    if len(gt.shape) > 3:
        gt = gt.squeeze()
    betti_pred = betti_numbers(pred)
    betti_gt = betti_numbers(gt)
    return betti_error(betti_pred, betti_gt)