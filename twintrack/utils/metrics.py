import numpy as np

def compute_mota(fp, fn, ids, gt):
    """
    Multiple Object Tracking Accuracy (MOTA)
    Args:
        fp: int, false positives
        fn: int, false negatives
        ids: int, identity switches
        gt: int, number of ground-truth objects
    Returns: float
    """
    return 1.0 - float(fp + fn + ids) / (gt + 1e-6)

def compute_idf1(idtp, idfp, idfn):
    """
    Identity F1 Score (IDF1)
    Args:
        idtp: int, true positive IDs
        idfp: int, false positive IDs
        idfn: int, false negative IDs
    Returns: float
    """
    return 2 * idtp / (2 * idtp + idfp + idfn + 1e-6)

def compute_hota(deta, assa):
    """
    Higher-Order Tracking Accuracy (HOTA)
    Args:
        deta: float, detection accuracy
        assa: float, association accuracy
    Returns: float
    """
    return np.sqrt(deta * assa)

def compute_assa(ass_tp, ass_fp, ass_fn):
    """
    Association Accuracy (AssA)
    Args:
        ass_tp: int, association true positives
        ass_fp: int, association false positives
        ass_fn: int, association false negatives
    Returns: float
    """
    return ass_tp / (ass_tp + 0.5 * (ass_fp + ass_fn) + 1e-6)

def compute_isr(id_switches, num_occlusion_events):
    """
    Identity Switch Rate under Occlusion (ISR)
    Args:
        id_switches: int, number of identity switches during occlusion
        num_occlusion_events: int, number of occlusion events
    Returns: float
    """
    return 100.0 * id_switches / (num_occlusion_events + 1e-6) 