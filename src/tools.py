def _tn(yt, yp):
    """
    False negative for yp: Y predicted, yt: Y true
    """
    return sum([1 if i == 0 and j == 0 else 0 for i, j in zip(yt, yp)])


def _fn(yt, yp):
    """
    False negative for yp: Y predicted, yt: Y true
    """
    return sum([1 if i == 1 and j == 0 else 0 for i, j in zip(yt, yp)])


def _fp(yt, yp):
    """
    False positive for yp: Y predicted, yt: Y true
    """
    return sum([1 if i == 0 and j == 1 else 0 for i, j in zip(yt, yp)])


def _tp(yt, yp):
    """
    True positive for yp: Y predicted, yt: Y true
    """
    return sum([1 if i == 1 and j == 1 else 0 for i, j in zip(yt, yp)])

def accuracy(yt, yp):
    """
    returns an accuracy of your choice
    """
    tp = _tp(yt, yp)
    tn = _tn(yt, yp)
    fp = _fp(yt, yp)
    fn = _fn(yt, yp)
    delim = tp + tn + fp + fn
    if delim == 0:
        return 0.
    return (tp + tn) / delim

def precision(yt, yp):
    """
    returns precision value
    """
    tp = _tp(yt, yp)
    fp = _fp(yt, yp)
    delim = tp + fp
    if delim == 0:
        return 0.
    return tp / delim