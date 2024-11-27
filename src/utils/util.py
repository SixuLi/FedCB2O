import numpy as np

def chunk(a, i, n):
    a2 = chunkify(a, n)
    return a2[i]

def chunkify(a, n):
    # splits list into even size list of lists
    # [1,2,3,4] -> [1,2], [3,4]

    k, m = divmod(len(a), n)
    gen = (a[i * k + min(i, m):(i+1) * k + min(i+1, m)] for i in range(n))
    return list(gen)

def split_list_uneven(lst, uneven_lengths):
    result = []
    start = 0

    for length in uneven_lengths:
        end = start + length
        result.append(np.array(lst[start:end]))
        start = end

    return result

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count