import numpy as np
import scipy.io as si

def _extract(prefix, md, max_dig):
    ret = []
    for dig in range(max_dig):
        samples = md[prefix + str(dig)]
        labels = np.empty([samples.shape[0], 1], dtype=np.float32)
        labels.fill(dig)
        ret.append(np.hstack((samples.astype(np.float32) / 256, labels)))
    return ret

def _split_sample_and_label(merged_mb):
    [s, l] = np.hsplit(merged_mb, [merged_mb.shape[1] - 1])
    # change label to sparse representation
    n = merged_mb.shape[0]
    ll = np.zeros([n, 10], dtype=np.float32)
    ll[np.arange(n), l.astype(int).flat] = 1
    return (s, ll)

def load_mb_from_mat(mat_file, mb_size):
    # load from mat
    md = si.loadmat(mat_file)
    # merge all data
    train_all = np.concatenate(_extract('train', md, 10))
    test_all = np.concatenate(_extract('test', md, 10))
    # shuffle
    np.random.shuffle(train_all)
    # make minibatch
    train_mb = np.vsplit(train_all, range(mb_size, train_all.shape[0], mb_size))
    train_data = map(_split_sample_and_label, train_mb)
    test_data = _split_sample_and_label(test_all)
    print 'Training data: %d mini-batches' % len(train_mb)
    print 'Test data: %d samples' % test_all.shape[0]
    print train_data[0][1].shape
    return (train_data, test_data)

