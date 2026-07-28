import re

import numpy as np
import tqdm


def z_score(data, mean=None, std=None):
    """
    Calculate the standardized value of the data, that is, subtract the mean from the data 
    and divide it by the standard deviation to ensure that the data follows a standard normal distribution in a statistical sense.

    NaN-safe: some years in xxltrafficdata (e.g. pems03 2006/2007/2009/...) contain NaN
    entries in the raw sensor data. Using np.mean / np.std on such arrays poisons the
    whole split with NaN. We fall back to nanmean / nanstd, guard against zero std, and
    impute remaining NaNs with 0 (i.e. sensor-global mean in z-score space).

    ``mean`` and ``std`` may be supplied to reuse training-split statistics for
    validation/test data.  This is the leakage-free default in
    :func:`generate_samples`.
    """
    mean = np.nanmean(data) if mean is None else float(mean)
    std = np.nanstd(data) if std is None else float(std)
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    out = (data - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def generate_dataset(data, idx, x_len=12, y_len=12):
    """"
    Generates a dataset of input x and output y from the input data at the given index idx
    """
    res = data[idx]  # Get data by index
    node_size = data.shape[1]  # Get the number of nodes
    t = len(idx)-1
    idic = 0
    x_index, y_index = [], []  # Initialize the x and y index lists
    
    # Traverse the index to generate the index of x and y
    for i in tqdm.tqdm(range(t, 0, -1)):
        if i-x_len-y_len>=0:
            x_index.extend(list(range(i-x_len-y_len, i-y_len)))
            y_index.extend(list(range(i-y_len, i)))

    x_index = np.asarray(x_index)  # Convert to numpy array
    y_index = np.asarray(y_index)
    x = res[x_index].reshape((-1, x_len, node_size))  # Reshape the data
    y = res[y_index].reshape((-1, y_len, node_size))
    
    return x, y


def _is_pems_path(path):
    """Whether a path contains a dataset component such as ``pems05``."""
    parts = re.split(r"[\\/]+", str(path))
    return any(
        re.fullmatch(r"pems\d+", part, flags=re.IGNORECASE)
        for part in parts
    )


def generate_samples(days, savepath, data, graph, train_rate=0.6, val_rate=0.2, test_rate=0.2, val_test_mix=False):
    """
    Generate training, validation and test datasets and save them as .npz files
    """
    edge_index = np.array(list(graph.edges)).T  # Get the edge index of the graph and transpose it
    del graph
    
    if _is_pems_path(savepath):
        # The published PEMS benchmark uses the first ``days`` days of each
        # yearly matrix at five-minute resolution. Detect the dataset by path
        # component so relative and absolute download paths behave identically.
        data = data[0:days*288, :]
    
    t, n = data.shape[0], data.shape[1]  # Get the time step and number of nodes of the data
    
    # Split the training, validation, and test set indices according to the ratio
    train_idx = [i for i in range(int(t*train_rate))]
    val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
    test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]
    
    train_x, train_y = generate_dataset(data, train_idx)
    val_x, val_y = generate_dataset(data, val_idx)
    test_x, test_y = generate_dataset(data, test_idx)
    
    # If you need to mix validation and test sets
    if val_test_mix:
        val_test_x = np.concatenate((val_x, test_x), 0)  # Combine validation and test sets x data
        val_test_y = np.concatenate((val_y, test_y), 0)  # Combine validation and test set y data
        val_test_idx = np.arange(val_x.shape[0]+test_x.shape[0])  # Generate Index
        np.random.shuffle(val_test_idx)  # Shuffle the index order
        val_x, val_y = val_test_x[val_test_idx[:int(t*val_rate)]], val_test_y[val_test_idx[:int(t*val_rate)]]  # Re-partition validation and test set data
        test_x, test_y = val_test_x[val_test_idx[int(t*val_rate):]], val_test_y[val_test_idx[int(t*val_rate):]]

    # Fit normalization on the training split only.  Re-estimating statistics
    # independently on validation/test exposes aggregate information from the
    # full future split.
    train_mean = np.nanmean(train_x)
    train_std = np.nanstd(train_x)
    if not np.isfinite(train_std) or train_std < 1e-8:
        train_std = 1.0
    train_x = z_score(train_x, train_mean, train_std)
    val_x = z_score(val_x, train_mean, train_std)
    test_x = z_score(test_x, train_mean, train_std)

    # Targets (y) are kept in the raw scale. If the raw sensor series contains NaN
    # (happens in xxltrafficdata for years with missing detectors), MSE loss on NaN
    # targets propagates NaN into the gradient. Impute y-NaNs with 0 so training is
    # numerically stable; a follow-up fix on the upstream RawData (interpolation /
    # forward-fill) is strongly recommended.
    train_y = np.nan_to_num(train_y, nan=0.0, posinf=0.0, neginf=0.0)
    val_y = np.nan_to_num(val_y, nan=0.0, posinf=0.0, neginf=0.0)
    test_y = np.nan_to_num(test_y, nan=0.0, posinf=0.0, neginf=0.0)

    # Save data to file
    np.savez(
        savepath,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        edge_index=edge_index,
        input_mean=np.asarray(train_mean, dtype=np.float64),
        input_std=np.asarray(train_std, dtype=np.float64),
    )
    
    # Build the returned data dictionary
    data = {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "edge_index": edge_index,
        "input_mean": np.asarray(train_mean, dtype=np.float64),
        "input_std": np.asarray(train_std, dtype=np.float64),
    }
    return data
