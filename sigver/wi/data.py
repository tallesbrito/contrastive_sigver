import numpy as np
from typing import Tuple


def split_ref_test(exp_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     num_gen_ref: int,
                     num_gen_test: int,
                     rng: np.random.RandomState) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                          Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """ Splits a set into reference set and testing set. Both sets contains the same users. The
        reference set contains only genuine signatures, while the testing set contains
        genuine signatures and forgeries. Note that the number of genuine signatures used
        as references plus the number of genuine signatures for test must be smaller or equal to
        the total number of genuine signatures (to ensure no overlap)

    Parameters
    ----------
    exp_set: tuple of np.ndarray (x, y, yforg)
        The dataset
    num_gen_ref: int
        The number of genuine signatures to be used as references within fusion function
    num_gen_test: int
        The number of genuine signatures to be used for testing
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    tuple of np.ndarray (x, y, yforg)
        The training set

    tuple of np.ndarray (x, y, yforg)
        The testing set
    """
    x, y, yforg = exp_set
    users = np.unique(y)

    ref_idx = []
    test_idx = []

    for user in users:
        user_genuines = np.flatnonzero((y == user) & (yforg == False))
        rng.shuffle(user_genuines)
        user_ref_idx = user_genuines[0:num_gen_ref]
        user_test_idx = user_genuines[-num_gen_test:]

        # Sanity check to ensure training samples are not used in test:
        assert len(set(user_ref_idx).intersection(user_test_idx)) == 0

        ref_idx += user_ref_idx.tolist()
        test_idx += user_test_idx.tolist()

        user_forgeries = np.flatnonzero((y == user) & (yforg == True))
        test_idx += user_forgeries.tolist()

    exp_ref = x[ref_idx], y[ref_idx], yforg[ref_idx]
    exp_test = x[test_idx], y[test_idx], yforg[test_idx]

    return exp_ref, exp_test


def split_train(dev_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     num_gen_train: int,
                     rng: np.random.RandomState) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                          Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """ Splits a set into a training set. The training set contains only genuine signatures.

    Parameters
    ----------
    dev_set: tuple of np.ndarray (x, y, yforg)
        The dataset
    num_gen_train: int
        The number of genuine signatures to be used for training
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    tuple of np.ndarray (x, y, yforg)
        The training set
    """
    x, y, yforg = dev_set
    users = np.unique(y)

    train_idx = []

    for user in users:
        user_genuines = np.flatnonzero((y == user) & (yforg == False))
        rng.shuffle(user_genuines)
        user_train_idx = user_genuines[0:num_gen_train]

        train_idx += user_train_idx.tolist()

    dev_train = x[train_idx], y[train_idx], yforg[train_idx]

    return dev_train


def set_to_genuine_array(input_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                        num_gen: int) -> np.ndarray:
    """ Converts a tuple (x, y, yforg) into an array (number of users, number of signatures, number of features),
    considering only genuine signatures.

    Parameters
    ----------
    input_set: tuple of np.ndarray (x, y, yforg)
        The dataset
    num_gen_train: int
        The number of genuine signatures per user

    Returns
    -------
    np.ndarray of shape (number of users, number of signatures, number of features)
        The converted array of genuine signatures
    """
    x, y, yforg = input_set
    users = np.unique(y)

    arr = np.zeros((len(users),num_gen,x.shape[1]))

    train_idx = []

    for i,user in enumerate(users):
        user_genuines = np.flatnonzero((y == user) & (yforg == False))
        user_idx = user_genuines[0:num_gen]
        arr[i] = x[user_idx]

    return arr


def create_training_set(dev_train: Tuple[np.ndarray, np.ndarray, np.ndarray],
                        num_gen_train: int,
                        rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """ Creates a training set for training a WI classifier for a user

    Parameters
    ----------
    dev_train: tuple of np.ndarray (x, y, yforg)
        The training set split of the development dataset
    num_gen_train: int
        Number of genuine signatures of each user used for training
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    np.ndarray (N), np.ndarray (N)
        The dataset for the user (x, y), where N is the number of signatures
        (genuine + random forgeries)
    """

    S = set_to_genuine_array(dev_train, num_gen_train)

    u_within = pairwise_within_disimilarity(S)
    u_within_y = np.ones((u_within.shape[0]))

    G = S[:,:num_gen_train-1,:]
    u_between = pairwise_between_dissimilarity(G,num_gen_train//2,rng)
    u_between_y = np.full((u_between.shape[0]),-1)

    LX = np.concatenate((u_within,u_between),axis=0) 
    Ly = np.concatenate((u_within_y,u_between_y),axis=0)
    
    return LX, Ly


def pairwise_between_dissimilarity(arr, n_against, rng):
    n_users = arr.shape[0]
    n_signs = arr.shape[1]
    dsim_len = int(n_users * n_signs * n_against)
    dsim = np.zeros((dsim_len,arr.shape[2]))
    
    pos = 0
    for i in range(n_users):
        user_all_signs = arr[i]
        candidates = [k for k in range(n_users) if k != i]
        selected = rng.choice(candidates,n_against,replace=False)
        for j in selected:
            diff = np.absolute(user_all_signs - arr[j][rng.randint(n_signs)])
            dsim[pos: pos + diff.shape[0]] = diff 
            pos += diff.shape[0]
            
    return dsim


def pairwise_within_disimilarity(arr):
    n_users = arr.shape[0]
    n_signs = arr.shape[1]
    dsim_len = int(n_users*(((n_signs)*(n_signs -1))/2))
    dsim = np.zeros((dsim_len,arr.shape[2]))
    
    pos = 0
    for i in range(n_users):
        user_all_signs = arr[i]
        for j in range(n_signs):
            diff = np.absolute(user_all_signs - user_all_signs[j])
            diff_below_j = diff[j+1:n_signs]
            dsim[pos: pos + diff_below_j.shape[0]] = diff_below_j
            pos += diff_below_j.shape[0]
            
    return dsim


def get_random_forgeries_from_dev(dev_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  num_forg_from_dev: int,
                                  rng: np.random.RandomState):
    """ Obtain a set of random forgeries form a development set (to be used
        as negative samples)

    Parameters
    ----------
    dev_set: tuple of np.ndarray (x, y, yforg)
        The development dataset
    num_forg_from_dev: int
        The number of random forgeries (signatures) from each user in the development
        set to be considered
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    np.ndarray (N x M)
        The N negative samples (M is the dimensionality of the feature set)

    """
    x, y, yforg = dev_set
    users = np.unique(y)

    random_forgeries = []
    for user in users:
        idx = np.flatnonzero((y == user) & (yforg == False))
        chosen_idx = rng.choice(idx, num_forg_from_dev, replace=False)
        random_forgeries.append(x[chosen_idx])

    return np.concatenate(random_forgeries)

