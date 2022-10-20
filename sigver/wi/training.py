import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, Dict
import sklearn
import sklearn.svm
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing

import sigver.performance.metrics as metrics
import sigver.wi.data as data


def train_wiclassifier(training_set: Tuple[np.ndarray, np.ndarray],
                            svmType: str,
                            C: float,
                            gamma: Optional[float]) -> sklearn.svm.SVC:
    """ Trains an SVM classifier for a user

    Parameters
    ----------
    training_set: Tuple (x, y)
        The training set (features and labels). y should have labels -1 and 1
    svmType: string ('linear' or 'rbf')
        The SVM type
    C: float
        Regularization for the SVM optimization
    gamma: float
        Hyperparameter for the RBF kernel

    Returns
    -------
    sklearn.svm.SVC:
        The learned classifier

    """

    assert svmType in ['linear', 'rbf']

    train_x = training_set[0]
    train_y = training_set[1]

    # Train the model
    if svmType == 'rbf':
        model = sklearn.svm.SVC(C=C, gamma=gamma)
    else:
        model = sklearn.svm.SVC(kernel='linear', C=C)

    model_with_scaler = pipeline.Pipeline([('scaler', preprocessing.StandardScaler(with_mean=False)),
                                           ('classifier', model)])

    model_with_scaler.fit(train_x, train_y)

    return model_with_scaler


def test_user(model: sklearn.svm.SVC,
              genuine_signatures: np.ndarray,
              random_forgeries: np.ndarray,
              skilled_forgeries: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Test the WI classifier of an user

    Parameters
    ----------
    model: sklearn.svm.SVC
        The learned classifier
    genuine_signatures: np.ndarray
        Genuine signatures for test
    random_forgeries: np.ndarray
        Random forgeries for test (signatures from other users)
    skilled_forgeries: np.ndarray
        Skilled forgeries for test

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        The predictions(scores) for genuine signatures,
        random forgeries and skilled forgeries

    """
    # Get predictions
    genuinePred = model.decision_function(genuine_signatures)
    randomPred = model.decision_function(random_forgeries)
    skilledPred = model.decision_function(skilled_forgeries)

    return genuinePred, randomPred, skilledPred


def train_all_users(dev_train: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    svm_type: str,
                    C: float,
                    gamma: float,
                    num_gen_train: int,
                    rng: np.random.RandomState) -> Dict[int, sklearn.svm.SVC]:
    """ Train classifiers for all users in the exploitation set

    Parameters
    ----------
    dev_train: tuple of np.ndarray (x, y, yforg)
        The development subset used for training
    svm_type: string ('linear' or 'rbf')
        The SVM type
    C: float
        Regularization for the SVM optimization
    gamma: float
        Hyperparameter for the RBF kernel
    num_gen_train: int
        Number of genuines from each user used in training
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    Dict int -> sklearn.svm.SVC
        A dictionary of trained classifiers, where the keys are the users.

    """
    
    training_set = data.create_training_set(dev_train, num_gen_train, rng)

    classifier = train_wiclassifier(training_set, svm_type, C, gamma)

    return classifier


def test_all_users(model: sklearn.svm.SVC,
                   exp_ref: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   exp_test: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   num_gen_test: int,
                   fusion: str,
                   global_threshold: float,
                   rng: np.random.RandomState) -> Dict:
    """ Test a WI classifier for all users and return the metrics

    Parameters
    ----------
    model: sklearn.svm.SVC
        The trained WI classifier for all users
    exp_ref: tuple of np.ndarray (x, y, yforg)
        The reference set from the exploitation set    
    exp_test: tuple of np.ndarray (x, y, yforg)
        The testing set split from the exploitation set
    num_gen_test: int
        Number of signatures per type (genuine, random and skilled) that should be tested
    global_threshold: float
        The threshold used to compute false acceptance and
        false rejection rates

    Returns
    -------
    dict
        A dictionary containing a variety of metrics, including
        false acceptance and rejection rates, equal error rates

    """

    xfeatures_test, y_test, yforg_test = exp_test
    xfeatures_ref, y_ref, yforg_ref = exp_ref

    genuinePreds = []
    randomPreds = []
    skilledPreds = []

    users = np.unique(y_test)
    
    for user in tqdm(users):
        
        skilled_forgeries_idx = np.flatnonzero((y_test == user) & (yforg_test == 1))
        test_genuine_idx = np.flatnonzero((y_test == user) & (yforg_test == 0))
        random_forgeries_idx = np.flatnonzero((y_test != user) & (yforg_test == 0))

        skilled_forgeries_chosen_idx = rng.choice(skilled_forgeries_idx, num_gen_test, replace=False)
        test_genuine_chosen_idx = rng.choice(test_genuine_idx, num_gen_test, replace=False)
        
        user_candidates = np.delete(users, np.where(users == user))
        chosen_users = rng.choice(user_candidates,num_gen_test,replace=False)
        random_forgeries_chosen_idx = []
        for chosen in chosen_users:
            user_idx = np.flatnonzero((y_test == chosen) & (yforg_test==0))
            chosen_user_idx = rng.choice(user_idx)
            random_forgeries_chosen_idx.append(chosen_user_idx)
        random_forgeries_chosen_idx = np.array(random_forgeries_chosen_idx)

        skilled_forgeries = xfeatures_test[skilled_forgeries_chosen_idx]
        test_genuine = xfeatures_test[test_genuine_chosen_idx]
        random_forgeries = xfeatures_test[random_forgeries_chosen_idx]

        references_idx = np.flatnonzero((y_ref == user) & (yforg_ref == 0))
        references = xfeatures_ref[references_idx]

        genuinePredUser = test_signatures(model, references, test_genuine, fusion)
        skilledPredUser = test_signatures(model, references, skilled_forgeries, fusion)
        randomPredUser = test_signatures(model, references, random_forgeries, fusion)

        genuinePreds.append(genuinePredUser)
        skilledPreds.append(skilledPredUser)
        randomPreds.append(randomPredUser)

    # Calculate all metrics (EER, FAR, FRR and AUC)
    all_metrics = metrics.compute_metrics(genuinePreds, randomPreds, skilledPreds, global_threshold)

    results = {'all_metrics': all_metrics,
               'predictions': {'genuinePreds': genuinePreds,
                               'randomPreds': randomPreds,
                               'skilledPreds': skilledPreds}}

    print(all_metrics['EER'], all_metrics['EER_userthresholds'])
    return results


def test_signatures(model: sklearn.svm.SVC,
              ref_signatures: np.ndarray,
              test_signatures: np.ndarray,
              fusion: str) -> np.ndarray:
    """ Test the WD classifier of an user

    Parameters
    ----------
    model: sklearn.svm.SVC
        The learned classifier
    genuine_signatures: np.ndarray
        Genuine signatures for test
    random_forgeries: np.ndarray
        Random forgeries for test (signatures from other users)
    skilled_forgeries: np.ndarray
        Skilled forgeries for test

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        The predictions(scores) for genuine signatures,
        random forgeries and skilled forgeries

    """

    all_predictions = np.empty(test_signatures.shape[0])
    all_predictions[:] = np.NaN

    for qsign in range(test_signatures.shape[0]):

        diff = np.absolute(test_signatures[qsign] - ref_signatures)  

        preds = model.decision_function(diff)
        
        if(fusion=='max'):
            decision = np.max(preds)
        elif(fusion=='min'):
            decision = np.min(preds)
        elif(fusion=='mean'):
            decision = np.mean(preds)
        elif(fusion=='median'):
            decision = np.median(preds)

        all_predictions[qsign] = decision

    return all_predictions


def train_test_all_users(exp_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         dev_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         svm_type: str,
                         C: float,
                         gamma: float,
                         num_gen_train: int,
                         num_gen_ref: int,
                         num_gen_test: int,
                         fusion: str,
                         global_threshold: float = 0,
                         rng: np.random.RandomState = np.random.RandomState()) \
        -> Tuple[Dict[int, sklearn.svm.SVC], Dict]:
    """ Train and test classifiers for every user in the exploitation set,
        and returns the metrics.

    Parameters
    ----------
    exp_set: tuple of np.ndarray (x, y, yforg)
        The exploitation set
    dev_set: tuple of np.ndarray (x, y, yforg)
        The development set
    svm_type: string ('linear' or 'rbf')
        The SVM type
    C: float
        Regularization for the SVM optimization
    gamma: float
        Hyperparameter for the RBF kernel
    num_gen_train: int
        Number of genuine signatures available for training
    num_forg_from_dev: int
        Number of forgeries from each user in the development set to
        consider as negative samples
    num_forg_from_exp: int
        Number of forgeries from each user in the exploitation set (other
        than the current user) to consider as negative sample.
    num_gen_ref: int
        Number of genuine signatures used as reference signatures within fusion function    
    num_gen_test: int
        Number of genuine signatures for testing
    fusion: str
        The applied fusion function when veryfing signatures
    global_threshold: float
        The threshold used to compute false acceptance and
        false rejection rates
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    dict sklearn.svm.SVC
        The classifier for all users

    dict
        A dictionary containing a variety of metrics, including
        false acceptance and rejection rates, equal error rates

    """
    if global_threshold:
        print('Global threshold is set to:', global_threshold)

    exp_ref, exp_test = data.split_ref_test(exp_set, num_gen_ref, num_gen_test, rng)


    dev_train = data.split_train(dev_set, num_gen_train, rng)

    print('Training Writer-Independent (WI) classifier...')
    classifier = train_all_users(dev_train, svm_type, C, gamma, num_gen_train, rng)

    print('Tests have been performed:')
    results = test_all_users(classifier, exp_ref, exp_test, num_gen_test, fusion, global_threshold, rng)

    return classifier, results
