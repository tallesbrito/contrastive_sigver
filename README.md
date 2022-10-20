# Contrastive Learning of Handwritten Signature Feature Representations

This repository contains code to train CNNs for feature extraction of Offline Handwritten Handwritten Signatures using contrastive learning methods. Besides, the repository contains code to train writer-dependent and writer-independent classifiers.

This project is a fork from https://github.com/luizgh/sigver, which is the implementation of the SigNet approach described in [1]. Writer-Dependent (WD) classifiers are verified following the method presented in [1]. Writer-Independent (WI) classifers are verified following the method described in [2].

[1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks" http://dx.doi.org/10.1016/j.patcog.2017.05.012

[2] V. L. F. Souza, A. L. I. Oliveira, R. Sabourin, A writer-independent approach for offline signature verification using deep convolutional neural networks features, in: 2018 7th Brazilian Conference on
Intelligent Systems (BRACIS), 2018, pp. 212–217. https://ieeexplore.ieee.org/document/8575615

# Installation

This package requires python 3. Installation can be done with pip:

```bash
pip install git+https://github.com/tallesbrito/contrastive_sigver.git
```

# Usage

## Data preprocessing

The functions in this package expect training data to be provided in a single .npz file, with the following components:

* ```x```: Signature images (numpy array of size N x 1 x H x W)
* ```y```: The user that produced the signature (numpy array of size N )
* ```yforg```: Whether the signature is a forgery (1) or genuine (0) (numpy array of size N )

It is provided functions to process some commonly used datasets in the script ```sigver.datasets.process_dataset```. 

As an example, the following code pre-process the MCYT dataset with the procedure from [1] (remove background, center in canvas and resize to 170x242)

```bash
python -m sigver.preprocessing.process_dataset --dataset mcyt \
 --path MCYT-ORIGINAL/MCYToffline75original --save-path mcyt_170_242.npz
```

During training a random crop of size 150x220 is taken for each iteration. During test we use the center 150x220 crop.

## Training a CNN for Contrastive Feature Learning

In this repository are implemented two contrastive loss functions: the Triplet loss with semi-hard negative mining and the NT-Xent loss.

The flag ```--users``` is used to define the users that are used for feature learning. For instance, (```--users 350 881```). 

Training a Triplet loss optimized model with margin hyper-parameter defined as 0.1:

```
python -m sigver.featurelearning.contrastive_train --dataset-path  <data.npz> \
    --users [first last]\  --epochs 60 --logdir triplet_model \
    --loss-type triplet --margin 0.1
```

Training a NT-Xent loss optimized model with temperture hyper-parameter defined as 0.01:

```
python -m sigver.featurelearning.contrastive_train --dataset-path \
    <data.npz> --users [first last]\  --epochs 60 --logdir ntxent_model \
    --loss-type ntxent --temperature 0.01   
```

For checking all command-line options, use ```python -m sigver.featurelearning.contrastive_train --help```. 

## Training WD classifiers

For training and testing the WD classifiers, use the ```sigver.wd.test``` script. Example:

```bash
python -m sigver.wd.test --model-path <path/to/trained_model> \
    --data-path <path/to/data> --save-path <path/to/save> \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12
```
The parameter ```--gen-for-train``` defines the number of reference signatures for each user of the exploitation set in WD approach.

The example above train WD classifiers for the exploitation set (users 0-300) using a development
set (users 300-881), with 12 genuine signatures per user (this is the setup from [1] - refer to 
the paper for more details). 

For knowing all command-line options, run ```python -m sigver.wd.test```.

## Training WI classifiers

For training and testing the WI classifiers, use the ```sigver.wi.test``` script. Example:

```bash
python -m sigver.wi.test --model-path <path/to/trained_model> \
    --data-path <path/to/data> --save-path <path/to/save> \
    --exp-users 0 300 --dev-users 300 881 --gen-for-ref 12
```

The parameter ```--gen-for-ref``` defines the number of reference signatures for each user of the exploitation set in WI approach.

The example above train a WI classifier for the exploitation set (users 0-300) using a development
set (users 300-881). The WI classifier is tested with 12 reference signatures per user (this is the setup from [2] - refer to the paper for more details). 

For knowing all command-line options, run ```python -m sigver.wi.test```.

# Evaluating the results

When training WD or WI classifiers, the trained_model is a .pth file (a model trained with the script above, or pre-trained - see the section below). These scripts will split the dataset into train/test, train WD/WI classifiers and evaluate then on the test set. This is performed for K random splits (default 10). The script saves a pickle file containing a list, where each element is the result  of one random split. Each item contains a dictionary with:

* 'all_metrics': a dictionary containing:
  * 'FRR': false rejection rate
  * 'FAR_random': false acceptance rate for random forgeries
  * 'FAR_skilled': false acceptance rate for skilled forgeries
  * 'mean_AUC': mean Area Under the Curve (average of AUC for each user)
  * 'EER': Equal Error Rate using a global threshold
  * 'EER_userthresholds': Equal Error Rate using user-specific thresholds
  * 'auc_list': the list of AUCs (one per user)
  * 'global_threshold': the optimum global threshold (used in EER)
* 'predictions': a dictionary containing the predictions for all images on the test set:
  * 'genuinePreds': Predictions to genuine signatures
  * 'randomPreds': Predictions to random forgeries
  * 'skilledPreds': Predictions to skilled forgeries

# Pre-trained models

Pre-trained models can be found here: 
* Triplet loss optimized model with margin ```m=0.1```  ([link](https://github.com/tallesbrito/contrastive_sigver/blob/master/models/triplet_01/model.pth))
* NT-Xent loss optimized model with temperature ```t=0.01``` ([link](https://github.com/tallesbrito/contrastive_sigver/blob/master/models/ntxent_001/model.pth))

See ```example.py``` for a complete example of how to use the models to obtain features from a signature image. 

# License

The source code is released under the BSD 3-clause license. Note that the trained models used the GPDS dataset for training, which is restricted for non-commercial use.  

Please do not contact me requesting access to any particular dataset. These requests should be addressed directly to the groups that collected them.
