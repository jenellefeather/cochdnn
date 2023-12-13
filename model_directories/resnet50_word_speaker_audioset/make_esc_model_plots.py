import pickle as pckl
import numpy as np
import sys
import h5py
import torch as ch
import os
import scipy
import matplotlib.pylab as plt
sys.path.append('/om/user/jfeather/python-packages/tfmatching/')
import synthhelpers
import pandas
from sklearn.svm import SVC
from sklearn import preprocessing
import time
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import time
import subprocess
from joblib import Parallel, delayed
import multiprocessing
import functools
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn import svm
from functools import partial

import matplotlib
matplotlib.rcParams.update({'font.size': 26})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# save the dictionary. 
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pckl.dump(obj, f, pckl.HIGHEST_PROTOCOL)
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pckl.load(f)


def preproc_sound_np(sound):
    """
    Normalizes and preprocesses sound; returns a tensor.
    """
    sound = sound - np.mean(sound)
    sound_rms = np.sqrt(np.mean(sound**2))
    if sound_rms == 0:
        sound_rms = 1
    sound = sound/sound_rms*0.1
    sound = np.expand_dims(sound, 0)
    sound = ch.from_numpy(sound).float().cuda()
    return sound

def get_train_and_test(left_out_fold, data_path, fold_info_path, num_reps, SEED):
    """
    Gets training and testing data and labels given path to ESC-50 dataset.
    Inputs
    ------
    left_out_fold : integer
        number of the left out fold to be the test data
    data_path : string
        path where all the audio files are located
    fold_info_path : string
        path where csv containing fold information is located
    
    Returns
    -------
    train_data
        list of preprocessed sounds for training data
    train_labels
        list of labels for training data
    test_data
        list of preprocessed sounds for testing data
    test_labels
        list of labels for testing data
    sound_identifiers
        ordered list of unique sound identifiers (e.g. 1-100032-A-0.wav)
        
    """
    print("Has begun get_train_and_test.")

    np.random.seed(SEED)

    train_data_paths = []
    train_labels_paths = []
    test_data_paths = []
    test_labels_paths = []
    sound_identifiers = {'train':[], 'test':[]}

    df = pandas.read_csv(fold_info_path)
    for index, row in df.iterrows():
        if row['fold']==left_out_fold:
            test_data_paths.append(data_path+row['filename'])
            test_labels_paths.append(row['target'])
            sound_identifiers['test'].append(row['filename'])
        else:
            train_data_paths.append(data_path+row['filename'])
            train_labels_paths.append(row['target'])
            sound_identifiers['train'].append(row['filename'])

    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    print("Has finished get_train_and_test.")
    return train_data_paths, train_labels_paths, test_data_paths, test_labels_paths, sound_identifiers

def train_svm(train_features, train_labels, test_features, test_labels, c_values, 
              average_test_predictions, n_splits_cv=3):
    """
    Trains an SVM on training data and returns predictions on testing data.
    Inputs
    ------
    train_features : list
        [n_samples x n_features] or [n_samples x n_reps x n_features]
    train_labels : list
        [n_samples,]
    test_features : list
        [n_samples x n_features] or [n_samples x n_reps x n_features]
    test_labels : list
        [n_samples,]
    average_test_predictions : bool
        If true (default) averages the predictions across the n_rep for 
        each sound. If false, each n_rep is predicted independently.  
    n_splits_cv : int
        number of unique splits to use for the cross validation search
    
    Returns
    -------
    predictions
        SVM predictions on testing data
        
    """

    def average_predictions_scorer(estimator, X, y):
        """
        Scorer that averages of n_rep predictions if the features are in 
        shape [samples, n_rep, features]. If samples are just [samples, features] 
        then returns the prediction for each one.
        """
        if len(X.shape)>2:
            orig_shape = X.shape
            X = X.reshape(orig_shape[0] * orig_shape[1], orig_shape[2])
        else:
            orig_shape = None
        y_prob = estimator.decision_function(X)
        if orig_shape is not None:
            y_prob = y_prob.reshape(orig_shape[0], orig_shape[1], -1)
            y_prob = np.mean(y_prob, axis=1) # Collapse across the samples
        y_compressed = np.argmax(y, axis=1)
        pred_label = np.argmax(y_prob, axis=1)
        acc_mean = metrics.accuracy_score(y_compressed, pred_label)
        return acc_mean

    def average_predictions_scorer_with_n_rep(n_rep, estimator, X, y):
        """
        Scorer that takes in n_rep as an argument, takes in the predictions, and 
        averages over the n_rep examples of each clip
        """
        y_prob = estimator.decision_function(X)
        # NOTE: this reshaping assumes that the indexes stay in order for the test data! 
        y_prob = y_prob.reshape(int(y_prob.shape[0]/n_rep), n_rep, -1)
        y_prob = np.mean(y_prob, axis=1) # Collapse across the samples
        y_compressed = np.argmax(y, axis=1)
        # Just take one of the repetition (they are all the same)
        y_compressed = y_compressed.reshape(y_prob.shape[0], n_rep)
        assert(np.sum(y_compressed[:,0]**2 - y_compressed[:,1]**2) == 0)
        y_compressed = y_compressed[:,0]
        pred_label = np.argmax(y_prob, axis=1)
        acc_mean = metrics.accuracy_score(y_compressed, pred_label)
        return acc_mean

    # If we have multiple repetitions, leave out full sounds for the cross validation
    train_features_shape = train_features.shape
    test_features_shape = test_features.shape
    if len(train_features.shape)>2:
        # An iterable yielding (train, test) splits as arrays of indices.
        train_features_shape = train_features.shape
        n_rep = train_features_shape[1]
        train_features = train_features.reshape(
            train_features_shape[0]*train_features_shape[1], train_features_shape[2])
        train_labels = np.repeat(train_labels, n_rep)
        cv_sounds = ShuffleSplit(n_splits=n_splits_cv, test_size=0.25, random_state=0)
        # Only make the index across the sound splits
        cv_index = cv_sounds.split(np.arange(train_features_shape[0]))
        new_cv_list = []
        n_rep = train_features_shape[1]
        for v in cv_index:
            new_cv_list.append([np.array([n_rep*k + np.arange(n_rep) for k in v[0]]).ravel(),
                                np.array([n_rep*k + np.arange(n_rep) for k in v[1]]).ravel()])
        print('Number CV splits:', len(new_cv_list))
        # An iterable yielding (train, test) splits as arrays of indices.
        cv = iter(new_cv_list)

        test_features = test_features.reshape(
            test_features_shape[0]*test_features_shape[1], test_features_shape[2])
        test_labels = np.repeat(test_labels, n_rep)

        # Wrap scorer in a partial function to take n_rep as an input
        if average_test_predictions:
            scorer = partial(average_predictions_scorer_with_n_rep, n_rep)
        else:
            scorer = average_predictions_scorer
    else:
        cv = ShuffleSplit(n_splits=n_splits_cv, test_size=0.25, random_state=0)
        scorer = average_predictions_scorer

    lb = preprocessing.LabelBinarizer()
    lb.fit(range(0,50))
    test_labels = lb.transform(test_labels)
    train_labels = lb.transform(train_labels)
    parameters = {"estimator__C": c_values}
    svc = OneVsRestClassifier(svm.LinearSVC(max_iter=1000, dual=True, # dual=False, 
                                            random_state=0), n_jobs=4)
    clf = GridSearchCV(svc, param_grid=parameters, cv=cv,
                       n_jobs=4, scoring=scorer, refit=True)

    tic = time.perf_counter()
    clf.fit(train_features, train_labels)
    toc = time.perf_counter()
    print(f"svc.fit() ran in {toc - tic:0.4f} seconds")
    acc = clf.score(test_features, test_labels)
#     predictions = clf.predict(test_features)
    predictions = lb.transform(np.argmax(clf.decision_function(test_features),1))
    if len(test_features_shape)>2:
        if average_test_predictions:
            predictions = np.mean(predictions.reshape(test_features_shape[0],
                                                      test_features_shape[1],-1), axis=1)
    # Bug here because some of the test things have no non-zero predictions????? 
    predictions = np.argmax(predictions, axis=1)

    return predictions, acc, svc

# downsampling activations on the *CPU*
def downsample_activations(activations, num_activations, layer_random_projections, layer):
    # activations is a np array on cpu
    current_layer_size = list(np.shape(activations))[1]
    print("current layer size: "+str(current_layer_size))
    print("want to downsample to: "+str(num_activations))
    num_splits = 4
    for iteration in range(len(layer_random_projections)):
        layer_random_projection = layer_random_projections[iteration]
        
        # project the data
        projected_values = np.matmul(activations, layer_random_projection)

        if iteration==0:
            activation_mat = projected_values
        else:
            activation_mat = np.hstack((activation_mat, projected_values))
    return activation_mat

def get_features_list(model, net_name, data_paths, data_labels_paths, fold, layer, 
                      num_activations, num_reps, SEED, scratch_activations_dir, 
                      sound_identifiers, overwrite, layer_random_projections=None):
    '''
    Passes each sound in data (in the form of a tensor) into model
    and collect activations from specified layer (features for svm)
    '''
    print("has begun get_features_list")
    np.random.seed(SEED*2)
    data_labels = data_labels_paths # TODO: remove -- this is due to old code 

    example_downsampled_filepath = os.path.join(scratch_activations_dir,
                                                'downsampled_activations',
                                                net_name, layer,
                                                '%d_activations'%num_activations,
                                                'nreps%d_rs%d_identifier%s.pickle'%(
                                                    num_reps, SEED, 
                                                    sound_identifiers[0]))
    # If this example file exists, then assume we can load all of the downsampled 
    # activations rather than measuring them again
    if os.path.isfile(example_downsampled_filepath) and not overwrite:
        print("Loading downsampled activations from files like %s"%example_downsampled_filepath)
        activations = []
        for idx in range(len(sound_identifiers)):
            downsampled_filename = os.path.join(scratch_activations_dir,'downsampled_activations',
                                    net_name, layer, '%d_activations'%num_activations,
                                    'nreps%d_rs%d_identifier%s.pickle'%(
                                        num_reps, SEED, sound_identifiers[idx]))
            f = open(downsampled_filename,"rb")
            single_activations = pckl.load(f)
            f.close()
            activations.append(single_activations)
        activation_mat = np.array(activations)
    else:
        print("Running save_activations()")
        save_activations(data_paths,  model, net_name, layer, num_reps, SEED,
                         scratch_activations_dir, sound_identifiers, overwrite)
        print("Finished save_activations()")
       
        activations = []
        for idx in range(len(sound_identifiers)):
            filename = os.path.join(scratch_activations_dir,'full_activations', net_name, layer,
                                    'nreps%d_rs%d_identifier%s.pickle'%(
                                        num_reps, SEED, sound_identifiers[idx]))
            f = open(filename,"rb")
            a = pckl.load(f)
            f.close()
            if isinstance(a, list):
                a = np.array([sample.ravel() for sample in a])
            else:
                a = a.ravel()
            activations.append(a)
        activations = np.concatenate(activations)
        print("activations.shape "+str(activations.shape))
    
        if num_activations<activations.shape[1]:
            current_layer_size = list(activations.shape)[1]
            # If we don't yet have random projections, then make them here. 
            if layer_random_projections is None:
                layer_random_projections = []
                # TODO: Add flag for the number of activation groupings
                num_splits = 4
                for i in range(num_splits):
                    if i!=num_splits-1:
                        a = np.random.normal(0,1,(current_layer_size, 
                                                  int(num_activations/num_splits)))
                    else:
                        # account for edge cases (if num_activations is not divisble by num_splits)
                        a=np.random.normal(0, 1, (current_layer_size, 
                                                  num_activations-(num_splits-1)*int(num_activations/num_splits)))
                    a = a/(np.sqrt(np.sum(np.square(a),0))) # set l2 norm to 1
                    layer_random_projections.append(a)
    
            activation_mat = downsample_activations(activations, num_activations, 
                                                    layer_random_projections, layer)
            print('activation_mat.shape: ', activation_mat.shape)
            print("done making random projections")
        else:
            activation_mat = activations
            print("do not need to downsample")
    
        # We need to restack the samples if they came into this function as a list
        if num_reps>1:
            activation_mat = [activation_mat[i:i+num_reps,:] for i in range(0, activation_mat.shape[0], num_reps)]
            activation_mat = np.array(activation_mat)
            print('activation_mat.shape: ', activation_mat.shape)
 
        for idx in range(len(sound_identifiers)):
            downsampled_filename = os.path.join(scratch_activations_dir,'downsampled_activations', 
                                    net_name, layer, '%d_activations'%num_activations,
                                    'nreps%d_rs%d_identifier%s.pickle'%(
                                        num_reps, SEED, sound_identifiers[idx]))
            if not os.path.isdir(os.path.join(scratch_activations_dir, "downsampled_activations", 
                                              net_name, layer, '%d_activations'%num_activations)):
                try:
                    os.makedirs(os.path.join(scratch_activations_dir, "downsampled_activations",
                                             net_name, layer, '%d_activations'%num_activations))
                except OSError:
                    pass
            f = open(downsampled_filename,"wb")
            pckl.dump(activation_mat[idx], f)
            f.close()

        print("activation_mat.shape "+str(activation_mat.shape))

    return activation_mat, data_labels, layer_random_projections
   
def save_activations(data_paths, model, net_name, layer, num_reps, SEED,
                     scratch_activations_dir, sound_identifiers, overwrite):
    """
    Measures the activations for the sound and saves it into the scratch directory. 
    If the file already exists and overwrite is false, skips to the next sound.
    """

    if not os.path.isdir(scratch_activations_dir):
        try:
            os.makedirs(scratch_activations_dir)
        except OSError:
            pass
    if not os.path.isdir(os.path.join(scratch_activations_dir, "full_activations")):
        try:
            os.makedirs(os.path.join(scratch_activations_dir, "full_activations"))
        except OSError:
            pass

    for idx, audio_path in enumerate(data_paths):
        filename = os.path.join(scratch_activations_dir,'full_activations', net_name, layer,
                                'nreps%d_rs%d_identifier%s.pickle'%(
                                    num_reps, SEED, sound_identifiers[idx]))
        # If the file exists and we aren't forcing overwrite just move on and use saved activations
        if os.path.isfile(filename) and not overwrite:
            continue

        sound_array = []
        for x in range(num_reps): # Loop through each sound num_reps times, always choose a random 2 second clip
            sound, SR = synthhelpers.load_audio_wav_resample(audio_path, 
                                                             resample_SR=20000, 
                                                             START_SECS='random')
            while sum(sound)==0: # If sound is silent, choose a new clip
                sound, SR = synthhelpers.load_audio_wav_resample(audio_path, 
                                                                 resample_SR=20000, 
                                                                 START_SECS='random')
            sound_array.append(preproc_sound_np(sound)) # normalize

        sound = sound_array

        all_activations = []
        for clip in sound:
            with ch.no_grad():
                (predictions, rep, all_outputs), orig = model(clip, with_latent=True, fake_relu=True)
            if len(layer.split('/'))>=2:
                split_layer = layer.split('/')
                print(split_layer[0])
                print('/'.join(split_layer[1:len(split_layer)]))
                activations = all_outputs[split_layer[0]]['/'.join(split_layer[1:len(split_layer)])]
            else:
                activations = all_outputs[layer]
            # make the activations numpy arrays not tensors
            activations = activations.detach().cpu().numpy()
            all_activations.append(activations)

        if not os.path.isdir(os.path.join(scratch_activations_dir, "full_activations", net_name, layer)):
            try:
                os.makedirs(os.path.join(scratch_activations_dir, "full_activations", net_name, layer))
            except OSError:
                pass
        filename = os.path.join(scratch_activations_dir,'full_activations', net_name, layer, 
                                'nreps%d_rs%d_identifier%s.pickle'%(
                                    num_reps, SEED, sound_identifiers[idx]))
        f = open(filename,"wb")
        pckl.dump(all_activations, f)
        f.close()

def get_category_accuracies(predictions,labels,target):
    """
    Returns the accuracy for target category.
    """
    correct = 0
    total = 0
    for idx in range(len(labels)):
        label = labels[idx]
        if label==target:
            prediction = predictions[idx]
            if label==prediction:
                correct = correct+1
            total = total+1
    return correct/total


def get_predictions_and_make_plots(model, net_name, scratch_activations_dir,
                                   layer='relufc', 
                                   num_activations=4096, 
                                   num_reps=1, 
                                   SEED=1,
                                   average_test_predictions=False, 
                                   c_values=[0.01,0.1,1],
                                   overwrite=False):
    """
    Trains an SVM to get predictions on ESC-50 sounds that are run through a pre-trained model.
    Inputs
    ------
    model :
        the pre-trained model
    net_name : str
        name of the network
    scratch_activations_dir : str
        directory in /om2/scratch/ to save the activations 
    layer : str
        name of layer in model's build_network.py metamer_layers variable
    num_activations : int
        number of activations to downsample to
    num_reps : int
        number of samples from each sound
    SEED : int
        random seed
    average_test_predictions : bool
        whether or not to average test predictions 
    c_values : list of floats
        c_values for sklearn to try during SVM cross-validation
    overwrite : bool
        if true, overwrites saved pickles for the first fold
    """
    folds = [1, 2, 3, 4, 5]
    data_path = '/om4/group/mcdermott/user/michl/ESC-50-master/audio/' # .wav files of the sound clips
    fold_info_path = '/om4/group/mcdermott/user/michl/ESC-50-master/meta/esc50.csv'

    all_predictions = []
    all_labels = []
    avg_accuracies = []
    all_svcs = []
    all_test_labels = []

    for left_out_fold in folds:
        print("Starting left out fold " +str(left_out_fold)+".")
        print("num_reps: "+str(num_reps))
        train_data_paths, train_labels_paths, test_data_paths, test_labels_paths, sound_identifiers = get_train_and_test(
            left_out_fold, data_path, fold_info_path, num_reps, SEED)
        print("getting train features")
        print("len(train_data_paths): "+str(len(train_data_paths)))
        print("len(test_data_paths): "+str(len(test_data_paths)))

        train_features, train_labels, layer_random_projections = get_features_list(model, net_name, train_data_paths, 
                                                         train_labels_paths, left_out_fold, 
                                                         layer, num_activations, num_reps, 
                                                         SEED, scratch_activations_dir, 
                                                         sound_identifiers['train'], overwrite)
        print("getting test features")

        test_features, test_labels, layer_random_projections = get_features_list(model, net_name, test_data_paths, 
                                                       test_labels_paths, left_out_fold, 
                                                       layer, num_activations, num_reps, 
                                                       SEED, scratch_activations_dir, 
                                                       sound_identifiers['test'], overwrite,
                                                       layer_random_projections=layer_random_projections)
        print("finished get_features_list()")

        # normalization
        train_features = np.array(train_features)
        print("train_features.shape: "+str(train_features.shape))

        if len(train_features.shape)>2:
            train_features_shape = train_features.shape
            train_features = train_features.reshape(train_features_shape[0]*train_features_shape[1], 
                                                    train_features_shape[2])
            # train_labels = np.repeat(train_labels, train_features_shape[1])
            # TODO: add flag if we don't want to use the mean and unit variance?
            scaler = preprocessing.StandardScaler().fit(train_features)
            train_features_preproc = scaler.transform(train_features).reshape(train_features_shape)
        else:
            # TODO: add flag if we don't want to use the mean and unit variance?
            scaler = preprocessing.StandardScaler().fit(train_features)
            train_features_preproc = scaler.transform(train_features)

        test_features = np.array(test_features)
        print("test_features.shape: "+str(test_features.shape))
        if len(test_features.shape)>2:
            test_features_shape = test_features.shape
            test_features_preproc = scaler.transform(test_features.reshape(
                    test_features_shape[0]*test_features_shape[1], test_features_shape[2])).reshape(
                    test_features_shape)
        else:
            test_features_preproc = scaler.transform(test_features)

        predictions, acc, svc = train_svm(train_features_preproc, train_labels, 
                                          test_features_preproc, test_labels, 
                                          c_values, average_test_predictions)
        print("finished train_svm()")
        all_predictions.append(predictions)
        all_labels.append(test_labels)
        avg_accuracies.append(acc)
        all_svcs.append(svc)
        all_test_labels.append(test_labels)

        print("Accuracy: "+str(acc))
        print("Done with fold " +str(left_out_fold) +".")

        overwrite=False # If we completed any fold, then reuse the activations
        
    ############################################
    ###### print results and create plots ######
    ############################################
    print('\n\n')
    print("Results:\n")
    
    # model accuracies averaged across all sounds for each of the 5 splits
    for idx in range(len(avg_accuracies)):
        print("Accuracy, fold "+str(idx)+": "+str(avg_accuracies[idx])+".")
    overall_acc = sum(avg_accuracies)/len(avg_accuracies)
    print("Average model accuracy across folds: " + str(overall_acc) + ".")

    print("'all_test_data_features': (list of lists).")
    print("'all_test_labels': (list of lists).")
    print("'all_predictions': (list of lists).")

    print("'overall_acc': (float) model accuracy averaged across all sounds across all folds.")
    print("'avg_accuracies': (list) model accuracies averaged across all sounds for each split.")

    # accuracy for each category for each of the 5 splits
    category_accuracies = []
    for fold_idx in [0,1,2,3,4]:
        predictions = all_predictions[fold_idx]
        labels = all_labels[fold_idx]
        fold_accuracies = []
        for target in range(50):
            fold_acc = get_category_accuracies(predictions,labels,target)
            fold_accuracies.append(fold_acc)
        category_accuracies.append(fold_accuracies)
    print("'category_accuracies': (list) accuracies by category (for each of the 5 folds).")

    # SVM parameters that are learned for each of the 5 splits
    print("'all_svcs': (list) SVM parameters learned.")

    # get target-category mapping so it's in saved_vars.pickle
    fold_info_path = '/om4/group/mcdermott/user/michl/ESC-50-master/meta/esc50.csv'
    df = pandas.read_csv(fold_info_path)
    target_to_category = {}
    while len(target_to_category)<50:
        for index, row in df.iterrows():
            if row['target'] not in target_to_category:
                target_to_category[row['target']] = row['category']

    save_analysis_path = 'esc_analysis/%s_%s_nact%d_nreps%d_rs%d_avgtest%s_cvals%s'%(
          net_name, layer, num_activations, num_reps, SEED, average_test_predictions, 
          str(c_values))
    try: 
        os.makedirs(save_analysis_path)
    except FileExistsError:
        pass

    p = open(os.path.join(save_analysis_path, "saved_vars_"+str(layer)+".pickle"),"wb")
    new_dict = {'overall_acc':overall_acc, 'avg_accuracies':avg_accuracies, 
                'category_accuracies':category_accuracies, 'all_svcs':all_svcs, 
                'target_to_category':target_to_category, 
                'all_test_labels':all_test_labels, 'all_predictions':all_predictions}
    pckl.dump(new_dict, p)
    p.close()


    ### Plot ###
    
    # per-category accuracy with error bars that are the SEM across the 5 split
    category_accuracies = np.array([category_accuracies[0],
                                    category_accuracies[1],
                                    category_accuracies[2],
                                    category_accuracies[3],
                                    category_accuracies[4]])
    category_list = np.array(list(target_to_category.values()))
    means = np.average(category_accuracies, axis=0)
    SEMs = np.std(category_accuracies, axis=0)

    # Build the plot
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax.bar(category_list, means, yerr=SEMs, align='center', alpha=0.5, 
           ecolor='black', capsize=10)
    #ax.set_xticks(category_list)
    ax.set_title('Average Per Category Accuracy: Layer '+str(layer))
    ax.axhline(overall_acc, color='red', linewidth=2)

    # Save the figure and show
    plt.tight_layout()
    plt.xticks(category_list, rotation=90)
    plt.savefig(os.path.join(save_analysis_path, 
                             'avg_per_category_accuracy_'+str(layer)+'.pdf'), 
                bbox_inches="tight", transparent=True)
    plt.show()

    ### Confusion matrix ###
    # average confusion matrix across the 5 folds, using sklearn's confusion_matrix
    target_list = np.array(list(target_to_category.keys()))
    
    # get average confusion matrix across folds
    conf_matrices = []
    for fold in range(5):
        y_true = all_test_labels[fold]
        y_pred = all_predictions[fold]
        c = confusion_matrix(y_true, y_pred, labels=target_list, normalize='true')
        conf_matrices.append(c)
    avg_conf_matrix = np.mean(conf_matrices,0)

    #  plot it
    fig, ax = plt.subplots(figsize=(25, 25))
    plt.imshow(avg_conf_matrix,cmap='Blues')
    plt.colorbar()
    ax.set_xticks(np.arange(50))
    ax.set_yticks(np.arange(50))
    ax.set_xticklabels(category_list)
    ax.set_yticklabels(category_list)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
            rotation_mode="anchor")

    ax.set_title("Confusion matrix for layer "+str(layer))
    fig.tight_layout()
    plt.savefig(os.path.join(save_analysis_path, 
                             'confusion_matrix_'+str(layer)+'.pdf'), 
                bbox_inches="tight", transparent=True)


# TODO: build in a way to run the 'save activations' function by itself, so that we can submit it to a GPU and then run the rest of the script 
# as a CPU only job. Right now, one way to do this is to cut everything off after the first fold? 
if __name__ == '__main__':
    import build_network
    import argparse

    #########PARSE THE ARGUMENTS FOR THE FUNCTION#########
    parser = argparse.ArgumentParser(description='Input parameters for SVM evaluation of ESC-50')
    parser.add_argument('-D', '--SCRATCH_DIR', metavar='--D', type=str, help='Scratch directory for activations')
    parser.add_argument('-L', '--LAYER_IDX', metavar='--L', type=int, help='Index into "metamer_layers" ' \
                        'within build_network.py script for which layer to analyze') 
    parser.add_argument('-A', '--NUM_ACTIVATIONS', metavar='--A', default=4096, type=int, help='Number of activations ' \
                        'to use for the SVM. Full activations will be downsampled with random projections')
    parser.add_argument('-R', '--NUM_REPS', metavar='--R', default=5, type=int, help='Number of 2 sec clips randomly ' \
                        'chosen from each sound')
    parser.add_argument('-S', '--SEED', metavar='--S', default=0, type=int, help='Random seed')
    parser.add_argument('-P', '--AVG_TEST_PREDICTIONS', action='store_true', help='Average test ' \
                        'predictions')
    parser.add_argument('-C', '--C_VALUES', metavar='--C', type=float, nargs='+', default=[0.01,0.1, 1], 
                        help='C parameters to sweep over for SVM with cross validation')
    parser.add_argument('-O', '--OVERWRITE', action='store_true', help='If set, overwrites saved activations.')
    
    args=parser.parse_args()

    print(args)
    try:
        os.mkdir('plots')
    except FileExistsError:
        pass

    model, ds, metamer_layers = build_network.main(return_metamer_layers=True)
    print("Layer name: "+str(metamer_layers[args.LAYER_IDX]))
    model = model.cuda()
    net_path = os.getcwd()
    net_name = '-'.join(net_path.split('/')[-2:])
    get_predictions_and_make_plots(model, net_name,
                                   scratch_activations_dir=args.SCRATCH_DIR,
                                   layer=metamer_layers[args.LAYER_IDX],
                                   num_activations=args.NUM_ACTIVATIONS,
                                   num_reps=args.NUM_REPS,
                                   SEED=args.SEED,
                                   average_test_predictions=args.AVG_TEST_PREDICTIONS,
                                   c_values=args.C_VALUES, 
                                   overwrite=args.OVERWRITE)

