import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import pandas as pd


def evaluate_self_supervised(sess, model, flags, batch_gen, lin, test=True, mode="synch"):
    """ Evaluate self-supervised accuracy over 10 batches of the test set

    (sess) tensorflow session
    (flags) Options specified by tensorflow flags(flags).
    (batch_gen) data generator to evaluate
    (lin) Gradient Reversal layer hyper-parameter
    (test) Use training or test dataset. default test
    (target_data) boolean to specify if half the batch should contain target data. False: only source data is used.
    (mode) Self-supervised mode. Default correspondence classifier: "synch".
    """
    accuracy_list = []
    batch_gen.reset_dataset(test=test)
    for num_batch in range(10):
        np_test_x_rgb, np_test_x_flow, np_test_y, np_synch = \
            batch_gen.nextBatch(flags.batch_size, test=test)

        np_test_d = np.array([1] * np_test_x_rgb.shape[0])
        if flags.modality == "flow":
            feed_dict = {model.placeholders['is_training']: False,
                         model.placeholders['images_flow']: np_test_x_flow,
                         model.placeholders['labels']: np_test_y,
                         model.placeholders['dataset_ident']: np_test_d,
                         model.placeholders['flip_weight']: lin,
                         model.placeholders['dropout']: 1.0,
                         model.placeholders['synch']: np_synch}
        elif flags.modality == "rgb":
            feed_dict = {model.placeholders['is_training']: False,
                         model.placeholders['images_rgb']: np_test_x_rgb,
                         model.placeholders['labels']: np_test_y,
                         model.placeholders['dataset_ident']: np_test_d,
                         model.placeholders['flip_weight']: lin,
                         model.placeholders['dropout']: 1.0,
                         model.placeholders['synch']: np_synch}
        elif flags.modality == "joint":
            feed_dict = {model.placeholders['is_training']: False,
                         model.placeholders['images_flow']: np_test_x_flow,
                         model.placeholders['images_rgb']: np_test_x_rgb,
                         model.placeholders['labels']: np_test_y,
                         model.placeholders['dataset_ident']: np_test_d,
                         model.placeholders['flip_weight']: lin,
                         model.placeholders['dropout']: 1.0,
                         model.placeholders['synch']: np_synch}
        else:
            raise Exception("Unknown modality given: "+flags.modality)
        if mode == "synch":
            to_predict = model.prediction['synch_accuracy']
        else:
            raise Exception("Cannot evaluate self supervised accuracy if no selfsupervision technique has been chosen")
        np_accuracy = sess.run(
            to_predict,
            feed_dict=feed_dict)

        accuracy_list.append(np_accuracy)
    return np.mean(accuracy_list)


def evaluate(sess, model, flags, batch_gen, lin, test=True, out_features=False, extra_info=False):
    """Evaluate action performance

    Returns top1 action accurcy, top1 domain accuracy, Average class recall. Extra information including filenames,
    labels, feature can be returned by setting extra_info=True.

    (sess) tensorflow session
    (model) model to evalutae
    (flags) option specifed in tensorflow flags
    (batch_gen) data generator to evaluate
    (lin) Gradient Reversal layer hyper-parameter
    (test) Use training or test dataset. default test
    (out_features) output feature represenation produced before classification.
    (extra_info) return extra information, including feature representation, filename, labels, etc
    """

    # Average predictions from multiple timesteps (axis=0)
    # Compute top1 accuracy
    def correct(prediction, label_onehot):
        if np.all(np.equal(prediction,-1)):
            return np.zeros(label_onehot.shape[0],dtype=np.float32), np.zeros(label_onehot.shape[0],dtype=np.float32)
        prediction_avg = np.mean(prediction, 0)
        predicted = np.argmax(prediction_avg, 1)
        label = np.argmax(label_onehot, 1)
        return np.equal(label, predicted), predicted

    # Repeat examples to fit required batch size,
    # returns padded examples and original length of batch
    def pad_batch(x_rgb, x_flow, y):
        num_real_examples = x_rgb.shape[0]
        while x_rgb.shape[0] < flags.batch_size:
            x_rgb = np.concatenate([x_rgb, x_rgb], axis=0)
            x_flow = np.concatenate([x_flow, x_flow], axis=0)
            y = np.concatenate([y, y], axis=0)
        x_rgb = x_rgb[:flags.batch_size]
        x_flow = x_flow[:flags.batch_size]
        y = y[:flags.batch_size]
        # d = d[:(self.flags.batch_size)]
        return x_rgb, x_flow, y, num_real_examples

    done = False
    correct_list = []
    predicted_list = []
    feature_list = []
    filenames_list = []
    label_list = []
    correct_domain_list = []
    batch_gen.reset_dataset(test=test)
    while not done:
        # Get batch
        done, np_test_x_rgb_all, np_test_x_flow_all, np_test_y = \
            batch_gen.nextBatchEval(flags.batch_size, test=test)
        np_test_x_rgb_all, np_test_x_flow_all, np_test_y, num_real_examples = pad_batch(np_test_x_rgb_all,
                                                                                        np_test_x_flow_all,
                                                                                        np_test_y)
        np_test_d = np.array([1] * np_test_x_rgb_all.shape[0])


        # Evaluate on a number of timesteps per action segment
        np_logits_all = []
        np_domain_logits_all = []
        np_features = []
        np_filenames = None
        label_np = np.argmax(np_test_y, axis=1)
        np_test_x_rgb_all = np.swapaxes(np_test_x_rgb_all, 0, 1)
        np_test_x_flow_all = np.swapaxes(np_test_x_flow_all, 0, 1)
        for np_test_x_rgb, np_test_x_flow in zip(np_test_x_rgb_all, np_test_x_flow_all):
            synch = [True]*np_test_x_rgb.shape[0]
            if flags.modality == "flow":
                feed_dict = {model.placeholders['is_training']: False,
                             model.placeholders['images_flow']: np_test_x_flow,
                             model.placeholders['labels']: np_test_y,
                             model.placeholders['dataset_ident']: np_test_d,
                             model.placeholders['flip_weight']: lin,
                             model.placeholders['dropout']: 1.0,
                             model.placeholders['synch']: synch}
            elif flags.modality == "rgb":
                feed_dict = {model.placeholders['is_training']: False,
                             model.placeholders['images_rgb']: np_test_x_rgb,
                             model.placeholders['labels']: np_test_y,
                             model.placeholders['dataset_ident']: np_test_d,
                             model.placeholders['flip_weight']: lin,
                             model.placeholders['dropout']: 1.0,
                             model.placeholders['synch']: synch}
            elif flags.modality == "joint":
                feed_dict = {model.placeholders['is_training']: False,
                             model.placeholders['images_flow']: np_test_x_flow,
                             model.placeholders['images_rgb']: np_test_x_rgb,
                             model.placeholders['labels']: np_test_y,
                             model.placeholders['dataset_ident']: np_test_d,
                             model.placeholders['flip_weight']: lin,
                             model.placeholders['dropout']: 1.0,
                             model.placeholders['synch']: synch}
            else:
                raise Exception("Unknown modality"+flags.modality)

            filenames_np = np_test_x_rgb
            logits_to_run = model.prediction['logits']

            # Run Validation
            if out_features:
                np_logits, features_np, np_domain_logits = sess.run(
                    [logits_to_run, model.model['features_flow'], model.prediction['domain_logits']],
                    feed_dict=feed_dict)
            else:
                np_logits, np_domain_logits = sess.run(
                    [logits_to_run, model.prediction['domain_logits']],
                    feed_dict=feed_dict)
                features_np = np_logits
                # concatenate results ignoring dummy example predictions
            np_logits_all.append(np_logits)
            np_domain_logits_all.append(np_domain_logits)
            np_features.append(features_np)
            np_filenames = filenames_np

        np_features = np_features[2]

        # average predictions and compute top1 accuracy for each sample
        correct_np, predicted_np = correct(np_logits_all, np_test_y)

        # average domain prediction and compute top1 accuracy
        np_test_one_hot = np.zeros((np_test_d.shape[0], 2), dtype=np.int32)
        np_test_one_hot[np.arange(np_test_d.shape[0]), np_test_d] = 1
        correct_domain_np, predicted_domain_np = correct(np_domain_logits_all, np_test_one_hot)

        #Remove padded examples add append to information to lists
        correct_list = np.concatenate((correct_list, correct_np[:num_real_examples]))
        predicted_list = np.concatenate((predicted_list, predicted_np[:num_real_examples]))
        feature_list.append(np_features[:num_real_examples])
        filenames_list.append(np_filenames[:num_real_examples])
        label_list = np.concatenate((label_list, label_np[:num_real_examples]))
        correct_domain_list = np.concatenate((correct_domain_list, correct_domain_np[:num_real_examples]))

    # Concatenate all testing batches
    feature_list = np.concatenate(feature_list, axis=0)
    filenames_list = np.concatenate(filenames_list, axis=0)

    # Macro average accuracies
    valaccuracy = correct_list.mean()
    domainaccuracy = correct_domain_list.astype(float).mean()

    #Compute per class recall
    perclass = pd.DataFrame({'correct': correct_list, 'label': label_list}).groupby('label')['correct'].mean()

    if extra_info:
        return valaccuracy, domainaccuracy, perclass, feature_list, filenames_list, label_list, predicted_list
    else:
        return valaccuracy, domainaccuracy, perclass.mean()


def train_step(sess, model, flags, batch_gen, batch_gen_unseen, lin, target_data):
    """Run a single training step

    Performs forward, backward passes through model (model) and performs a single SGD step.

    (sess) tensorflow session
    (batch_gen) Source batch generator
    (batch_gen_unseen) Target batch generator
    (flags) Options specified by tensorflow flags(flags).
    (lin) Gradient Reversal layer hyper-parameter
    (target_data) boolean to specify if half the batch should contain target data. False: only source data is used.
    """

    summaries = model.get_summaries()

    if flags.steps_before_update < 1:
        raise Exception("Must update have a least one training step before update")

    sess.run(model.zero_grads)
    # Train the network
    loss_total = 0.0
    accuracy_train = 0.0
    summary_train = None
    for part_step in range(flags.steps_before_update):

        # Get target+source training data if adaptating
        if target_data:  # self.domain_loss or self.discrepancy or self.bn_align or self.mmd:
            np_x_rgb_seen, np_x_flow_seen, np_y_seen, np_synch_seen = batch_gen.nextBatch(
                flags.batch_size / 2)

            np_x_rgb_unseen, np_x_flow_unseen, np_y_unseen, np_synch_unseen = \
                batch_gen_unseen.nextBatch(flags.batch_size / 2)

            np_y_unseen = np.zeros([len(np_y_unseen), flags.num_labels])
            np_y_unseen[:, 0] = 1.0

            np_d_seen = len(np_y_seen) * [0]
            np_d_unseen = len(np_y_seen) * [1]

            np_d = np.concatenate([np_d_seen, np_d_unseen])
            np_y = np.concatenate([np_y_seen, np_y_unseen])
            np_x_flow = np.concatenate([np_x_flow_seen, np_x_flow_unseen])
            np_x_rgb = np.concatenate([np_x_rgb_seen, np_x_rgb_unseen])
            np_synch = np.concatenate([np_synch_seen, np_synch_unseen])

        # Get only source training data if normal training
        else:
            np_x_rgb, np_x_flow, np_y, np_synch = batch_gen.nextBatch(flags.batch_size)
            np_d = len(np_y) * [0]
        if flags.modality == "rgb":
            feed_dict = {model.placeholders['is_training']: True,
                         model.placeholders['images_rgb']: np_x_rgb,
                         model.placeholders['labels']: np_y,
                         model.placeholders['dataset_ident']: np_d,
                         model.placeholders['flip_weight']: lin,
                         model.placeholders['dropout']: 0.5,
                         model.placeholders['synch']: np_synch}
        elif flags.modality == "flow":
            feed_dict = {model.placeholders['is_training']: True,
                         model.placeholders['images_flow']: np_x_flow,
                         model.placeholders['labels']: np_y,
                         model.placeholders['dataset_ident']: np_d,
                         model.placeholders['flip_weight']: lin,
                         model.placeholders['dropout']: 0.5,
                         model.placeholders['synch']: np_synch}
        elif flags.modality == "joint":
            feed_dict = {model.placeholders['is_training']: True,
                         model.placeholders['images_rgb']: np_x_rgb,
                         model.placeholders['images_flow']: np_x_flow,
                         model.placeholders['labels']: np_y,
                         model.placeholders['dataset_ident']: np_d,
                         model.placeholders['flip_weight']: lin,
                         model.placeholders['dropout']: 0.5,
                         model.placeholders['synch']: np_synch}
        else:
            raise Exception("Unknown Modality"+flags.modality)

        _, summary_train, loss_total, accuracy_train = sess.run(
            [model.accum_grads, summaries, model.losses['total_loss'],
             model.prediction['accuracy']],
            feed_dict=feed_dict)

    sess.run(model.train_op)
    return loss_total, accuracy_train, summary_train
