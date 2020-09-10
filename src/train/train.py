import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_here, '../..'))

from src.models.model import Model
from src.data_gen.batch_generator import BatchGenerator
from src.train.training_loops import *
import time
from datetime import datetime
import tensorflow as tf


class TrainTestScript:
    """ Creates a framework to train/test an MM-SADA model
        (FLAGS)         TensorFlow flags
        (results_dir)   Directory of tensorboard files and other testing logs
        (train_dir)     Director of saved model

        Methods:
            train - train MM-SADA
            test  - evaluate an MM-SADA saved model
    """

    # Initialise model and results directory
    def __init__(self, FLAGS, results_dir, train_dir):
        # inputs
        self.FLAGS = FLAGS
        self.train_dir = train_dir
        self.datasets = FLAGS.datasets
        self.unseen_dataset = FLAGS.unseen_dataset
        self.num_gpus = FLAGS.num_gpus
        self.num_labels = FLAGS.num_labels
        self.target_data = not (not (FLAGS.domain_mode))

        if self.target_data:
            if FLAGS.domain_mode == "None" or FLAGS.domain_mode == "Pretrain":
                self.target_data = False
                print("No adaptation")

        if FLAGS.domain_mode:
            self.domain_mode = FLAGS.domain_mode
        else:
            self.domain_mode = "None"

        self.lr = FLAGS.lr

        if not FLAGS.modality:
            raise Exception("Need to Specify modality")

        if FLAGS.modality != "rgb" and FLAGS.modality != "flow" and FLAGS.modality != "joint":
            raise Exception("Invalid Modality")

        self.results_dir = results_dir + "_" + FLAGS.modality
        self.modality = FLAGS.modality

        #if self.domain_loss or self.bn_align or self.discrepancy or self.mmd:
        self.model = Model(num_gpus=self.num_gpus, num_labels=self.num_labels, modality=self.modality,
                           temporal_window=self.FLAGS.temporal_window, batch_norm_update=self.FLAGS.batch_norm_update,
                           domain_mode=self.domain_mode,steps_per_update=FLAGS.steps_before_update,
                           aux_classifier=self.FLAGS.aux_classifier, synchronised=self.FLAGS.synchronised,
                           predict_synch=self.FLAGS.pred_synch, selfsupervised_lambda=self.FLAGS.self_lambda)


    def training_batch_gen(self):
        batch_gen = BatchGenerator(self.num_labels, self.datasets,
                                   temporal_window=self.FLAGS.temporal_window,
                                   rgb_data_path=self.FLAGS.rgb_data_path, flow_data_path=self.FLAGS.flow_data_path,
                                   synchronised=self.FLAGS.synchronised, random_sync=self.FLAGS.pred_synch)
        batch_gen_unseen = BatchGenerator(self.num_labels, self.unseen_dataset,
                                          temporal_window=self.FLAGS.temporal_window,
                                          rgb_data_path=self.FLAGS.rgb_data_path, flow_data_path=self.FLAGS.flow_data_path,
                                          synchronised=self.FLAGS.synchronised, random_sync=self.FLAGS.pred_synch)
        return batch_gen, batch_gen_unseen

    def testing_batch_gen(self):
        batch_gen = BatchGenerator(self.num_labels, self.datasets,
                                    temporal_window=self.FLAGS.temporal_window,
                                    rgb_data_path=self.FLAGS.rgb_data_path, flow_data_path=self.FLAGS.flow_data_path,
                                   synchronised=self.FLAGS.synchronised, random_sync=self.FLAGS.pred_synch)
        batch_gen_unseen = BatchGenerator(self.num_labels, self.unseen_dataset,
                                          temporal_window=self.FLAGS.temporal_window,
                                          rgb_data_path=self.FLAGS.rgb_data_path, flow_data_path=self.FLAGS.flow_data_path,
                                          synchronised=self.FLAGS.synchronised, random_sync=self.FLAGS.pred_synch)
        return batch_gen, batch_gen_unseen


    def train(self):
        """ Train MM-SADA model"""
        g1 = tf.Graph()
        with g1.as_default(), tf.device('/cpu:0'):

            # Initialize savers
            self.model.init_savers()

            train_writer = tf.summary.FileWriter(self.results_dir + '/train')
            seen_writer = tf.summary.FileWriter(self.results_dir + '/seen')
            unseen_writer = tf.summary.FileWriter(self.results_dir + '/unseen')

            batch_gen, batch_gen_unseen = self.training_batch_gen()

            with tf.Session(graph=g1, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                print("init variables")
                sess.run(tf.global_variables_initializer())

                start_step = self.model.restore_model_train(sess, self.train_dir, self.FLAGS.restore_model_flow,
                                                            self.FLAGS.restore_model_rgb,
                                                            self.FLAGS.restore_model_joint,
                                                            self.FLAGS.restore_mode)

                # Iterate over training steps
                for step in range(int(start_step), self.FLAGS.max_steps+1):

                    # Gradient Reversal Layer hyperparameter
                    p = float(step) / self.FLAGS.max_steps
                    lin = (2 / (1. + np.exp(-10. * p)) - 1) * self.FLAGS.lambda_in

                    start_time = time.time()

                    # Perform single training step
                    training_loss, training_accuracy, summary = train_step(sess, self.model, self.FLAGS, batch_gen,
                                                                           batch_gen_unseen, lin, self.target_data)
                    for s in summary:
                        train_writer.add_summary(s, step)

                    duration = time.time() - start_time

                    # Evaluate the model periodically
                    if step % 50 == 0:

                        # Calculate the training efficiency
                        num_examples_per_step = self.FLAGS.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration

                        # Write Training Log Information
                        format_str = ('(Train) %s: step %d, loss %.3f, acc %.3f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (datetime.now(), step, training_loss, training_accuracy,
                                            examples_per_sec, sec_per_batch))

                        # Evaluate Source Kitchens
                        valaccuracy, domainaccuracy, average_class = evaluate(sess, self.model, self.FLAGS,
                                                                              batch_gen, lin)
                        domainaccuracy = 1.0 - domainaccuracy  # uses target data so flip domain loss
                        val_summary = tf.Summary()
                        val_summary.value.add(tag="acc/Accuracy", simple_value=valaccuracy)
                        domain_summary = tf.Summary()
                        domain_summary.value.add(tag="acc/Domain", simple_value=domainaccuracy)
                        seen_writer.add_summary(val_summary, step)
                        seen_writer.add_summary(domain_summary, step)
                        # Write Validation Log Information
                        format_str = '(Val) %s: domain:%s step:%d accuracy:%f avg_class %f domain_accuracy %f'
                        print(format_str % (
                        datetime.now(), "Source", step, valaccuracy, average_class, domainaccuracy))

                        if self.FLAGS.pred_synch:
                            synch_accuracy = evaluate_self_supervised(sess, self.model, self.FLAGS, batch_gen, lin,
                                                                      mode="synch")
                            val_summary = tf.Summary()
                            val_summary.value.add(tag="acc/Synch_Accuracy", simple_value=synch_accuracy)
                            seen_writer.add_summary(val_summary, step)
                            format_str = '(Val) %s: domain:%s step:%d synch_accuracy:%f'
                            print(format_str % (
                                datetime.now(), "Source", step, synch_accuracy))

                        # Evaluate Target Kitchen
                        valaccuracy, domainaccuracy, average_class = evaluate(sess, self.model, self.FLAGS, batch_gen_unseen, lin)
                        val_summary = tf.Summary()
                        val_summary.value.add(tag="acc/Accuracy", simple_value=valaccuracy)
                        domain_summary = tf.Summary()
                        domain_summary.value.add(tag="acc/Domain", simple_value=domainaccuracy)
                        unseen_writer.add_summary(val_summary, step)
                        unseen_writer.add_summary(domain_summary, step)
                        # Write Validation Log Information
                        format_str = '(Val) %s: domain:%s step:%d accuracy:%f avg_class %f  domain_accuracy %f'
                        print(format_str % (datetime.now(), "Target", step, valaccuracy, average_class, domainaccuracy))

                        if self.FLAGS.pred_synch:
                            synch_accuracy = evaluate_self_supervised(sess, self.model, self.FLAGS, batch_gen_unseen,
                                                                      lin, mode="synch")
                            val_summary = tf.Summary()
                            val_summary.value.add(tag="acc/Synch_Accuracy", simple_value=synch_accuracy)
                            unseen_writer.add_summary(val_summary, step)
                            format_str = '(Val) %s: domain:%s step:%d synch_accuracy:%f'
                            print(format_str % (
                                datetime.now(), "Target", step, synch_accuracy))

                    # Save the model checkpoint periodically.
                    if step % 50 == 0 or step == self.FLAGS.max_steps:
                        self.model.save_model(sess, self.train_dir, step)



    def test(self):
        """ Evaluate MM-SADA model"""

        def _save_results(FLAGS, feature_list, label_list, predict_list, img_path_list, ident, test=True):
            """ Save statistics and extracted features to feature_path folder"""
            if test:
                stringtest = "test"
            else:
                stringtest = "train"
            source_domain = os.path.basename(FLAGS.datasets)
            np.save(
                FLAGS.feature_path + "/" + stringtest + "_feat_" + source_domain + "_" + str(FLAGS.modelnum) + "_" + str(
                    ident),
                feature_list)
            np.save(
                FLAGS.feature_path + "/" + stringtest + "_label" + source_domain + "_" + str(FLAGS.modelnum) + "_" + str(
                    ident),
                label_list)
            np.save(
                FLAGS.feature_path + "/" + stringtest + "_pred" + source_domain + "_" + str(FLAGS.modelnum) + "_" + str(
                    ident),
                predict_list)
            np.save(FLAGS.feature_path + "/" + stringtest + "_filenames" + source_domain + "_" + str(
                FLAGS.modelnum) + "_" + str(ident),
                    img_path_list)

        with tf.Graph().as_default(), tf.device('/cpu:0'):
            batch_gen, batch_gen_unseen = self.testing_batch_gen()

            self.model.init_savers()

            # Run Graph
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

                # restore checkpoint
                self.model.restore_model_test(sess, self.train_dir, self.FLAGS.modelnum)
                lin = 0.0
                step = 0

                # Evaluate Seen Kitchens
                seen_filenames = ""
                seen_accuracy = ""

                #Evaluate Source
                valaccuracy, domainaccuracy, valperclass, valfeat, valfile, vallabel, valpredict  = \
                    evaluate(sess, self.model, self.FLAGS, batch_gen, lin,
                                  test=(not self.FLAGS.eval_train), out_features=self.FLAGS.features,
                                  extra_info=True)
                if self.FLAGS.features:
                    _save_results(self.FLAGS, valfeat, vallabel, valpredict, valfile, "Source",
                              test=(not self.FLAGS.eval_train))
                seen_accuracy = seen_accuracy + str(valaccuracy) + ","
                seen_filenames = seen_filenames + "Source" + ","
                # Write Validation Log Information
                format_str = '(Val) %s: domain:%s step:%d accuracy:%f domain_accuracy %f'
                print(format_str % (datetime.now(), "Source", step, valaccuracy, domainaccuracy))

                # Evaluate Target
                valaccuracy, domainaccuracy, valperclass, valfeat, valfile, vallabel, valpredict  = \
                    evaluate(sess, self.model, self.FLAGS, batch_gen_unseen, lin, test=(not self.FLAGS.eval_train),
                                       out_features=self.FLAGS.features, extra_info=True)
                domainaccuracy = 1.0 - domainaccuracy
                if self.FLAGS.features:
                    _save_results(self.FLAGS, valfeat, vallabel, valpredict, valfile, "Target",
                              test=(not self.FLAGS.eval_train))
                # Write Validation Log Information
                format_str = '(Val) %s: domain:%s step:%d accuracy:%f domain_accuracy %f'
                print(format_str % (datetime.now(), "Target", step, valaccuracy, domainaccuracy))

                results_log_file = '/logs/results.list'
                if not os.path.exists(self.results_dir + "/logs"):
                    os.makedirs(self.results_dir + "/logs")

                if not os.path.isfile(self.results_dir + results_log_file):
                    f = open(self.results_dir + results_log_file, 'w')
                    f.write(seen_filenames + "target,step,target_directory" + "\n")
                    f.close()

                f = open(self.results_dir + results_log_file, 'a')
                f.write(seen_accuracy + str(valaccuracy) + "," + str(
                    self.FLAGS.modelnum) + "," + self.FLAGS.unseen_dataset + "\n")
                f.close()




def parse_args(FLAGS):
    error = False
    if FLAGS.train is None:
        print("Specify whether to train (True) or test (False) --train")
        error = True
    if FLAGS.results_path is None:
        print("Specify path to save logs and models --results_path")
        error = True
    if FLAGS.datasets is None:
        print("Specify the Source domain dataset --datasets")
        error = True
    if FLAGS.unseen_dataset is None:
        print("Specify the Target domain dataset --unseen_dataset")
        error = True
    if FLAGS.rgb_data_path is None:
        print("Specify the path to rgb frames --rgb_data_path")
        error = True
    if FLAGS.flow_data_path is None:
        print("Specify the path to flow frames --flow_data_path")
        error = True
    if error:
        return True

    if FLAGS.train:
        if FLAGS.restore_mode is None:
            print("Specify the restore mode --restore_mode ('pretrain', 'model', 'continue')")
            error = True
        if (FLAGS.restore_model_flow is None) and (FLAGS.flow):
            print('Specify pretrained model to use --pretrained_model')
            error = True
        if (FLAGS.restore_model_rgb is None) and (not FLAGS.flow):
            print('Specify pretrained model to use --pretrained_model')
            error = True
    else:
        if FLAGS.modelnum is None:
            print("Specify model number to restore for testing --modelnum")
            error = True
        if FLAGS.features:
            if FLAGS.feature_path is None:
                print("Specify path to store features --feature_path")
                error = True
    if error:
        return True
    return False


def input_parser():
    flags = tf.app.flags

    # Train or test flag
    flags.DEFINE_boolean('train', None, 'Weither to train or evaluate (False)')

    # Where to store log files
    flags.DEFINE_string('results_path', None, 'Where to store the log files and saved models')

    # Training flags
    flags.DEFINE_float('lr', 0.001, 'Initial Learning Rate')
    flags.DEFINE_float('batch_norm_update', 0.9, 'Update rate of batch norm statistics')
    flags.DEFINE_integer('num_gpus', 8, "number of gpus to run")
    flags.DEFINE_integer('max_steps', 6000, """Number of batches to run.""")
    flags.DEFINE_integer('steps_before_update', 1, "number of steps to run before updating weights")

    # Loss hyperparameters
    flags.DEFINE_string('domain_mode', None, 'background only for dataset2')
    flags.DEFINE_float('lambda_in',1.0,'grl hyperparameter')
    flags.DEFINE_float('self_lambda', 5.0, 'weigthing of self supervised loss')

    # Dataset commands
    flags.DEFINE_string('datasets', None, "Comma seperated list of datasets")
    flags.DEFINE_string('unseen_dataset', None, 'Specify file path to unseen dataset folder')
    flags.DEFINE_integer('num_labels', 8, 'Total number of combined labels')

    # Batch Generation aguments
    flags.DEFINE_integer('batch_size', 128, 'Size of a batch')
    flags.DEFINE_boolean('synchronised', None, 'Weither to synchronise flow and rgb')

    # Archetecture
    flags.DEFINE_string('modality', "joint", 'rgb, flow or joint (default: joint)')
    flags.DEFINE_integer('temporal_window', 16, "i3d temporal window")
    flags.DEFINE_boolean('aux_classifier', None, '2 classifiers')
    flags.DEFINE_boolean('pred_synch', None, 'Predict if modalities are synchronised')

    # Evaluation commands
    flags.DEFINE_boolean('features', None, 'Weither to produce features of evalutate')
    flags.DEFINE_string('feature_path', None, "path to store features")
    flags.DEFINE_boolean('eval_train', None, 'Weither to evaludate training example rather than test')
    flags.DEFINE_integer('modelnum', None, "model number to restore for testing")

    # load weights and train part of model according to these rules
    flags.DEFINE_string('restore_model_rgb',
                        None,
                        'Load these weights excluding Logits')
    flags.DEFINE_string('restore_model_flow',
                        None,
                        'Load these weights excluding Logits')
    flags.DEFINE_string('restore_model_joint',
                        None,
                        'Load these weights excluding Logits')
    flags.DEFINE_string('rgb_data_path', None, "path to rgb data")
    flags.DEFINE_string('flow_data_path', None, "path to flow data")
    flags.DEFINE_string('restore_mode', None, "pretrain (for base netwrok without logits),"
                                              " model (restore base model with classification logits) "
                                              " or continue (restore everything)")
    FLAGS = flags.FLAGS

    # Create Directories to store saved models and log files respectively
    source_domain = os.path.basename(FLAGS.datasets)
    target_domain = os.path.basename(FLAGS.unseen_dataset)
    train_dir = FLAGS.results_path + "/saved_model_" + source_domain + "_" + target_domain + "_" + str(FLAGS.lr) + "_" + str(
        FLAGS.batch_norm_update)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    results_dir = FLAGS.results_path + "/results_" + source_domain + "_" + target_domain + "_" + str(FLAGS.lr) + "_" + str(
        FLAGS.batch_norm_update)
    return FLAGS, train_dir, results_dir


def main():
    flags, train_dir, results_dir = input_parser()
    if parse_args(flags):
        return
    train_test = TrainTestScript(flags, results_dir, train_dir)
    if flags.train:
        train_test.train()
    else:
        train_test.test()


if __name__ == "__main__":
    main()
