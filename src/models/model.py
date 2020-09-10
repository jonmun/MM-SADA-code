from src.models.flip_gradient import flip_gradient
from src.models.arch import *

from src.models.mmd import *
import src.models.utils as ult
import os


class Model:
    """ MM-SADA model definition

     (num_gpus) number of gpus to split the batch. batch size must be a multiple of num_gpus.
     (num_labels) Output side of classification head.
     (feature_layer) Layer of i3d for feature extraction/MMD/Domain discrimination. Deafult after avg pool 'features'
     (temporal window) Number of frames in video segment
     (batch_norm_update) Adjusts how quickly batch norm statistics are updated
     (modality) Modalities in model. Either "rgb", "flow" or late fusion "joint"
     (domain_mode) Mode for domain adaptation and/or allowing unlabelled target data during pretraining,
     (learning_rate) learning rate
     (steps_per_update) Number of forward-backward passes before SGD step
     (gradient_reversal) Negate gradient in backward pass after domain discriminators,
     (aux_classifier) Use multiple classification heads, overrided as true if using MCD baseline
     (synchronised) synchronise flow and rgb augmentations
     (predict_synch) Use correspondence classification head
     (selfsupervised_lambda) weighting of self-supervised alignment loss,
     (lambda_class) Override weighting of classification loss. Default 0.2 unless pretraining lambda_class=1.0.

     Train network: A series of properties on model object can be used to train the network:
     model - dictionary of model outputs/logits
     losses - dictionary of losses
     placeholders - dictinoary of input places to the model
     predictions - softmax predictions of the model and metrics
     zero_grads - zero accumulated gradients
     accum_grads - forward-backward pass and accumulated gradients
     train_op - SGD step with accumulated gradients

     Save/load models: A series of methods on model object to save/load models
     init_savers - initialise model savers
     restore_model_train - restore model for training. Options "pretrain" (only base model, no classification heads),
                           "model" (only base model with action classification head) or "continue" (all weights in model).
     restore_model_test - restore model for testing.

     get_summaries - return TensorFlow summaries


    """
    def __init__(self, num_gpus, num_labels=10,  feature_layer='features',
                 temporal_window=16, batch_norm_update=0.9, modality="joint", domain_mode="None",
                 learning_rate=0.01, steps_per_update=1, gradient_reversal=True,
                 aux_classifier=False, synchronised=False, predict_synch=False,  selfsupervised_lambda=5,
                 lambda_class=False):
        self.selfsupervised_lambda=selfsupervised_lambda
        self.synchronised = synchronised
        self.modality = modality
        self.num_gpus = num_gpus
        self.steps_per_update = steps_per_update
        self.num_labels = num_labels
        self.network_output = self.num_labels
        self.temporal_window = temporal_window
        self.batch_norm_update = batch_norm_update
        self.predict_synch = predict_synch
        self.__placeholders = None
        self.__predictions = None
        self.__losses = None
        self.__model = None
        self.__train_op = None
        self.__savers_rgb = None
        self.__savers_flow = None
        self.__savers_joint = None
        self.feat_level = feature_layer
        self.summaries = []

        # Test for valid options
        if not (domain_mode == "DANN"  or domain_mode == "BN" or
                domain_mode == "MMD" or domain_mode == "MCD" or domain_mode == "None" or domain_mode == "Pretrain" or domain_mode == "PretrainM"):
            raise Exception("Invalid domain_mode option: {}".format(domain_mode))

        # Modality specification
        if self.modality == "joint":
            self.joint_classifier = True
            self.rgb = True
            self.flow = True
        elif self.modality == "rgb" or self.modality == "flow":
            self.joint_classifier = False
            self.rgb = self.modality == "rgb"
            self.flow = self.modality == "flow"
        else:
            raise Exception("Invalid modality option: {}".format(self.modality))

        self.flip_weight = gradient_reversal

        if domain_mode == "None" or domain_mode == "Pretrain":
            self.target_data = False
        else:
            self.target_data = True

        self.domain_loss = (domain_mode == "DANN")
        self.mmd = (domain_mode == "MMD")
        self.MaxClassDiscrepany = (domain_mode == "MCD")
        self.bn_align = (domain_mode == "BN")

        if lambda_class:
            self.softmax_lambda = 1.0
        else:
            self.softmax_lambda = 1.0 if (domain_mode == "Pretrain" or domain_mode == "PretrainM") else 0.2

        self.aux_classifier = aux_classifier or self.MaxClassDiscrepany
        self.opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

    @property
    def placeholders(self):
        if not self.__placeholders:
            self.__placeholders = dict()
            self.__placeholders['is_training'] = tf.placeholder(tf.bool)
            self.__placeholders['dataset_ident'] = tf.placeholder(tf.int32, [None, ])
            self.__placeholders['lambda_rate'] = tf.placeholder(tf.float32)
            self.__placeholders['dropout'] = tf.placeholder(tf.float32)
            self.__placeholders['synch'] = tf.placeholder(tf.bool, [None, ])
            self.__placeholders['images_rgb'] = tf.placeholder(tf.string, [None, self.temporal_window])
            self.__placeholders['images_flow'] = tf.placeholder(tf.string, [None, self.temporal_window, 2])
            self.__placeholders['flip_weight'] = tf.placeholder(tf.float32)
            self.__placeholders['labels'] = tf.placeholder(tf.float32, [None, self.num_labels])
            self.__placeholders['global_step'] = tf.Variable(0, trainable=False, name='global_step')
        return self.__placeholders

    @property
    def model(self):
        """Creates a model from the model class to work with multiple GPUs

        Returns a dictionary of outputs of the model:
            (logits_rgb)        logits of rgb classifier
            (logits_flow)       logits of flow classifier
            (logits_fusion)     logits of late fusision classifier
            (features_rgb)      features in rgb base model at layer self.feat_level
            (features_flow)     features in flow base model at layer self.feat_level
            (domain_logits_rgb) logits of rgb domain classifier
            (domain_logits_flow)logits of flow domain classifier
            (batch_norm_updates)batch normalisation update op
            (logits_aux_rgb)    logits of auxiliary rgb classifier (for MCD baseline)
            (logits_aux_flow)   logits of auxiliary flow classifier (for MCD baseline)
            (logits_aux_fusion) logits of late fusion auxiliary classifier (for MCD baseline)
            (synch_logits)      logits of correspondence classifier

        If a given output does not exist, it is assigned -1 in the dictionary.
        """
        if not self.__model:
            domains = self.placeholders['dataset_ident']
            logits_rgb_list = []
            logits_flow_list = []
            logits_fusion_list = []
            domain_logits_rgb_list = []
            domain_logits_flow_list = []
            synch_logits_list = []
            features_rgb_list = []
            features_flow_list = []
            logits_aux_flow_list = []
            logits_aux_rgb_list = []
            logits_aux_fusion_list = []
            reuse_variables = None
            num_models = max(self.num_gpus, 2)
            if self.rgb:
                input_images_splits_rgb = tf.split(axis=0, num_or_size_splits=num_models, value=self.placeholders['images_rgb'])
            if self.flow:
                input_images_splits_flow = tf.split(axis=0, num_or_size_splits=num_models, value=self.placeholders['images_flow'])
            domain_splits = tf.split(axis=0, num_or_size_splits=num_models, value=domains)

            # Initialise Architecture over each gpu, reuse variables from 1st iteration
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(num_models):
                    with tf.device('/device:GPU:%d' % (i if self.num_gpus > 1 else 0)):
                        with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                            if i < num_models / 2 or (not self.target_data):
                                assert_op = tf.Assert(tf.logical_or(tf.logical_not(self.placeholders['is_training']),
                                                                    tf.reduce_all(tf.equal(domain_splits[i], 0))),
                                                      [domain_splits[i]])
                                flip_classifier_grad = False
                            else:
                                assert_op = tf.Assert(tf.logical_or(tf.logical_not(self.placeholders['is_training']),
                                                                    tf.reduce_all(tf.equal(domain_splits[i], 1))),
                                                      [domain_splits[i]])
                                if self.MaxClassDiscrepany:
                                    flip_classifier_grad = True
                                else:
                                    flip_classifier_grad = False
                            domain_logits_flow = -1
                            with tf.control_dependencies([assert_op]):
                                if self.flow and self.rgb:
                                    logits_rgb, logits_flow, logits_fusion, features_rgb, features_flow, domain_logits_rgb, \
                                    domain_logits_flow, logits_aux_rgb, logits_aux_flow, logits_aux_fusion, logits_synch = \
                                        self.__build_model_joint(input_images_splits_rgb[i],
                                        input_images_splits_flow[i],
                                        reuse_variables=reuse_variables,
                                        flip_classifier_grad=flip_classifier_grad)
                                else:
                                    logits_fusion = logits_synch = logits_aux_fusion = None
                                    if self.rgb:
                                        logits_flow = features_flow = domain_logits_flow = logits_aux_flow = None
                                        logits_rgb, features_rgb, domain_logits_rgb, logits_aux_rgb = self.__build_model(
                                            input_images_splits_rgb[i], reuse_variables=reuse_variables, flow=False, flip_classifier_grad=flip_classifier_grad)
                                    if self.flow:
                                        logits_rgb = features_rgb = domain_logits_rgb = logits_aux_rgb = None
                                        logits_flow, features_flow, domain_logits_flow, logits_aux_flow = self.__build_model(
                                            input_images_splits_flow[i], reuse_variables=reuse_variables, flow=True, flip_classifier_grad=flip_classifier_grad)

                            reuse_variables = True
                            batch_norm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                            logits_rgb_list.append(logits_rgb)
                            logits_flow_list.append(logits_flow)
                            logits_fusion_list.append(logits_fusion)
                            domain_logits_rgb_list.append(domain_logits_rgb)
                            domain_logits_flow_list.append(domain_logits_flow)
                            features_rgb_list.append(features_rgb)
                            features_flow_list.append(features_flow)
                            logits_aux_flow_list.append(logits_aux_flow)
                            logits_aux_rgb_list.append(logits_aux_rgb)
                            logits_aux_fusion_list.append(logits_aux_fusion)
                            synch_logits_list.append(logits_synch)
                # Concatenate results from all GPUs

                if self.rgb and (not self.joint_classifier):
                    logits_rgb = tf.concat(logits_rgb_list, 0)
                    features_rgb = tf.concat(features_rgb_list, 0)
                else:
                    logits_rgb = -1
                    features_rgb = -1

                if self.flow and (not self.joint_classifier):
                    logits_flow = tf.concat(logits_flow_list, 0)
                    features_flow = tf.concat(features_flow_list, 0)
                else:
                    logits_flow = -1
                    features_flow = -1

                if self.rgb and self.flow and self.joint_classifier:
                    logits_fusion = tf.concat(logits_fusion_list, 0)
                    features_rgb = tf.concat(features_rgb_list, 0)
                    features_flow = tf.concat(features_flow_list, 0)
                else:
                    logits_fusion = -1

                if self.rgb and self.domain_loss:
                    domain_logits_rgb = tf.concat(domain_logits_rgb_list, 0)
                else:
                    domain_logits_rgb = -1

                if self.flow and self.domain_loss:
                    domain_logits_flow = tf.concat(domain_logits_flow_list, 0)
                else:
                    domain_logits_flow = -1

                logits_aux_rgb = tf.concat(logits_aux_rgb_list, 0) if (self.rgb and self.aux_classifier and not self.joint_classifier) else -1
                logits_aux_flow = tf.concat(logits_aux_flow_list,0) if (self.flow and self.aux_classifier and not self.joint_classifier) else -1
                logits_aux_fusion = tf.concat(logits_aux_fusion_list, 0) if (self.rgb and self.flow and self.aux_classifier and self.joint_classifier) else -1

                synch_logits = tf.concat(synch_logits_list,0) if self.predict_synch else -1

                self.__model = {'logits_rgb': logits_rgb, 'logits_flow': logits_flow, 'logits_fusion': logits_fusion,
                                'features_rgb': features_rgb, 'features_flow': features_flow,
                                'domain_logits_rgb': domain_logits_rgb, 'domain_logits_flow': domain_logits_flow,
                                'batch_norm_updates': batch_norm_updates,
                                'logits_aux_rgb': logits_aux_rgb, 'logits_aux_flow': logits_aux_flow,
                                'logits_aux_fusion':logits_aux_fusion,
                                "synch_logits": synch_logits}

        return self.__model

    def init_savers(self):
        """ Initialise savers for save and load the model
        """
        if not self.__model:
            self.model
            self.losses
            self.prediction
            self.train_op

        if self.flow:
            pretrain_loader, model_loader, savesave = ult.init_savers_base(True)
            self.__savers_flow = {"pretrain": pretrain_loader,"model": model_loader,
                                  "continue": savesave}
        if self.rgb:
            pretrain_loader, model_loader, savesave = ult.init_savers_base(False)
            self.__savers_rgb = {"pretrain": pretrain_loader,"model": model_loader,
                                 "continue": savesave}
        if self.modality == "joint" and self.predict_synch:
            model_loader = ult.read_joint("restore")
            joint_saver = ult.read_joint("continue")
            self.__savers_joint = {"model": model_loader, "continue": joint_saver}

    def restore_model_train(self, sess, checkpoint_path, restore_model_flow, restore_model_rgb, restore_model_joint,
                            restore_mode="model"):
        """ Restore all weights of model for training
            (sess)                  TensorFlow session
            (checkpoint_path)       Path to current model (if continueing training)
            (restore_model_flow)    Path to pretrained flow model
            (restore_model_rgb)     Path to pretrained rgb model
            (restore_model_joint)   Path to pretrained fusion module (e.g. correspondence classifier)
            (restore_mode)          Specify what to restore:
                                    'continue' continue training current model
                                    'model'    load rgb and flow base models + correspondence and action classifier
                                                                               heads (if applicable)
                                    'pretrain' load rgb and flow base without classification heads
        """
        if self.flow and self.rgb:
            start_step_flow = ult.restore_base(sess, self.__savers_flow, checkpoint_path + "/flow",
                                                  restore_model_flow, restore_mode)
            start_step_rgb = ult.restore_base(sess, self.__savers_rgb, checkpoint_path + "/rgb", restore_model_rgb,
                                                 restore_mode)
            if self.predict_synch:
                start_step_joint = ult.restore_joint(sess, self.__savers_joint, checkpoint_path + "/joint",
                                                        restore_model_joint, restore_mode)

                if restore_mode == "continue" and (
                        start_step_flow != start_step_rgb and start_step_flow != start_step_joint):
                    raise Exception("Different steps in checkpoint files for flow, rgb and joint")
            else:
                if restore_mode == "continue" and (start_step_flow != start_step_rgb):
                    raise Exception("Different steps in checkpoint files for flow, rgb and joint")
            start_step = start_step_rgb
        elif self.flow:
            start_step = ult.restore_base(sess, self.__savers_flow, checkpoint_path + "/flow", restore_model_flow,
                                             restore_mode)
        elif self.rgb:
            start_step = ult.restore_base(sess, self.__savers_rgb, checkpoint_path + "/rgb", restore_model_rgb,
                                             restore_mode)
        else:
            raise Exception("Invalid Modality in restore model train")
        return start_step

    def restore_model_test(self, sess, checkpoint_path, modelnum):
        """ Restore all weights of model for testing
            (checkpoint_path)   path to folder containing checkpoints
            (modelnum)          integer step of trained model to load
        """
        if self.flow:
            ckpnum = checkpoint_path + "/flow/model.ckpt-" + str(modelnum)
            self.__savers_flow['continue'].restore(sess, str(ckpnum))
        if self.rgb:
            ckpnum = checkpoint_path + "/rgb/model.ckpt-" + str(modelnum)
            self.__savers_rgb['continue'].restore(sess, str(ckpnum))
        if self.flow and self.rgb and self.predict_synch:
            ckpnum = checkpoint_path + "/joint/model.ckpt-" + str(modelnum)
            self.__savers_joint['model'].restore(sess, str(ckpnum))

    def save_model(self, sess, checkpoint_path_base, step):
        """ save weights of model
            (sess)                  TensorFlow session
            (checkpoint_path_base)  Folder to save checkpoints
            (step)                  current step of trained model
        """
        if self.flow:
            checkpoint_path = os.path.join(checkpoint_path_base + "/flow", 'model.ckpt')
            self.__savers_flow['continue'].save(sess, checkpoint_path, global_step=step)
        if self.rgb:
            checkpoint_path = os.path.join(checkpoint_path_base + "/rgb", 'model.ckpt')
            self.__savers_rgb['continue'].save(sess, checkpoint_path, global_step=step)
        if self.flow and self.rgb and self.predict_synch:
            checkpoint_path = os.path.join(checkpoint_path_base + "/joint", 'model.ckpt')
            self.__savers_joint['continue'].save(sess, checkpoint_path, global_step=step)

    def __build_model(self, filenames, reuse_variables=False, flow=False,flip_classifier_grad=False):
        """ Build DA architecture for a single modality """
        if flow:
            variable_scope = "Flow"
        else:
            variable_scope = 'RGB'

        data_augmentation = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=flow)
        images = tf.map_fn(lambda x: data_augmentation.preprocess(x, self.placeholders['is_training']),
                               filenames, dtype=tf.float32)
        with tf.variable_scope(variable_scope, reuse=reuse_variables):
            logits, logits_aux, features = self.__build_arch(images, reuse_variables=reuse_variables, flow=flow,flip_classifier_grad=flip_classifier_grad)
            feat_flip = flip_gradient(features, self.placeholders['flip_weight'])

            if self.domain_loss:
                domain_logits = domain_classifier(feat_flip)
            else:
                domain_logits = tf.zeros([0, 2], tf.float32)

        return logits, features, domain_logits,logits_aux,

    def __build_model_joint(self, filenames_rgb, filenames_flow, reuse_variables=False, flip_classifier_grad=False):
        """ Build DA architecture for a multi-modal architecture """
        if self.synchronised:
            filenames = tf.concat([tf.expand_dims(filenames_rgb, -1), filenames_flow], axis=-1)
            data_augmentation = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=True)
            images_rgb, images_flow = tf.map_fn(lambda x: data_augmentation.preprocess_rgb_flow(x, self.placeholders['is_training']), filenames, dtype=(tf.float32, tf.float32))
        else:
            data_augmentation_rgb = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=False)
            images_rgb = tf.map_fn(lambda x: data_augmentation_rgb.preprocess(x, self.placeholders['is_training']), filenames_rgb, dtype=tf.float32)
            data_augmentation_flow = DataAugmentation(224, [1, .875, 0.75], 256, 456, flow=True)
            images_flow = tf.map_fn(lambda x: data_augmentation_flow.preprocess(x, self.placeholders['is_training']), filenames_flow, dtype=tf.float32)

        with tf.variable_scope("Flow", reuse=reuse_variables):
            logits_flow, logits_aux_flow, features_flow = self.__build_arch(images_flow, reuse_variables=reuse_variables, flow=True,
                                                           flip_classifier_grad=flip_classifier_grad)
        with tf.variable_scope("RGB", reuse=reuse_variables):
            logits_rgb, logits_aux_rgb, features_rgb = self.__build_arch(images_rgb, reuse_variables=reuse_variables, flow=False,
                                                         flip_classifier_grad=flip_classifier_grad)

        # Create a fusion archetecture
        if self.joint_classifier:
            ######ARCH DEFINITION
            with tf.variable_scope("Joint", reuse=reuse_variables):
                # fusion layer
                def fusion(feat_rgb,feat_flow,log_rgb,log_flow):
                    logits_fusion = tf.reduce_sum([log_rgb, log_flow], axis=0)
                    features_fusion = None
                    return logits_fusion, features_fusion
                logits_fusion, features_fusion = fusion(features_rgb, features_flow, logits_rgb, logits_flow)
                if self.aux_classifier:
                    logits_aux_fusion, features_fusion_aux = fusion(features_rgb, features_flow, logits_aux_rgb, logits_aux_flow)
                else:
                    logits_aux_fusion = features_fusion_aux = None
                logits_rgb = None
                logits_aux_rgb = None
                logits_flow = None
                logits_aux_flow = None
        else:
            logits_fusion = None
            logits_aux_fusion = None

        if self.domain_loss:
            with tf.variable_scope("Flow", reuse=reuse_variables):
                if self.flip_weight:
                    feat_flip_flow = flip_gradient(features_flow, self.placeholders['flip_weight'])
                else:
                    feat_flip_flow = features_flow
                if self.domain_loss:
                    domain_logits_flow = domain_classifier(feat_flip_flow)
                else:
                    domain_logits_flow = tf.zeros([0, 2], tf.float32)

            with tf.variable_scope("RGB", reuse=reuse_variables):
                if self.flip_weight:
                    feat_flip_rgb = flip_gradient(features_rgb, self.placeholders['flip_weight'])
                else:
                    feat_flip_rgb = features_rgb
                if self.domain_loss:
                    domain_logits_rgb = domain_classifier(feat_flip_rgb)
                else:
                    domain_logits_rgb = tf.zeros([0, 2], tf.float32)
        else:
            domain_logits_rgb = None
            domain_logits_flow = None

        if self.predict_synch:
            with tf.variable_scope("Joint", reuse=reuse_variables):
                with tf.variable_scope("synch_test", reuse=reuse_variables):
                    logits_synch, synch_feat = predict_synch(tf.concat([features_rgb, features_flow], -1))
        else:
            logits_synch = None

        shape_rgb = features_rgb.get_shape().as_list()
        dim_rgb = np.prod(shape_rgb[1:])
        features_rgb = tf.reshape(features_rgb, [-1, dim_rgb])
        shape_flow = features_flow.get_shape().as_list()
        dim_flow = np.prod(shape_flow[1:])
        features_flow = tf.reshape(features_flow, [-1, dim_flow])
        return logits_rgb, logits_flow, logits_fusion, features_rgb, features_flow, domain_logits_rgb, domain_logits_flow,\
               logits_aux_rgb, logits_aux_flow, logits_aux_fusion, logits_synch

    def __build_arch(self, images, reuse_variables=False, flow=False, flip_classifier_grad=False):
        """ Build the base architecture for a single modality """
        logits, logits_aux, features = build_i3d(reuse_variables, images, self.placeholders['is_training'], self.network_output,
                                         flow, self.temporal_window, dropout=self.placeholders['dropout'],
                                         flip_classifier_gradient=flip_classifier_grad,
                                         flip_weight=self.placeholders['flip_weight'],aux_classifier=self.aux_classifier,
                                         feat_level=self.feat_level)
        return logits, logits_aux, features

    @property
    def losses(self):
        """ Produces a dictionary of losses for MM-SADA:
                'total_loss':    sum of all losses
                'domain_loss':   sum of domain discriminator losses
                'mmd_loss':      sum of MMD losses
        """
        if not self.__losses:
            dataset_ident = self.placeholders['dataset_ident']
            labels = self.placeholders['labels']

            def get_domain_loss(logits, domain_labels):
                return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=domain_labels, logits=logits))

            with tf.name_scope('loss'):

                # Apply Alignment losses
                with tf.name_scope('domain_loss'):

                    domain_disc_labels = tf.one_hot(dataset_ident, 2)

                    domain_loss = tf.constant(0.0)
                    if self.domain_loss:
                        if self.rgb and self.domain_loss:
                            domain_loss_rgb = get_domain_loss(self.model['domain_logits_rgb'], domain_disc_labels)
                            self.summaries.append(tf.summary.scalar('Domain Loss RGB', domain_loss_rgb))
                            domain_loss += domain_loss_rgb
                        if self.flow and self.domain_loss:
                            domain_loss_flow = get_domain_loss(self.model['domain_logits_flow'], domain_disc_labels)
                            self.summaries.append(tf.summary.scalar('Domain Loss Flow', domain_loss_flow))
                            domain_loss += domain_loss_flow

                    # flip label and apply calc domain loss

                with tf.name_scope("Predict_Sychronised"):
                    synch_loss = tf.constant(0.0)
                    if self.predict_synch and self.rgb and self.flow:
                        synch_labels = tf.cast(self.placeholders['synch'], dtype=tf.int32)
                        synch_logits = self.model['synch_logits']
                        synch_labels = tf.one_hot(synch_labels, 2)
                        synch_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=synch_labels, logits=synch_logits))
                        self.summaries.append(tf.summary.scalar('Synch loss', synch_loss))

                with tf.name_scope('MMD_loss'):
                    mmd_loss = tf.constant(0.0)
                    if self.mmd:
                        if self.rgb:
                            rgb_features_1, rgb_features_2 = tf.dynamic_partition(self.model['features_rgb'],
                                                                                  dataset_ident, 2)
                            mmd_loss_rgb, _ = mix_rbf_mmd2(rgb_features_1, rgb_features_2,
                                                           gammas=[(2.0 ** gamma) * 9.7 for gamma in
                                                                   np.arange(-8.0, 8.0, 2.0 ** 0.5)])
                            self.summaries.append(tf.summary.scalar('MMD RGB', mmd_loss_rgb))
                            mmd_loss += mmd_loss_rgb

                        if self.flow:
                            flow_features_1, flow_features_2 = tf.dynamic_partition(self.model['features_flow'],
                                                                                    dataset_ident, 2)
                            mmd_loss_flow, _ = mix_rbf_mmd2(flow_features_1, flow_features_2,
                                                            gammas=[(2.0 ** gamma) * 9.7 for gamma in
                                                                    np.arange(-8.0, 8.0, 2.0 ** 0.5)])

                            self.summaries.append(tf.summary.scalar('MMD Flow', mmd_loss_flow))
                            mmd_loss += mmd_loss_flow

                # Only keep inputs with corresponding modalities for the classification task
                dataset_ident_masked = tf.boolean_mask(dataset_ident, self.placeholders['synch'])
                labels = tf.boolean_mask(labels, self.placeholders['synch'])
                source_labels, target_labels = tf.dynamic_partition(labels, dataset_ident_masked, 2)

                if self.rgb and (not self.joint_classifier):
                    logits_rgb = tf.boolean_mask(self.model['logits_rgb'], self.placeholders['synch'])
                    dataset1_logits_rgb, dataset2_logits_rgb = tf.dynamic_partition(logits_rgb, dataset_ident_masked, 2)
                    source_logits_rgb = dataset1_logits_rgb[:, :self.num_labels]
                    target_logits_rgb = dataset2_logits_rgb[:, :self.num_labels]
                if self.flow and (not self.joint_classifier):
                    logits_flow = tf.boolean_mask(self.model['logits_flow'], self.placeholders['synch'])
                    dataset1_logits_flow, dataset2_logits_flow = tf.dynamic_partition(logits_flow, dataset_ident_masked, 2)
                    source_logits_flow = dataset1_logits_flow[:, :self.num_labels]
                    target_logits_flow = dataset2_logits_flow[:, :self.num_labels]
                if self.rgb and self.flow and self.joint_classifier:
                    logits_fusion = tf.boolean_mask(self.model['logits_fusion'], self.placeholders['synch'])
                    dataset1_logits_fusion, dataset2_logits_fusion = tf.dynamic_partition(logits_fusion, dataset_ident_masked, 2)
                    source_logits_fusion = dataset1_logits_fusion[:, :self.num_labels]
                    target_logits_fusion = dataset2_logits_fusion[:, :self.num_labels]

                if self.rgb and (not self.joint_classifier) and self.aux_classifier:
                    dataset1_logits_rgb_aux, dataset2_logits_rgb_aux = tf.dynamic_partition(self.model['logits_aux_rgb'], dataset_ident, 2)
                    source_logits_rgb_aux = dataset1_logits_rgb_aux[:, :self.num_labels]
                    target_logits_rgb_aux = dataset2_logits_rgb_aux[:, :self.num_labels]
                if self.flow and (not self.joint_classifier) and self.aux_classifier:
                    dataset1_logits_flow_aux, dataset2_logits_flow_aux = tf.dynamic_partition(self.model['logits_aux_flow'], dataset_ident, 2)
                    source_logits_flow_aux = dataset1_logits_flow_aux[:, :self.num_labels]
                    target_logits_flow_aux= dataset2_logits_flow_aux[:, :self.num_labels]
                if self.rgb and self.flow and self.joint_classifier and self.aux_classifier:
                    dataset1_logits_fusion_aux, dataset2_logits_fusion_aux = tf.dynamic_partition(self.model['logits_aux_fusion'], dataset_ident, 2)
                    source_logits_fusion_aux = dataset1_logits_fusion_aux[:, :self.num_labels]
                    target_logits_fusion_aux = dataset2_logits_fusion_aux[:, :self.num_labels]


                with tf.name_scope("Maximum_Classifer_Discrepancy_Loss"):
                    classifier_discrepancy = tf.constant(0.0)
                    if self.rgb and (not self.joint_classifier) and self.MaxClassDiscrepany:
                        classifier_discrepancy_rgb = - tf.reduce_mean(tf.abs(
                            tf.nn.softmax(target_logits_rgb[:, :self.num_labels]) - tf.nn.softmax(
                                target_logits_rgb_aux[:, :self.num_labels])))
                        self.summaries.append(tf.summary.scalar('MCD rgb', classifier_discrepancy_rgb))
                        classifier_discrepancy += classifier_discrepancy_rgb
                    if self.flow and (not self.joint_classifier) and self.MaxClassDiscrepany:
                        classifier_discrepancy_flow = - tf.reduce_mean(tf.abs(
                            tf.nn.softmax(target_logits_flow[:, :self.num_labels]) - tf.nn.softmax(
                                target_logits_flow_aux[:, :self.num_labels])))
                        self.summaries.append(tf.summary.scalar('MCD flow', classifier_discrepancy_flow))
                        classifier_discrepancy += classifier_discrepancy_flow
                    if self.rgb and self.flow and self.joint_classifier and self.MaxClassDiscrepany:
                        classifier_discrepancy_fusion = - tf.reduce_mean(tf.abs(
                            tf.nn.softmax(target_logits_fusion[:, :self.num_labels]) - tf.nn.softmax(
                                target_logits_fusion_aux[:, :self.num_labels])))
                        self.summaries.append(tf.summary.scalar('MCD fusion', classifier_discrepancy_fusion))
                        classifier_discrepancy += classifier_discrepancy_fusion

                with tf.name_scope('Softmax_loss'):
                    softmax_loss = tf.constant(0.0)
                    if self.rgb and (not self.joint_classifier):
                        softmax_loss_rgb = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(labels=source_labels, logits=source_logits_rgb))
                        softmax_loss += softmax_loss_rgb
                        # If entropy loss enabled (and adaptating) use target entropy.
                        self.summaries.append(tf.summary.scalar('SoftmaxLoss RGB', softmax_loss_rgb))

                    if self.flow and (not self.joint_classifier):
                        softmax_loss_flow = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(labels=source_labels, logits=source_logits_flow))
                        # If entropy loss enabled (and adaptating) use target entropy.
                        softmax_loss += softmax_loss_flow
                        self.summaries.append(tf.summary.scalar('SoftmaxLoss Flow', softmax_loss_flow))

                    if self.rgb and self.flow and self.joint_classifier:
                        softmax_loss_fusion = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(labels=source_labels, logits=source_logits_fusion))
                        # If entropy loss enabled (and adaptating) use target entropy.
                        softmax_loss += softmax_loss_fusion
                        self.summaries.append(tf.summary.scalar('SoftmaxLoss Fusion', softmax_loss_fusion))

                with tf.name_scope('Aux_Softmax_loss'):
                    softmax_loss_aux = tf.constant(0.0)
                    if self.rgb and (not self.joint_classifier) and self.aux_classifier:
                        softmax_loss_aux_rgb = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(labels=source_labels, logits=source_logits_rgb_aux))
                        softmax_loss_aux += softmax_loss_aux_rgb
                        # If entropy loss enabled (and adaptating) use target entropy.
                        self.summaries.append(tf.summary.scalar('SoftmaxLoss Aux RGB', softmax_loss_aux_rgb))

                    if self.flow and (not self.joint_classifier) and self.aux_classifier:
                        softmax_loss_aux_flow = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(labels=source_labels, logits=source_logits_flow_aux))
                        # If entropy loss enabled (and adaptating) use target entropy.
                        softmax_loss_aux += softmax_loss_aux_flow
                        self.summaries.append(tf.summary.scalar('SoftmaxLoss Aux Flow', softmax_loss_aux_flow))

                    if self.rgb and self.flow and self.joint_classifier and self.aux_classifier:
                        softmax_loss_aux_fusion = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(labels=source_labels, logits=source_logits_fusion_aux))
                        # If entropy loss enabled (and adaptating) use target entropy.
                        softmax_loss_aux += softmax_loss_aux_fusion
                        self.summaries.append(tf.summary.scalar('SoftmaxLoss Aux Fusion', softmax_loss_aux_fusion))


                # Sum losses
                with tf.name_scope('total_loss'):
                    total_loss = (self.softmax_lambda) * (softmax_loss + softmax_loss_aux) +\
                                 0.2 * mmd_loss + domain_loss + 0.2 * classifier_discrepancy + \
                                 self.selfsupervised_lambda *synch_loss
                    self.summaries.append(tf.summary.scalar('TotalLoss', total_loss))

            self.__losses = {'total_loss': total_loss, 'domain_loss': domain_loss, 'mmd_loss': mmd_loss}
        return self.__losses

    def __training_ops(self):
        """ Produces a dictionary of update and gradient computation operators.
        Gradient accumulators are given per update to allow for batch sizes that cannot fit into memory."""
        def get_training_ops(losses, weights, accum_name, optimiser):
            grads = self.opt.compute_gradients(losses/ self.steps_per_update, colocate_gradients_with_ops=True,
                                               aggregation_method=tf.AggregationMethod.ADD_N,
                                               var_list=weights)
            with tf.variable_scope(accum_name):
                accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in
                              weights]
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
            accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads)]
            apply_gradient_op = optimiser.apply_gradients(
                [(accum_vars[i] , gv[1]) for i, gv in enumerate(grads)],
                global_step=self.placeholders['global_step'])
            return zero_ops, accum_ops, apply_gradient_op

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # Compute Gradients using back propagation
        reglosses = tf.add_n([tf.nn.l2_loss(v) for v in variables]) * 1e-7

        zero_ops, accum_ops, apply_gradient_op = get_training_ops(self.losses['total_loss'] + reglosses, variables,
                                                                  "accumulators", self.opt)

        # Group Batch Norm Statistics Update Ops
        batch_norm_updates_op = tf.group(*self.model['batch_norm_updates'])
        # Apply all update ops

        grouped_train_ops = tf.group(apply_gradient_op)
        grouped_update_ops = tf.group(accum_ops, batch_norm_updates_op)
        grouped_zero_ops = zero_ops
        self.__train_op = {'train_op': grouped_train_ops, 'update_op': grouped_update_ops,
                           'zero_op': grouped_zero_ops}

    @property
    def train_op(self):
        """ Update op for feature extractor and classifier weights."""
        if not self.__train_op:
            self.__training_ops()
        return self.__train_op['train_op']

    @property
    def accum_grads(self):
        """ compute gradients for feature extractor and classifier weights."""
        if not self.__train_op:
            self.__training_ops()
        return self.__train_op['update_op']

    @property
    def zero_grads(self):
        """ zero gradient accumulators """
        if not self.__train_op:
            self.__training_ops()
        return self.__train_op['zero_op']

    @property
    def prediction(self):
        """ Produces softmaxed logits and performance metrics """
        if not self.__predictions:
            with tf.name_scope('acc'):
                if self.rgb and self.flow and (not self.joint_classifier):
                    logits_rgb = tf.nn.softmax(self.model['logits_rgb'])
                    logits_flow = tf.nn.softmax(self.model['logits_flow'])
                    logits_presoftmax = tf.reduce_mean([logits_rgb, logits_flow], axis=0)
                    softmax_logits = tf.nn.softmax(logits_presoftmax)
                    predicted = tf.argmax(logits_presoftmax, 1)

                elif self.rgb and (not self.joint_classifier):
                    logits_presoftmax = self.model['logits_rgb']
                    softmax_logits = tf.nn.softmax(logits_presoftmax)
                    logits_rgb = None
                    logits_flow = None
                    predicted = tf.argmax(logits_presoftmax, 1)

                elif self.flow and (not self.joint_classifier):
                    logits_presoftmax = self.model['logits_flow']
                    softmax_logits = tf.nn.softmax(logits_presoftmax)
                    predicted = tf.argmax(logits_presoftmax, 1)
                    logits_rgb = None
                    logits_flow = None

                elif self.rgb and self.flow and self.joint_classifier:
                    logits_presoftmax = self.model['logits_fusion']
                    softmax_logits = tf.nn.softmax(logits_presoftmax)
                    predicted = tf.argmax(logits_presoftmax, 1)
                    logits_rgb = None
                    logits_flow = None

                else:
                    raise Exception("Invalid choice")

                if self.predict_synch and self.rgb and self.flow:
                    synch_labels = tf.cast(self.placeholders['synch'], dtype=tf.int64)
                    synch_pred = tf.argmax(self.model['synch_logits'], 1)
                    correct_synch_prediction = tf.cast(tf.equal(synch_pred, synch_labels), tf.float32)
                    synch_accuracy = tf.reduce_mean(correct_synch_prediction)
                    self.summaries.append(tf.summary.scalar('Synch Accuracy', synch_accuracy))
                else:
                    synch_accuracy = -1


                true = tf.argmax(self.placeholders['labels'], 1)
                correct_prediction = tf.cast(tf.equal(predicted, true), tf.float32)
                accuracy = tf.reduce_mean(correct_prediction)
                self.summaries.append(tf.summary.scalar('Accuracy', accuracy))
                if self.domain_loss:
                    domain_logits = tf.nn.softmax(self.model['domain_logits_rgb'] + self.model['domain_logits_flow'])
                    predicted_domain = tf.cast(tf.argmax(domain_logits, 1), tf.int32)
                    correct_domain = tf.cast(tf.equal(self.placeholders['dataset_ident'], predicted_domain), tf.float32)
                else:
                    domain_logits = tf.zeros([0, 2], tf.float32)
                    correct_domain = tf.zeros(tf.shape(self.placeholders['dataset_ident']), tf.float32)
                self.__predictions = {'pred': predicted, 'logits': softmax_logits, 'true': true,
                                      'correct': correct_prediction, 'domain_logits': domain_logits,
                                      'accuracy': accuracy, 'correct_domain': correct_domain, 'logits_rgb': logits_rgb,
                                      'logits_flow': logits_flow, 'logits_pre_softmax': logits_presoftmax,
                                      'synch_accuracy': synch_accuracy}
        return self.__predictions

    def get_summaries(self):
        """ Return TensorFlow summaries """
        return self.summaries
