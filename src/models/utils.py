import tensorflow.contrib.slim as slim
import tensorflow as tf

""" Functions to produce TensorFlow savers to save and load MM-SADA models
"""


def _get_variables_to_restore_load(to_ignore,flow):
    to_ignore.append('domain_accumulators')
    to_ignore.append('accum_accumulators') 
    to_ignore.append('accumulators')
    scope_to_ignore = to_ignore
    if flow:
        scope_to_ignore.append("RGB")
        scope_to_ignore.append("Joint")
    else:
        scope_to_ignore.append("Flow")
        scope_to_ignore.append("Joint")
    variables = slim.get_variables_to_restore(exclude=scope_to_ignore)
    keyword_filter = to_ignore
    return [x for x in variables if not any(word in x.name for word in keyword_filter)]


def read_joint(mode=""):

    if mode == "restore":
        to_ignore = ["Adam", "adam", "Momentum",
                     "beta1_power", "beta2_power", "global_step", "Domain_Classifier"]#, "synch_test"]
    elif mode == "continue":
        to_ignore = []
    else:
        raise Exception("Unknown mode for read_joint")
    to_ignore.append('domain_accumulators')
    to_ignore.append('accum_accumulators') 
    to_ignore.append('accumulators')
    to_ignore.append("Flow")
    to_ignore.append("RGB")
    variables = slim.get_variables_to_restore(exclude=to_ignore)
    variables_restore =  [x for x in variables if not any(word in x.name for word in to_ignore)]
    rgb_variable_map = {}

    for variable in variables_restore:
        name = variable.name.replace(':0', '')
        rgb_variable_map[name] = variable


    variable_loader = tf.train.Saver(var_list = rgb_variable_map, reshape=True,max_to_keep=20)

    return variable_loader


def read_i3d_checkpoint(mode="",flow=False,aux_logits=False):
    if mode == "pretrain":
        to_ignore = ["inception_i3d/Logits","Adam","adam","Momentum",
                     "beta1_power","beta2_power","global_step","Domain_Classifier", "arrow_test"]#, "synch_test"]
    elif mode == "restore":
        to_ignore = ["Adam","adam","Momentum",
                     "beta1_power","beta2_power","global_step","Domain_Classifier"]#, "synch_test"]
    elif mode == "continue":
        to_ignore = []
    else:
        raise Exception("Unkown mode for read_i3d_checkpoint")


    variables_restore = _get_variables_to_restore_load(to_ignore,flow)

    rgb_variable_map = {}
    #variables_restore = [x for x in variables_restore_to_train if not any(word in x.name for word in keywordFilter)]
    for variable in variables_restore:
        name = variable.name.replace(':0','')
        rgb_variable_map[name] = variable

    
    variable_loader = tf.train.Saver(var_list = rgb_variable_map, reshape=True,max_to_keep=20)
    return variable_loader


def restore_base(sess, saver, checkpoint_path, model_to_restore, restore_mode="model"):
    if restore_mode == "continue":
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver['continue'].restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("Cannot find a model to continue training")
        start_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    elif restore_mode == "model":
        saver['model'].restore(sess, model_to_restore)
        start_step = 0
    elif restore_mode == "pretrain" or restore_mode == "pretrain_from_synch":
        saver['pretrain'].restore(sess, model_to_restore)
        start_step = 0
    else:
        raise Exception("A valid restore Mode must be set --restore_mode==[continue,model,pretrain]")
    return start_step


def restore_joint(sess, saver, checkpoint_path, model_to_restore, restore_mode="model"):
    if restore_mode == "continue":
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver['continue'].restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("Cannot find a model to continue training")
        start_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    elif restore_mode == "model" or restore_mode == "pretrain_from_synch":
        saver['model'].restore(sess, model_to_restore)
        start_step = 0
    elif restore_mode == "pretrain":
        start_step = 0
    else:
        raise Exception("A valid restore Mode must be set --restore_mode==[continue,model,pretrain]")
    return start_step


def init_savers_base(flow=False):
    pretrain_loader = read_i3d_checkpoint(mode="pretrain", flow=flow)
    model_loader = read_i3d_checkpoint(mode="restore", flow=flow)
    savesave = read_i3d_checkpoint(mode="continue", flow=flow)
    return pretrain_loader, model_loader, savesave