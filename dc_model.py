# -*- coding: utf-8 -*-
""" The d-band center model
"""
import pinn.networks
import tensorflow as tf
import numpy as np

from pinn.utils import atomic_dress
from pinn.utils import pi_named

default_params = {
    # Loss function definition
    'use_l2': False,         # L2 regularization
    # Optimizer related
    'learning_rate': 3e-4,   # Learning rate
    'use_norm_clip': True,   # see tf.clip_by_global_norm
    'norm_clip': 0.01,       # see tf.clip_by_global_norm
    'use_decay': True,       # Exponential decay
    'decay_step': 10000,     # every ? steps
    'decay_rate': 0.999,     # scale by ?
}


def dc_model(params, **kwargs):
    """Shortcut for generating d_band model from paramters

    When creating the model, a params.yml is automatically created 
    in model_dir containing network_params and model_params.

    The potential model can also be initiated with the model_dir, 
    in that case, params.yml must locate in model_dir from which
    all parameters are loaded

    Args:
        params(str or dict): parameter dictionary or the model_dir
        **kwargs: additional options for the estimator, e.g. config
    """
    import os
    import yaml
    from tensorflow.python.lib.io.file_io import FileIO
    from datetime import datetime

    if isinstance(params, str):
        model_dir = params
        assert tf.io.gfile.exists('{}/params.yml'.format(model_dir)),\
            "Parameters files not found."
        with FileIO(os.path.join(model_dir, 'params.yml'), 'r') as f:
            params = yaml.load(f, Loader=yaml.Loader)
    else:
        model_dir = params['model_dir']
        yaml.Dumper.ignore_aliases = lambda *args: True
        to_write = yaml.dump(params)
        params_path = os.path.join(model_dir, 'params.yml')
        if not tf.io.gfile.IsDirectory(model_dir):
            tf.io.gfile.MakeDirs(model_dir)
        if tf.io.gfile.exists(params_path):
            original = FileIO(params_path, 'r').read()
            if original != to_write:
                os.rename(params_path, params_path+'.' +
                          datetime.now().strftime('%y%m%d%H%M'))
        FileIO(params_path, 'w').write(to_write)

    model = tf.estimator.Estimator(
        model_fn=_dc_model_fn, params=params,
        model_dir=model_dir, **kwargs)
    return model


def _dc_model_fn(features, labels, mode, params):
    """Model function for neural network potentials"""
    if isinstance(params['network'], str):
        network_fn = getattr(pinn.networks, params['network'])
    else:
        network_fn = params['network']

    network_params = params['network_params']
    model_params = default_params.copy()
    model_params.update(params['model_params'])
    pred = network_fn(features, **network_params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        n_trainable = np.sum([np.prod(v.shape)
                              for v in tf.trainable_variables()])
        print("Total number of trainable variables: {}".format(n_trainable))

        loss, metrics = _get_loss(features, pred, model_params)
        _make_train_summary(metrics)
        train_op = _get_train_op(loss,  model_params)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss, metrics = _get_loss(features, pred, model_params)
        metrics = _make_eval_metrics(metrics)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'dcenter': tf.expand_dims(pred,0),
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions)

    
@pi_named('LOSSES')
def _get_loss(features, pred, model_params):
    metrics = {}  # Not editting features here for safety, use a separate dict

    dc_pred = pred
    dc_data = features['dc_data']

    dc_error = dc_pred - dc_data
    metrics['dc_data'] = dc_data
    metrics['dc_pred'] = dc_pred
    metrics['dc_error'] = dc_error

    dc_loss = dc_error**2
    metrics['dc_loss'] = dc_loss
    
    tot_loss = tf.reduce_mean(dc_loss)

    if model_params['use_l2']:
        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([
            tf.nn.l2_loss(v) for v in tvars if
            ('bias' not in v.name and 'E_OUT' not in v.name)])
        metrics['l2_loss'] = l2_loss * model_params['l2_loss_multiplier']
        tot_loss += l2_loss

    metrics['tot_loss'] = tot_loss
    
    return tot_loss, metrics


@pi_named('METRICS')
def _make_eval_metrics(metrics):
    eval_metrics = {
        'METRICS/DC_MAE': tf.metrics.mean_absolute_error(
            metrics['dc_data'], metrics['dc_pred']),
        'METRICS/DC_RMSE': tf.metrics.root_mean_squared_error(
            metrics['dc_data'], metrics['dc_pred']),
        'METRICS/DC_LOSS': tf.metrics.mean(metrics['dc_loss']),
        'METRICS/TOT_LOSS': tf.metrics.mean(metrics['tot_loss'])
    }

    if 'l2_loss' in metrics:
        eval_metrics['METRICS/L2_LOSS'] = tf.metrics.mean(metrics['l2_loss'])
    return eval_metrics


@pi_named('METRICS')
def _make_train_summary(metrics):
    tf.summary.scalar('DC_RMSE', tf.sqrt(tf.reduce_mean(metrics['dc_error']**2)))
    tf.summary.scalar('DC_MAE', tf.reduce_mean(tf.abs(metrics['dc_error'])))
    tf.summary.scalar('DC_LOSS', tf.reduce_mean(metrics['dc_loss']))
    tf.summary.scalar('TOT_LOSS', metrics['tot_loss'])
    tf.summary.histogram('DC_DATA', metrics['dc_data'])
    tf.summary.histogram('DC_PRED', metrics['dc_pred'])
    tf.summary.histogram('DC_ERROR', metrics['dc_error'])

    if 'l2_loss' in metrics:
        tf.summary.scalar('L2_LOSS', metrics['l2_loss'])


@pi_named('TRAIN_OP')
def _get_train_op(loss, model_params):
    # Get the optimizer
    global_step = tf.train.get_global_step()
    learning_rate = model_params['learning_rate']
    if model_params['use_decay']:
        learning_rate = tf.train.exponential_decay(
            learning_rate, global_step,
            model_params['decay_step'], model_params['decay_rate'],
            staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Get the gradients
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    if model_params['use_norm_clip']:
        grads, _ = tf.clip_by_global_norm(grads, model_params['norm_clip'])
    return optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
