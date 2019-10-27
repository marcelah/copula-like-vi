# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:56:47 2018

@author: Marcel
"""

#very similiar to
# https://github.com/tensorflow/probability/blob/master/discussion/higher_level_modeling_api_demo.ipynb
#but allows access to the log prior and log likelihood and not just the log joint

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from tensorflow_probability import edward2 as ed
import copy
import collections


class MetaModel(type):

  def __call__(cls, *args, **kwargs):
    print('load latents')
    obj = type.__call__(cls, *args, **kwargs)
    obj._load_observed()
    obj._load_unobserved()
    return obj


class BaseModel(object):
  __metaclass__ = MetaModel

  def _load_unobserved(self):
    unobserved_fun = self._unobserved_vars()
    self.unobserved = unobserved_fun()
    
  def _load_observed(self):
    self.observed = copy.copy(vars(self))

  def _unobserved_vars(self):

    def unobserved_fn(*args, **kwargs):
      unobserved_vars = collections.OrderedDict()

      def interceptor(f, *args, **kwargs):
        name = kwargs.get("name")
        rv = f(*args, **kwargs)
        if name not in self.observed:
          unobserved_vars[name] = rv.shape
        return rv

      with ed.interception(interceptor):
        self.__call__()
      return unobserved_vars

    return unobserved_fn
  


  def target_log_prob_fn(self, *args, **kwargs):
    """Unnormalized target density as a function of unobserved states."""

    def log_joint_fn(*args, **kwargs):
      states = dict(zip(self.unobserved, args))
      states.update(self.observed)
      log_probs = []

      def interceptor(f, *args, **kwargs):
        name = kwargs.get("name")
        for name, value in states.items():
          if kwargs.get("name") == name:
            kwargs["value"] = value
        rv = f(*args, **kwargs)
        log_prob = tf.reduce_sum(rv.distribution.log_prob(rv.value))
        log_probs.append(log_prob)
        return rv

      with ed.interception(interceptor):
        self.__call__()
      log_prob = sum(log_probs)
      return log_prob

    return log_joint_fn



  def get_latents_fn(self, states={}, *args, **kwargs):
    """Get the joint prob given arbitrary values for vars"""

    def latents_fn(*args, **kwargs):

      def interceptor(f, *args, **kwargs):
        name = kwargs.get("name")
        for name, value in states.items():
          if kwargs.get("name") == name:
            kwargs["value"] = value
        rv = f(*args, **kwargs)
        return rv

      with ed.interception(interceptor):
        return self.__call__()

    return latents_fn



  def get_log_prior_fn(self, states={}, *args, **kwargs):
    """Get the log prob for latent vars with values in states"""

    def log_prior_fn(*args, **kwargs):
      log_probs = collections.OrderedDict()#[]

      def interceptor(f, *args, **kwargs):
        name_ = kwargs.get("name")
        for name, value in states.items():
          if name_ == name:
            kwargs["value"] = value
        rv = f(*args, **kwargs)

        if name_ in states.keys():
          log_prob = tf.reduce_sum(rv.distribution.log_prob(rv.value))
          log_probs[name_]=log_prob
        
        return rv
      
      with ed.interception(interceptor):
        self.__call__()
      log_prob=log_probs
      return log_prob

    return log_prior_fn

  def get_llh_fn(self, observation_states={}, 
                     latent_states={}, *args, **kwargs):
    """Get the log prob for observation values in observation_states
    and fixed latent_states"""

    def llh_fn(*args, **kwargs):
      log_probs = collections.OrderedDict()#[]

      def interceptor(f, *args, **kwargs):
        name_ = kwargs.get("name")
        for name, value in latent_states.items():
          if name_ == name:
            kwargs["value"] = value
        for name, value in observation_states.items():
          if name_ == name:
            kwargs["value"] = value
        rv = f(*args, **kwargs)
        if name_ in observation_states.keys():
          log_prob = tf.reduce_sum(rv.distribution.log_prob(rv.value))
          log_probs[name_]=log_prob
        return rv

      with ed.interception(interceptor):
        self.__call__()

      log_prob=log_probs
      return log_prob

    return llh_fn


  def __call__(self):
    return self.call()

  def call(self, *args, **kwargs):
    raise NotImplementedError
