# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:01:11 2018

@author: Marcel
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from beta_liouville_dist import BetaLiouvilleCopulaLike

       
class VectorNormalBijector(tfb.Bijector):
  """Bijector that encodes a vector of Normal distributions.
  
  """
  def __init__(self, loc, scale):
    self.normal_dist = tfd.Normal(loc=loc, scale=scale)
    super(VectorNormalBijector, self).__init__(
        forward_min_event_ndims=1,
        validate_args=False,
        name="NormalBijector")
    
  def _forward(self, y):
    return self.normal_dist.quantile(y)
  
  def _inverse(self, x):
    return self.normal_dist.cdf(x)
  
  def _inverse_log_det_jacobian(self, x):
    res1=tf.reduce_sum(self.normal_dist.log_prob(x),-1)
    return res1


class VectorisedIndependentCopula(tfd.TransformedDistribution):
  def __init__(self, d, marginal_bijectors,
               name="VectorisedIndependentCopula"):
    
    ind = tfd.Independent(distribution=tfd.Uniform(
        low=tf.zeros(d),high=tf.ones(d)),
                          reinterpreted_batch_ndims=1)    
    super(VectorisedIndependentCopula, self).__init__(
        distribution=ind,
        bijector=(marginal_bijectors),
        validate_args=False,
        name=name)     


class VectorisedBetaLiouvilleCopula(tfd.TransformedDistribution):
  def __init__(self, concentration, alpha, beta, 
               marginal_bijectors,name="VectorisedBetaLiouvilleCopula"):
    super(VectorisedBetaLiouvilleCopula, self).__init__(
        distribution=BetaLiouvilleCopulaLike(concentration=concentration,
                                             alpha=alpha,
                                             beta=beta),
        bijector=(marginal_bijectors),
        validate_args=False,
        name=name)     


class WeightedBetaLiouvilleCopulaLike(tfd.TransformedDistribution):
  
  def __init__(self, concentration, alpha, beta, 
               weights,name="WeightedBetaLiouvilleCopulaLike"):
    super(WeightedBetaLiouvilleCopulaLike, self).__init__(
        distribution=BetaLiouvilleCopulaLike(concentration=concentration,
                                             alpha=alpha,
                                             beta=beta),
        bijector=tfb.Affine(shift=tf.ones([])-weights,
                        scale_diag=2*weights-tf.ones([])),
        validate_args=False,
        name=name)   


class VectorisedWeightedBetaLiouvilleCopula(tfd.TransformedDistribution):
  def __init__(self, concentration, alpha, beta, weights,
               marginal_bijectors,name="VectorisedWeightedBetaLiouvilleCopula"):
    super(VectorisedWeightedBetaLiouvilleCopula, self).__init__(
        distribution=WeightedBetaLiouvilleCopulaLike(concentration=concentration,
                                             alpha=alpha,
                                             beta=beta,
                                             weights=weights),
        bijector=(marginal_bijectors),
        validate_args=False,
        name=name)     
     


class VectorisedCopula(tfd.TransformedDistribution):
  def __init__(self, copula_dist,
               marginal_bijectors,name="VectorisedCopula"):
    super(VectorisedCopula, self).__init__(
        distribution=copula_dist,
        bijector=marginal_bijectors,
        validate_args=False,
        name=name)     
