# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:12:01 2018

@author: Marcel
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import collections
import time as time
import os
from datetime import datetime

def killnan(X):
    return tf.where(tf.is_nan(X), tf.zeros_like(X), X)

def killinf(X):
    return tf.where(tf.logical_or(tf.is_inf(X),tf.is_inf(-X)),
                    tf.zeros_like(X), X)

def killnaninf(X,v):
    if X is None:
      return tf.zeros_like(v)
    else:
      return tf.where(tf.logical_or(tf.logical_or(
        tf.is_nan(X),tf.is_inf(X)),tf.is_inf(-X)),tf.zeros_like(X), X)


def init_vi(model, batch_size=None, log_dir=None):
  
  model.log_dir=log_dir
  model.batch_size=min(batch_size,model.data_size)
  model.variational_samples=collections.OrderedDict()
  model.variational_entropy=collections.OrderedDict()
  model.marginal_variational_entropy=collections.OrderedDict()

  #sample from variational distribution
  for name, shape in model.unobserved.items():
    model.variational_samples[name]=model.variational_distribution[name].distribution.sample(model.mc_samples)
  
  #compute cross-entropy estimate between posterior and prior
  model.prior_log_prob=model.get_log_prior_fn(
          states=dict(model.variational_samples))()
  model.prior_log_probs=tf.add_n([v[1] for v in model.prior_log_prob.items()])

  model.prior_log_probs=tf.Print(model.prior_log_probs,
                                       [model.prior_log_probs],
                                       message="model.prior_log_probs: ")

  model.prior_log_probs=tf.reduce_sum(model.prior_log_probs)


  #compute reconstruction error
  model.reconstruction_error=model.get_llh_fn(
          latent_states=dict(model.variational_samples),
          observation_states=dict(model.observed))()
  model.reconstruction_errors=tf.add_n(
      [v[1] for v in model.reconstruction_error.items()])

  model.reconstruction_errors=tf.Print(model.reconstruction_errors,
                                       [model.reconstruction_errors],
                                      message="model.reconstruction_errors: ")

  #compute entropy estimate of variational distribution
  for name, shape in model.unobserved.items():
    entropy=-model.variational_distribution[name].distribution.log_prob(
        model.variational_samples[name])
    model.variational_entropy[name]=tf.reduce_sum(entropy)
  model.variational_entropies=tf.add_n(
      [v[1] for v in model.variational_entropy.items()])
  model.variational_entropies=tf.reduce_sum(model.variational_entropies)
  model.variational_entropies=tf.Print(model.variational_entropies,
                                       [model.variational_entropies],
                                       message="variational_entropies: ")
  
  #entropy of the marginals only
  for name, shape in model.unobserved.items():
    model.marginal_variational_entropy[name]=tf.reduce_sum(-model.marginal_model[name].distribution.log_prob(
        model.variational_samples[name]))
  model.marginal_variational_entropies=tf.add_n(
      [v[1] for v in model.marginal_variational_entropy.items()])
  model.marginal_variational_entropies=tf.reduce_sum(
      model.marginal_variational_entropies)
  
  if model.build_copula_dist:
    model.copula_variational_entropy=collections.OrderedDict()
    model.hypercube_samples=collections.OrderedDict()
    #entropy of the copula-like part only
    for name, shape in model.unobserved.items():
      model.hypercube_samples[name]=model.marginal_model[name].distribution.cdf(model.variational_samples[name])
                 
      model.copula_variational_entropy[name]=tf.reduce_sum(
          -model.copula_dist[name].distribution.log_prob(
          model.hypercube_samples[name]))
    model.copula_variational_entropies=tf.add_n(
        [v[1] for v in model.copula_variational_entropy.items()])
    model.copula_variational_entropies=tf.reduce_sum(model.copula_variational_entropies)
   
  #compute variational lower bound 
  model.elbo=model.reconstruction_errors*model.data_size/model.batch_size\
    +model.variational_entropies+model.prior_log_probs
  model.elbo=model.elbo/tf.cast(model.mc_samples,tf.float32)

  model.elbo=tf.Print(model.elbo,[model.elbo],message="model.elbo: ",
                      )

  model.posterior_prior_kl=-model.variational_entropies-model.prior_log_probs
  model.posterior_prior_kl=tf.Print(model.posterior_prior_kl,
                                    [model.posterior_prior_kl],
                                    message="posterior_prior_kl: ")
 
 
def init_vi_grads(model,learning_rate=.0001):  
  
  #get gradients of elbo
  model.var_list_copula= tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="copula")
  model.var_list_marginals= tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="marginal")
  model.var_list=model.var_list_copula+model.var_list_marginals  

  start_time = time.time()

  if True:
     model.grads_elbo=tf.gradients(
          -model.elbo, model.var_list)
     if model.kill_nan_grads:
      model.grads_elbo=[killnaninf(grad1,var1) for (grad1,var1) in zip(
          model.grads_elbo,model.var_list)]

  print("--- %s seconds for grads  ---" % (time.time() - start_time))

  model.grads_and_vars_elbo=list(zip(
    model.grads_elbo,model.var_list))

  print(model.grads_and_vars_elbo)
  
  model.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                      name="adam")
  model.train_step=model.optimizer.apply_gradients(
      model.grads_and_vars_elbo)


  if model.log_dir is not None:
    #log to tensorboard
    logdir = os.path.expanduser(model.log_dir)
    logdir = os.path.join(
    logdir, datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S"))

    model._summary_key = tf.get_default_graph().unique_name("summaries")

    for var in model.var_list:
      # replace colons which are an invalid character
      var_name = var.name.replace(':', '/')
      # Log all scalars.
      if len(var.shape) == 0:
        tf.summary.scalar("parameter/{}".format(var_name),
                          var, collections=[model._summary_key])
      elif len(var.shape) == 1 and var.shape[0] == 1:
        tf.summary.scalar("parameter/{}".format(var_name),
                          var[0], collections=[model._summary_key])
      else:
        # If var is multi-dimensional, log a histogram of its values.
        tf.summary.histogram("parameter/{}".format(var_name),
                             var, collections=[model._summary_key])

    tf.summary.scalar("VI/elbo",model.elbo, collections=[model._summary_key])
    tf.summary.scalar("VI/reconstruction_errors",model.reconstruction_errors,
                      collections=[model._summary_key])
    tf.summary.scalar("VI/variational_entropies",model.variational_entropies, 
                      collections=[model._summary_key])
    tf.summary.scalar("VI/prior_log_probs",model.prior_log_probs, 
                      collections=[model._summary_key])
    tf.summary.scalar("VI/posterior_prior_kl",model.posterior_prior_kl, 
                      collections=[model._summary_key])
    tf.summary.scalar("VI/scaled_reconstruction_errors",
                      model.reconstruction_errors*model.data_size/model.batch_size,
                      collections=[model._summary_key])

    for grad, var in model.grads_and_vars_elbo:
      print(grad)
      print(var)
      if grad is not None:
        # replace colons which are an invalid character
        tf.summary.histogram("gradient/" +
                             var.name.replace(':', '/'),
                             grad, collections=[model._summary_key])
        tf.summary.scalar("gradient_norm/" +
                          var.name.replace(':', '/'),
                          tf.norm(grad), collections=[model._summary_key])


    for name, shape in model.unobserved.items():
      
      var_name = name.replace(':', '/')
      tf.summary.histogram("var_samples_/{}".format(var_name),
                             killinf(model.variational_samples[name]), 
                             collections=[model._summary_key])
      if model.build_copula_dist:

        tf.summary.histogram("copula_samples_/{}".format(var_name),
                             killinf(model.hypercube_samples[name]), 
                             collections=[model._summary_key])

    model.summarize = tf.summary.merge_all(key=model._summary_key)

    model.train_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    

def optimize_elbo(model,sess,num_epochs):   

  t=[]
  for i in range(num_epochs):

    if len(model.var_list_copula)>0:
      _,elbo_copula_=sess.run([model.train_copula,model.elbo])            

    _,elbo_marginal_=sess.run([model.train_marginals,model.elbo])
    t.append(elbo_marginal_)
    if i % 500 == 0:
      model.var_list_marginals_,model.var_list_copula_=sess.run([
          model.var_list_marginals,model.var_list_copula])
      print('elbo')
      print(elbo_marginal_)
      print(model.var_list_marginals_)
      print(model.var_list_copula_)
      
      posterior_prior_kl_,elbo_,variational_entropies_,\
      reconstruction_errors_,prior_log_probs_,variational_samples_,\
      marginal_variational_entropies_=sess.run(
          [model.posterior_prior_kl,model.elbo,model.variational_entropies,
           model.reconstruction_errors*model.data_size/model.batch_size,model.prior_log_probs,
           model.variational_samples,model.marginal_variational_entropies,
           ])
      print('elbo')
      print(elbo_)
      print('posterior_prior_kl_')
      print(posterior_prior_kl_)          
      print('entropy')
      print(variational_entropies_)
      print('rec err')
      print(reconstruction_errors_)
      print('prior_log_probs')
      print(prior_log_probs_)
      print('variational_samples')
      print(variational_samples_)
      print('marginal_variational_entropies')
      print(marginal_variational_entropies_)
      #print('copula entropies')
      #print(copula_variational_entropies_)
      if model.build_copula_dist:
       hypercube_samples_=sess.run(model.hypercube_samples)
        
       print('hypercube_samples')
       print(hypercube_samples_)
 
  return t
    
    

def test_vi(model,sess,mc_samples):  
    #compute elbo on test set averaged over 
    #monte carlo samples from variational distribution
    elbo_test_estimate=0.
    for s in range(mc_samples):
      elbo_test_=sess.run(model.elbo)
      elbo_test_estimate+=elbo_test_
    elbo_test_estimate=elbo_test_estimate/mc_samples/model.mc_samples  
    print('elbo test')
    print(elbo_test_estimate)
    return elbo_test_estimate
