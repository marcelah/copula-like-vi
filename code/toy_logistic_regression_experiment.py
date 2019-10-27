# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 20:10:39 2018

@author: Marcel
"""

# Dependency imports
import os
from absl import flags
import matplotlib
matplotlib.use("Agg")
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import seaborn as sns


import sys
sys.path.extend(['alg/'])
from logistic_regression_model import LogisticRegressionModel
from variational_inference import init_vi,init_vi_grads,\
  optimize_elbo,test_vi
import time as time


#copula_model must be 'RotatedBetaLiouville', 'BetaLiouville', 'LowRankGaussian' or 'Independent'
flags.DEFINE_string("copula_model",
                   default='BetaLiouville',
                   help="copula model.")
flags.DEFINE_float("learning_rate",
                   default=0.005,
                   help="Initial learning rate.")
flags.DEFINE_integer("epochs",
                     default=100001,
                     help="Number of training epochs to run.")
flags.DEFINE_string("data_set",
                    default="toy_logistic_regression",
                    help="data set used.")
flags.DEFINE_float("rank",
                   default=1,
                   help="Rank of gaussian covariance if chosen.")


flags.DEFINE_string(
    "model_dir",
    default=os.path.join("PATH",
                         'elbo_results_and_plots'),
    help="Directory to put the model's fit.")



FLAGS = flags.FLAGS



def get_toy_logistic_data(num_examples=60):
  """Generates toy logistic regression data as in VADAM paper.
  """
  num_examples=num_examples//2
  mu_1=np.array([1,5])  
  mu_2=np.array([-5,1])
  scale_1=np.ones([2])
  scale_2=1.1*np.ones([2])
  x1=np.random.normal(loc=mu_1,scale=scale_1,size=(num_examples,2))
  x2=np.random.normal(loc=mu_2,scale=scale_2,size=(num_examples,2))
  y1=np.ones([num_examples])
  y2=0*np.ones([num_examples])
  x=np.concatenate([x1,x2])
  y=np.concatenate([y1,y2])
  return x,y



def plot_weight_posteriors(model,sess, fname, mc_samples=1000):
  """Save a PNG plot with histograms of weight means and stddevs.
  """
  
  for name, q_marginal in model.marginal_model.items(): 
    variational_samples_mc=sess.run(
        model.variational_distribution[name].distribution.sample(mc_samples))

    fig = figure.Figure(figsize=(6, 3))
    canvas = backend_agg.FigureCanvasAgg(fig)
  
    ax = fig.add_subplot(1, 2, 1)
    sns.distplot(np.ma.masked_invalid(variational_samples_mc).mean(axis=0), ax=ax)
    ax.set_title("weight means")
    #ax.set_xlim([-1.5, 1.5])
    #ax.set_ylim([0, 4.])
    ax.legend()
  
    ax = fig.add_subplot(1, 2, 2)
    sns.distplot(np.ma.masked_invalid(variational_samples_mc).std(axis=0), ax=ax)
    ax.set_title("weight stddevs")
    #ax.set_xlim([0, 1.])
    #ax.set_ylim([0, 25.])
  
    fig.tight_layout()
    canvas.print_figure(fname+'.png', format="png")
    print("saved {}".format(fname))



    
def plot_posteriors_params(variational_params, fname):
  """Save a PNG plot with histograms of variational parameters.
  """
  d=len((variational_params))
  for i in range(d):
   try: 
    fig = figure.Figure(figsize=(6, 3))
    canvas = backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    vp=variational_params[i]
    sns.distplot(vp, ax=ax)
    #ax.set_xlim([-1.5, 1.5])
    #ax.set_ylim([0, 4.])
    ax.legend()
    
    fig.tight_layout()
    canvas.print_figure(fname+str(i)+'_.png', format="png")
    print("saved {}".format(fname))
    np.savetxt(fname+str(i)+'_.npy',np.array([vp]))
   except:
    pass
     
  

def plot_hmc_joint_samples(model, sess, hmc_samples,fig_name):
  for name, q_marginal in model.marginal_model.items(): 
    samples_mc_=hmc_samples[name][::100,0,:]
    dim=samples_mc_.shape[1]
    for i in range(dim):
      for j in range(i,dim):
        try:
          if i==0 and j==0:
            cor=np.corrcoef(np.transpose(samples_mc_))   
            sns_plot=sns.heatmap(cor,vmin=-1,vmax=1,center=0,cmap="RdBu_r")
            figure = sns_plot.get_figure()  
            figure.savefig(fig_name+'_'+'correlation'+name+'hmc.png')
            time.sleep(1)
            plt.clf();plt.close('all')

          sns_plot=(sns.jointplot((samples_mc_[:,i]),
                  (samples_mc_[:,j]),kind="kde")).set_axis_labels(str(i), str(j))
          sns_plot.ylim(0, 25)
          sns_plot.xlim(0, 15)
          sns_plot.savefig(fig_name+'_'+str(i)+'_'+str(j)+name+'hmc.png')
          time.sleep(1)
          plt.clf();plt.close('all')
        except:
          pass
    

def plot_joint(model, sess, mc_samples,fig_name):
  for name, q_marginal in model.marginal_model.items(): 
    variational_samples_mc_=sess.run(
        model.variational_distribution[name].distribution.sample(mc_samples))
    variational_samples_mc=variational_samples_mc_[:,0:1000]
    dim=variational_samples_mc.shape[1]
    for i in range(min(dim-1,5)):
      for j in range(i,min(dim-1,5)):
        try:
          if i==0 and j==0:
            cor=np.corrcoef(np.transpose(variational_samples_mc))   
            sns_plot=sns.heatmap(cor,vmin=-1,vmax=1,center=0,cmap="RdBu_r")
            figure = sns_plot.get_figure()  
            figure.savefig(fig_name+'_'+'correlation'+name+'.png')
            time.sleep(1)
            plt.clf();plt.close('all')

          sns_plot=(sns.jointplot((variational_samples_mc[:,i]),
                  (variational_samples_mc[:,j]),kind="kde")).set_axis_labels(str(i), str(j))
          sns_plot.ylim(0, 25)
          sns_plot.xlim(0, 15)
          sns_plot.savefig(fig_name+'_'+str(i)+'_'+str(j)+name+'.png')
          time.sleep(1)
          plt.clf();plt.close('all')
        except:
          pass
    variational_samples_mc=variational_samples_mc_[:,-5:]
    dim=variational_samples_mc.shape[1]
    for i in range(dim):
      for j in range(i,dim):
        try:
          if i==0 and j==0:
            cor=np.corrcoef(np.transpose(variational_samples_mc))   
            sns_plot=sns.heatmap(cor,vmin=-1,vmax=1,center=0,cmap="RdBu_r")
            figure = sns_plot.get_figure()  
            figure.savefig(fig_name+'_'+'correlation_'+name+'.png')
            time.sleep(1)
            plt.clf();plt.close('all')

          sns_plot=(sns.jointplot((variational_samples_mc[:,i]),
                  (variational_samples_mc[:,j]),kind="kde")).set_axis_labels(str(i), str(j))
          print(fig_name+'_'+str(i)+'_'+str(j)+name+'.png')
          sns_plot.savefig(fig_name+'_'+str(i)+'_'+str(j)+name+'_.png')
          time.sleep(1)
          plt.clf();plt.close('all')
        except:
          pass


def hmc_sample(sess,model, inputs, outputs,
           num_results=3000,
           num_burnin_steps=1000,
           step_size=.001,
           num_leapfrog_steps=3,
           numpy=True):
  
  initial_state = []
  for name, shape in model.unobserved.items():
       initial_state.append(tf.expand_dims(0.0 * tf.ones(shape[1:], 
                                      name="init_{}".format(name)),0))


  states, kernel_results = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=initial_state,
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=model.target_log_prob_fn(),
          step_size=step_size,
          num_leapfrog_steps=num_leapfrog_steps))

  states, is_accepted_ = sess.run([states, kernel_results.is_accepted],
                                  feed_dict={
                                      model.inputs: inputs,
                         model.outputs: outputs[:1,:],
                         model.mc_samples: 1
                         })

  accepted = np.sum(is_accepted_)
  print("Acceptance rate: {}".format(accepted / num_results))
  return dict(zip(model.unobserved.keys(), states))



def main(argv):
  del argv  # unused
  
  copula_family=FLAGS.copula_model
  no_train_samples=10
  no_eval_samples=1000
  print(copula_family)
  prior_precision=.01
  num_examples=60

  if copula_family=='LowRankGaussian':
    FLAGS.copula_model = FLAGS.copula_model+str(FLAGS.rank)

  batch_x,batch_y=  get_toy_logistic_data(num_examples=num_examples)
  batch_y_eval=np.tile(np.expand_dims(batch_y,0),[no_eval_samples,1])
  batch_y=np.tile(np.expand_dims(batch_y,0),[no_train_samples,1])

  tf.reset_default_graph()
  with tf.Graph().as_default():
    
    model=LogisticRegressionModel(dim_covariates=2,data_size=num_examples, 
                                  prior_precision=prior_precision,
                                  copula_family=FLAGS.copula_model)
    

    init_vi(model,batch_x.shape[0],
            log_dir=FLAGS.model_dir+'/'+FLAGS.data_set)
    init_vi_grads(model,learning_rate=FLAGS.learning_rate)


    
    init = tf.global_variables_initializer()
  
    
    with tf.Session() as sess:
      sess.run(init)

      for epoch in range(FLAGS.epochs):
        # Run optimization
        _ = sess.run(
              model.train_step, 
              feed_dict={model.inputs: batch_x,
                         model.outputs: batch_y,
                         model.mc_samples: no_train_samples})      

        if epoch%5000==0 :
          #save results
          summary = sess.run(model.summarize, feed_dict={
                                model.inputs:batch_x,
                                model.outputs: batch_y,
                                model.mc_samples: no_train_samples})
          model.train_writer.add_summary(summary, epoch)
          
          elbo_,posterior_prior_kl_,reconstruction_errors_,mc_samples_, variational_params_ = sess.run(
              [model.elbo, model.posterior_prior_kl,model.reconstruction_errors, model.mc_samples,
               model.var_list], 
              feed_dict={model.inputs: batch_x,
                         model.outputs: batch_y_eval,
                         model.mc_samples: no_eval_samples})   
          
          print(variational_params_)
          
          elbo__=(posterior_prior_kl_-reconstruction_errors_)/no_eval_samples

          print('elbo')
          print(elbo__)
          print(elbo_)

  
          fname=os.path.join(FLAGS.model_dir,
                             FLAGS.data_set+"/plots/"+FLAGS.copula_model+"_epoch{:05d}_".format(epoch))
          dir_=os.path.join(FLAGS.model_dir,
                             FLAGS.data_set+"/plots/")
          if not os.path.exists(dir_):
                os.makedirs(dir_)
          np.savetxt(fname+'elbo.txt',np.array([elbo_]))
          plot_weight_posteriors(model,sess,fname, mc_samples=500)
          plot_posteriors_params(variational_params_, fname)
          plot_joint(model,sess,500,fname)


          #save variational samples
          variational_samples_= sess.run(
                    model.variational_samples, 
                    feed_dict={model.inputs: batch_x,
                               model.outputs: batch_y,
                               model.mc_samples:2000})   
          variational_samples_=variational_samples_['w']   
          np.save(fname+'w_samples.npy',variational_samples_)


      #compare with hmc sampling
      hmc_samples=hmc_sample(sess,model, batch_x, batch_y,
           num_results=100000,
           num_burnin_steps=10000,
           step_size=.05,
           num_leapfrog_steps=20,
           numpy=True)


      thinned_hmc_samples_=hmc_samples['w'][::100,0,:]
      np.save(fname+'w_samples_hmc.npy',thinned_hmc_samples_)
      plot_hmc_joint_samples(model,sess,hmc_samples,fname)


      #hmc trace plots
      fig, axes = plt.subplots(thinned_hmc_samples_.shape[1], 2, sharex='col', sharey='col')
      for j in range(thinned_hmc_samples_.shape[1]):        
        axes[j][0].plot(thinned_hmc_samples_[:,j])
        sns.kdeplot(thinned_hmc_samples_[:,j], ax=axes[j][1], shade=True)
      plt.savefig(fname+'w_hmc_marginals.png')




    
if __name__ == "__main__":
  tf.app.run()
  
 