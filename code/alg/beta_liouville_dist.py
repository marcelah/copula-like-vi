import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from tensorflow_probability.python.internal import dtype_util
#from tensorflow_probability.python.internal import reparameterization
from tensorflow.python.ops import control_flow_ops
from tensorflow_probability.python.internal import distribution_util as util
from tensorflow.python.ops import array_ops

class BetaLiouvilleCopulaLike(tfd.Distribution):
  """Generating dirichlet and beta distributions follows very
      closly the respective tensorflow code
  """

  def __init__(self,
               concentration,
               alpha,
               beta,  
               validate_args=False,
               allow_nan_stats=True,
               name="BetaLiouvilleCopulaLike"):
    """Initializes BetaLiouvilleCopulaLike distributions.
    Args:
      concentration: Positive floating-point `Tensor` indicating mean number
        of class occurrences; aka "alpha" for the Dirichlet distribution.
      alpha: Positive floating-point `Tensor` indicating mean
        number of successes of the beta distribution.
      beta: Positive floating-point `Tensor` indicating mean
        number of failures of the beta distribution. Otherwise has same semantics as
        `alpha`.       
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name, values=[concentration,alpha,beta]) as name:
      dtype = dtype_util.common_dtype([concentration, 
                                       alpha,
                                       beta],
                                      tf.float32)
      self._concentration = self._maybe_assert_valid_concentration(
          tf.convert_to_tensor(
              concentration,
              name="concentration",
              dtype=dtype),
          validate_args)
      self._total_concentration_dirichlet = tf.reduce_sum(self._concentration, -1)
      self._alpha = self._maybe_assert_valid_concentration_beta(
          tf.convert_to_tensor(
              alpha, name="alpha", dtype=dtype),
          validate_args)
      self._beta = self._maybe_assert_valid_concentration_beta(
          tf.convert_to_tensor(
              beta, name="beta", dtype=dtype),
          validate_args) 
      self._total_concentration_beta = self._alpha + self._beta  
    super(BetaLiouvilleCopulaLike, self).__init__(
        dtype=self._concentration.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type="FULLY_REPARAMETERIZED",#reparameterization.FULLY_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[self._concentration,
                       self._total_concentration_dirichlet,
                       self._alpha,
                       self._beta,
                       self._total_concentration_beta],
        name=name)

  @property
  def concentration(self):
    """Concentration parameter for dirichlet distribution."""
    return self._concentration

  @property
  def total_concentration_dirichlet(self):
    """Sum of last dim of concentration parameter for dirichlets."""
    return self._total_concentration_dirichlet

  def _batch_shape_tensor(self):
    return  array_ops.shape(self.total_concentration_dirichlet)

  def _batch_shape(self):
    return self.total_concentration_dirichlet.get_shape()#self.total_concentration_dirichlet.get_shape()
  
  def _event_shape_tensor(self):
    return array_ops.shape(self.concentration)[-1:]  
  
  def _event_shape(self):
    return self.concentration.get_shape().with_rank_at_least(1)[-1:]
  
  @property
  def alpha(self):
    """Concentration parameter associated with a `1` outcome."""
    return self._alpha

  @property
  def beta(self):
    """Concentration parameter associated with a `0` outcome."""
    return self._beta

  @property
  def total_concentration_beta(self):
    """Sum of concentration parameters for beta distribution."""
    return self._total_concentration_beta  
  
  
  def _sample_n(self, n, seed=None):
    #sample the dirichlet distribution using gamma random variables
    gamma_sample_dirichlet = tf.random_gamma(
        shape=[n],
        alpha=self.concentration,
        dtype=self.dtype,
        seed=seed)
    dirichlet_sample=gamma_sample_dirichlet / tf.reduce_sum(
        gamma_sample_dirichlet, -1, keepdims=True)    
    #sample the beta distribution using gamma random variables
    expanded_alpha = tf.ones_like(
        self.total_concentration_beta, dtype=self.dtype) * self.alpha
    expanded_beta = tf.ones_like(
        self.total_concentration_beta, dtype=self.dtype) * self.beta
    gamma1_sample = tf.random_gamma(
        shape=[n],
        alpha=expanded_alpha,
        dtype=self.dtype,
        seed=seed)
    gamma2_sample = tf.random_gamma(
        shape=[n],
        alpha=expanded_beta,
        dtype=self.dtype,
        seed=util.gen_new_seed(seed, "beta"))
    beta_sample = gamma1_sample / (gamma1_sample + gamma2_sample)
    

    expanded_beta_sample=tf.tile(tf.expand_dims(beta_sample,-1),
                                   multiples=[1,dirichlet_sample.shape[1]]) 
    
    
    max_dirichlet_sample=tf.reduce_max(dirichlet_sample,axis=-1)
    expanded_max_dirichlet_sample=tf.tile(tf.expand_dims(max_dirichlet_sample,-1),
                                   multiples=[1,dirichlet_sample.shape[1]]) 
   
    sample= tf.multiply(expanded_beta_sample,
                       tf.div(dirichlet_sample,expanded_max_dirichlet_sample))

    
    return sample
  
  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()
  
  def _log_unnormalized_prob(self, x):
        
    val=-self.total_concentration_dirichlet*tf.log(tf.reduce_sum(x,axis=-1))\
      +(self.beta-1.)*tf.log(1.-tf.reduce_max(x,axis=-1))\
      +self.alpha *tf.log(tf.reduce_max(x,axis=-1))\
      +tf.reduce_sum((self.concentration-1.) *tf.log(x),-1)
    
    return val

  def _log_normalization(self):
    return tf.lbeta(self.concentration)+(tf.lgamma(self.alpha)
            + tf.lgamma(self.beta)
            - tf.lgamma(self.total_concentration_beta))
  
  

  def _maybe_assert_valid_concentration(self, concentration, validate_args):
    """Checks the validity of the concentration parameter."""
    if not validate_args:
      return concentration
    return control_flow_ops.with_dependencies([
        tf.assert_positive(
            concentration,
            message="Concentration parameter must be positive."),
        tf.assert_rank_at_least(
            concentration, 1,
            message="Concentration parameter must have >=1 dimensions."),
        tf.assert_less(
            1, tf.shape(concentration)[-1],
            message="Concentration parameter must have event_size >= 2."),
    ], concentration)

  def _maybe_assert_valid_concentration_beta(self, concentration, validate_args):
    """Checks the validity of a concentration parameter."""
    if not validate_args:
      return concentration
    return control_flow_ops.with_dependencies([
        tf.assert_positive(
            concentration,
            message="Concentration parameter must be positive."),
    ], concentration)