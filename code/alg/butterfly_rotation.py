# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:10:44 2019

@author: Marcel
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

class ButterflyRotation(tfb.Bijector):

    def __init__(self, theta, skip=0, validate_args=False, name="ButterflyRotation"):
      self._theta=theta
      self._skip=skip
      self._rotation_matrices=self._get_rotation_matrix()
      super(ButterflyRotation, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          name=name)
  
    def _get_rotation_matrix(self):
      print('start build Q')
      
      def get_s_indices(lst, le):
        i=0
        v=[]
        while (i<len(lst)):
          v=v+lst[i:i+le]
          i+=2*le
        return v

      d_=self._theta.get_shape().as_list()[-1]      
      k=np.ceil(np.log(d_)/np.log(2))
      d=int(np.power(2,k))
      Qs=[]

      k=int(k)
      for l in list(range(0,k)[::(self._skip+1)]):
        lp=np.power(2,l)
        s_plus_list=[(i-lp,i) for i in range(lp,d)]
        s_minus_list=[(i,i-lp) for i in range(lp,d)]
        s_plus_indices=get_s_indices(s_plus_list,lp)
        s_minus_indices=get_s_indices(s_minus_list,lp)
        del_c_indices=[i[0] for i in s_plus_indices if (i[0]>=d_ or i[1]>=d_) ]
        del_c_minus_indices=[i[0] for i in s_minus_indices if (i[0]>=d_ or i[1]>=d_) ]
        c_indices=[(i,i) for i in range(d_) if i not in del_c_indices]
        s_plus_indices=[i for i in s_plus_indices if i[0] not in del_c_indices]
        s_minus_indices=[i for i in s_minus_indices if i[0] not in del_c_minus_indices]
        del_s=len([i for i in del_c_indices if i<d_])

        c_value_idx=np.repeat(range(lp-1,d_)[::2*lp],2*(lp))
        s_value_idx=np.repeat(range(lp-1,d_)[::2*lp],lp)
        #remove idx values if necessary
        idx=[i for i in range(d_) if i not in del_c_indices]
        c_value_idx=c_value_idx[idx]
        idx=[i-lp for i in range(lp,d_+lp) if i-lp not in del_c_indices]
        s_value_idx=c_value_idx[::2]#s_value_idx[idx]

        v=tf.gather(tf.cos(self._theta),c_value_idx)
        values_diag=tf.concat([tf.ones([del_s]),v],axis=0)
        c_indices=[(i,i) for i in del_c_indices if i<d_]+c_indices

        values_off_diag1=tf.gather(-tf.sin(self._theta),s_value_idx)
        values_off_diag2=tf.gather(tf.sin(self._theta),s_value_idx) 
        indices=c_indices+s_plus_indices+s_minus_indices
        values=tf.concat([values_diag,values_off_diag1,values_off_diag2],axis=0)
        Q=tf.SparseTensor(indices=tf.cast(indices,tf.int64), 
                       values=tf.cast(values,tf.float32), dense_shape=[d_, d_])
        Q=tf.sparse_reorder(Q)
        print('finished build Q'+str(l))
        Qs.append(Q)     
        #Q_dense=tf.sparse.to_dense(Q,validate_indices=False)
        #Qs_dense.append(Q_dense)
      
      return Qs
            
    def _forward(self, x):
      #print(tf.sparse_tensor_to_dense(self._rotation_matrix1,validate_indices=False))
      v_t=x
      for l in range(len(self._rotation_matrices)):          
        v_t=tf.sparse_tensor_dense_matmul(self._rotation_matrices[l],v_t,
                                           adjoint_a=False,adjoint_b=True)
        v_t=tf.transpose(v_t)
     
      return v_t
      

    def _inverse(self, y):
      #return self._forward(y)

      v_t = y
      for l in range(len(self._rotation_matrices)):
        #v_t = tf.sparse_tensor_dense_matmul(self._rotation_matrices[-l], v_t,
        #                                  adjoint_a=True, adjoint_b=True)
        v_t = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(self._rotation_matrices[-l])
                                            , v_t,
                                  adjoint_a=False, adjoint_b=True)


        v_t = tf.transpose(v_t)

      return v_t


    def _inverse_log_det_jacobian(self, y):
      return tf.zeros_like(y)
