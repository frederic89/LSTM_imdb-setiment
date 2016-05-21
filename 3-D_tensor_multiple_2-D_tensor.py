# coding=utf-8
__author__ = 'gyq-mac'

import theano
import theano.tensor as T
import numpy as  np

x1=T.itensor3('x1')
x2=T.imatrix('x2')

f=T.dot(x1,x2)
r=theano.function([x1,x2],f,
                  allow_input_downcast=True)  #试着把allow_input_downcast改为None

a1=np.array([[[1,  2],
              [3,  4],
              [5,  6]],
             [[7,  8],
              [9, 10],
              [11,12]]],dtype=theano.config.floatX)

a2=np.array([[1,2,1],
             [2,1,1]],dtype=theano.config.floatX)

print(a1)
print('===============')
print(a2)
print('===============')
print(r(a1,a2))