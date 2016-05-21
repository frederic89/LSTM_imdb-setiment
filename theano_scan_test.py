# coding=utf-8
__author__ = 'gyq-mac'

import theano
import theano.tensor as T
import numpy as np

#例子地址：http://deeplearning.net/software/theano/tutorial/loop.html

print("\nScan Example: Computing trace of X")
# define tensor variables
X = T.vector("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")
U = T.matrix("U")
Y = T.matrix("Y")
V = T.matrix("V")
P = T.matrix("P")

results, updates = theano.scan(lambda y, p, x_tm1: T.tanh(T.dot(x_tm1, W) + T.dot(y, U) + T.dot(p, V)),
          sequences=[Y, P[::-1]], outputs_info=[dict(initial=X, taps=[-1])])
          #outputs_info=[dict(initial=X, taps=[-1])]等价于 outputs_info=[X]，因为
          #对于 outputs_info 参数中的序列来说，taps默认值为 [-1]，表示时间 t 传入 t-1 时刻的序列值，只能为负值。
          # 如果taps值为 None，表示这个输出结果不会作为参数传入 fn 中。
compute_seq = theano.function(inputs=[X, W, Y, U, P, V], outputs=results)

# test values
x = np.zeros((2), dtype=theano.config.floatX)
x[1] = 1
w = np.ones((2, 2), dtype=theano.config.floatX)
y = np.ones((5, 2), dtype=theano.config.floatX)
y[0, :] = -3
u = np.ones((2, 2), dtype=theano.config.floatX)
p = np.ones((5, 2), dtype=theano.config.floatX)
p[0, :] = 3
v = np.ones((2, 2), dtype=theano.config.floatX)

print(compute_seq(x, w, y, u, p, v))
print("\n=============================")
print("\nScan Example: Computing trace of X")

floatX = "float32"

# define tensor variable
X_ = T.matrix("X_")
results, updates = theano.scan(lambda i, j, t_f: T.cast(X_[i, j] + t_f, floatX),
                  sequences=[T.arange(X_.shape[0]), T.arange(X_.shape[1])], #sequences序列中的[a,b] a,b同时在序列中步进，导致求得的是方阵的迹
                  outputs_info=np.asarray(0., dtype=floatX))
result = results[-1]
compute_trace = theano.function(inputs=[X_], outputs=result)

# test value
x = np.eye(5, dtype=theano.config.floatX)
# x[0] = np.arange(5, dtype=theano.config.floatX)
print(compute_trace(x))