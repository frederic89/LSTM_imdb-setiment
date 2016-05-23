# coding=utf-8
'''
Build a tweet sentiment analyzer
'''

from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb

datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    n是输入集的总个数
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n): # 终点不是倍数
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)  #返回一个二元序列(序号，batch索引号)，用于运行。每个批次号，包含句子的索引。
    # batch的索引号是全集唯一的


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):  # dropout防过拟合trick
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options): # dict作为输入，options转化为params
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    拼合model_options参数、LSTM 初始化参数和全局params，返回到外部params（命名正好也是params）
    """
    params = OrderedDict()  # OrderedDict的Key会按照插入的顺序排列，不是Key本身排序（题外： 可以实现一个FIFO（先进先出）的dict，当容量超出限制时，先删除最早添加的Key）
    # embedding
    randn = numpy.random.rand(options['n_words'], options['dim_proj']) # 生成2维（n_words、dim_proj）uniform分布矩阵
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,  #get_layer(options['encoder'])[0]即LSTM 初始化函数的名称，保存在名称管理器中
                                              params,
                                              prefix=options['encoder'])
    # classifier
    #初始化Softmax输出层参数（Emb_Dim，2）
    # ydim = #{0,1} = 2 确定是几类分类问题，imdb测试集是2分类问题
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)
    #这个U不是lstm_U,而是分类器的U，b也不是lstm_b

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    '''
    这个函数意义就是一键将Numpy标准的params全部转为Theano.shared标准。
    替代大量的Theano.shared(......)
    启用tparams作为Model的正式params，而原来的params废弃。
    '''
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    '''
    LSTM的初始化值很特殊，先用[0,1]随机数生成矩阵，然后对随机矩阵进行SVD奇异值分解。取正交基矩阵来初始化，即 ortho_weight
    '''
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """

    # 4个正交基矩阵沿着横轴拼合, axis=1表示横轴方向
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    # U.shape = (dim,4dim) U.ndim=2 U是2维矩阵
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params  # 补充了lstm_W、lstm_U、lstm_b的3params后，返回了整体params字典


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    '''
    LSTM计算的核心,首先得注意参数state_below，这是个3D矩阵，[n_Step，BatchSize，Emb_Dim] [句子数，[单词batch数，词向量维度] ]
    '''
    nsteps = state_below.shape[0] # 最高维
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]  # 取出单词batch数
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):

        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        # x_是形参，下面的state_below是实参
        # _step中的四个形参： x_是state_below降为二维矩阵形成的序列，m_是mask降为1维“行”向量的序列
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])  # 每次新的h_与lstm_U矩阵相乘，使得f、o、c均不再为零，其中f、o是中间变量
        preact += x_  # 2维矩阵序列(x_) + 2维矩阵 = 2维矩阵序列

        # 每一个preact矩阵序列 的4块并列子矩阵 进行切片分块运算
        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))  # 拿出了preact中的input GATE相关项做sigmoid激活
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))  # forget GATE
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))  # output GATE
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))  # cell

        c = f * c_ + i * c
        # [:,None] 表示“行数组”升一维变为“列向量”
        c = m_[:, None] * c + (1. - m_)[:, None] * c_ #c_代表初始时刻或上一时刻

        # 每个Step里，h结果是一个2D矩阵，[BatchSize，Emb_Dim]
        h = o * tensor.tanh(c)  # 相当于octave中的.*
        h = m_[:, None] * h + (1. - m_)[:, None] * h_  # h_是h的上一时刻或toutputs_info中的初始时刻的值

        return h, c  # 输出值对应着outputs_info中的元素
        # scan函数最终的return时，2维矩阵序列还原为三维矩阵，向量序列还原为2维矩阵，即scan函数的输出结果会增加1-D

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])  # 3维矩阵 乘 2维矩阵仍是3维矩阵，无须每一个Step做一次Wx+b，而是把所有Step的Wx一次性预计算好了

    dim_proj = options['dim_proj']
    # scan函数的一旦sequence不为空，就进入序列循环模式
    # theano.scan中的sequences中的张量会被降低一维成为在迭代函数中使用，原最高维的维数作为sequences的迭代的次数,out
    # 在scan函数的Sequence里，每步循环，都会降解n_Step维，得到一个Emb矩阵，作为输入X_
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],  # mask对应m_，state_below对应x_，迭代次数由sequences的序列个数决定
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples, # n_samples是句子中单词batch个数(即LSTM时序上输入的个数)
                                                           dim_proj), # 这个张量是outputs[-1]时刻所初始化的值，在第一次loop之后（outputs[0]之后）将会被覆盖，覆盖后此处对应着_step中的h_(h的前一时刻)
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj) # 第1次loop后此张量被覆盖，此处对应c_(c的前一时刻)
                                              ],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]  # rval[0]是h  rval[1]是c  它们都是tensor.shared类型


# ff: Feed Forward (normal neural net), only useful to put after lstm
# before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}  #函数名称管理器，一种罗嗦的解耦


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    # 该部分是联合LSTM和Softmax，构成完整Theano.function的重要部分。定义了多种对矩阵的变化方式和代数函数。连接了初始化参数和lstm_layer核心运算函数,train_lstm主函数。
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')  # 词矩阵x是文字方向在竖式伸展，并列拼成一组batch
    n_timesteps = x.shape[0]  # 行数即最长句子里单词的个数(即LSTM时序输入的个数)，（矩阵有多少列）
    n_samples = x.shape[1]  # n_samples是一组batch里并行的句子个数，
    # 对应语言模型的序列学习算法，每个Step就相当于取一个句子的一个词。
    # x每次scan取的一排词，称为examples，数量等于batch_size。

    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    # tparams['Wemb']是纵向存储词表的单词、横向存储“词向量”的theano矩阵
    # x.flatten()后是个一维numpy.array序列
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,  # 第3维，决定了有几个子矩阵（有几个单词序列就有3维矩阵的几层）
                                                n_samples,  # 子矩阵的行数（=batch的个数）
                                                options['dim_proj']])  # 词向量的维度或并行LSTM Block的个数
                                                # reshape成三维矩阵，三维矩阵的每一层代表一组batch的词向量
    # emb对应lstm_layer中的state_below

    #      get_layer(options['encoder'])[1]即函数名称：lstm_layer，返回值是LSTM的h [n_timestep, BatchSize，Emb_Dim]
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)  # 括号内是 3维矩阵.*3维列向量 = 3维矩阵，sum参数中“axis=0”代表在“最高维”轴的方向求和，维度降为了2维单层矩阵（求和“象征”着所有单词特征“压扁”为句子特征）
        proj = proj / mask.sum(axis=0)[:, None]  # 2维单层矩阵./2维单层列向量 = 2维单层矩阵 mean pooling
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    # proj的维度是(BatchSize,Emb_Dim)，U是（Emb_Dim,2）,点乘得到(BatchSize,2)，加上偏置b
    # 2是classifier中输出分类标签的个数（y_dim）
    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    # 将上面所有表达式整合在theano函数中，输入值是词矩阵x和Mask矩阵 输出值是pred矩阵
    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob') # 定义计算推测概率的函数
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')  # 定义计算推测分类的函数

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6
    # imdb分类中，pred是(BatchSize,2)  所以pred矩阵中的[tensor.arange(n_samples), y]位置是具体分类标签结果
    # cost结果
    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    # batch的索引号是全集唯一的， data是全集，而iterator是形成batch的抽象结果

    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err


def train_lstm(
        dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
        patience=10,  # Number of epoch to wait before early stop if no progress
        max_epochs=5000,  # The maximum number of epoch to run
        dispFreq=10,  # Display to stdout the training progress every N updates
        decay_c=0.,  # Weight decay for the classifier applied to the U weights.
        lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=10000,  # Vocabulary size
        optimizer=adadelta,
        # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        encoder='lstm',  # TODO: can be removed must be lstm.
        saveto='lstm_model.npz',  # The best model will be saved there
        validFreq=370,  # Compute the validation error after this number of update.
        saveFreq=1110,  # Save the parameters after every saveFreq updates
        maxlen=100,  # Sequence longer then this get ignored
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation/test set.
        dataset='imdb',  #上面也有名称管理的结构定义imdb

        # Parameter for extra option
        noise_std=0.,
        use_dropout=True,  # if False slightly faster, but worst test error
        # This frequently need a bigger model.
        reload_model=None,  # Path to a saved model we want to start from.
        test_size=-1,  # If >0, we keep only this number of test example.
):
    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    load_data, prepare_data = get_dataset(dataset)  # 链接到imdb.py中定义的函数

    print('Loading data')
    train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
                                   maxlen=maxlen)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])  #test[0]是输入数组（test_x），test[1]是情感标签(test_y)

    ydim = numpy.max(train[1]) + 1 #确定是几类分类问题，imdb是2值分类

    model_options['ydim'] = ydim  #补充options中缺失的ydim参数

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []  # 保存结果的数组
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)  # 返回一个二元序列(序号，batch索引号)，batch索引号即下文的train_index

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                # 重新定义x，y为列表；x是一组batch的句子，y则是该batch集的标签
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)   # 再次赋值x为词矩阵，并已经转为theano变量
                n_samples += x.shape[1]  # This swap the axis!

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid,
                                           kf_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                                valid_err <= numpy.array(history_errs)[:,
                                             0].min()):
                        best_p = unzip(tparams)
                        bad_counter = 0

                    print(('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err))

                    if (len(history_errs) > patience and
                                valid_err >= numpy.array(history_errs)[:-patience,
                                             0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print(('Training took %.1fs' % (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=100,
        test_size=500,
    )
