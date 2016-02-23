# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from collections import OrderedDict


class Optimizer(object):
    """
    Optimizer Base Object, initialises generic optimizer variables

    Parameters
    ----------

    model_obj: cnn-object
      Encapsulation of theano model (instead of giving X,Y etc. manually), all other arguments are
      retrieved from this object if they are ``None``. If an argument is not ``None`` it will override the
      value from the model
    X:         symbolic input variable
      Data
    Y:         symbolic output variable
      Target
    Y_aux:     symbolic output variable
      Auxiliary masks/weights/etc. for the loss, type: list!
    top_loss:  symbolic loss function:
      Requires (X, Y (,*Y_aux)) for compilation
    params:    list of shared variables
      List of parameter arrays against which the loss is optimised

    Returns
    -------
    Callable optimizer object: loss = Optimizer(X, Y (,*Y_aux)) performs one iteration

    """

    def __init__(self,
                 model_obj=None,
                 X=None,
                 Y=None,
                 Y_aux=[],
                 top_loss=None,
                 params=None):
        self.t_init = time.time()
        self.optimizer_params = None
        self.step = None
        if model_obj is None:
            assert X is not None and Y is not None and top_loss is not None and params is not None, "Missing arguments!"
            self.gradients = T.grad(top_loss,
                                    params,
                                    disconnected_inputs="warn")

        else:
            if X is None:
                self.X = model_obj._x
            else:
                self.X = X
            if Y is None:
                self.Y = model_obj._y
            else:
                self.Y = Y
            if Y_aux == []:
                self.Y_aux = model_obj._y_aux
            else:
                self.Y_aux = Y_aux
            if params is None:
                self.params = model_obj.params
            else:
                self.params = params

            if top_loss is None:  # no loss means, loss&grad must already be defined in the model
                self.gradients = model_obj._gradients
                self.top_loss = model_obj._loss

            else:
                self.gradients = T.grad(top_loss, params, disconnected_inputs="warn")

        assert len(self.params) > 0, "no params, call compileOutputFunctions() before calling compileOptimizer()!"

        if hasattr(model_obj, '_last_grads') and model_obj._last_grads != [] and model_obj._last_grads is not None:
            self.last_grads = model_obj._last_grads
        else:
            self.last_grads = []
            for i, p in enumerate(self.params):
                if p in self.params[:i]:
                    print "Detected shared param: param[%i]" % i
                else:
                    self.last_grads.append(theano.shared(np.zeros(p.get_value().shape, dtype='float32'),
                                                         name=p.name + str('_LG'), borrow=False))

        if hasattr(model_obj, 'global_weightdecay'):
            self.weightdecay = model_obj.global_weightdecay
        else:
            self.weightdecay = 0

        if hasattr(model_obj, '_loss_instance'):
            self.loss_instance = model_obj._loss_instance
        else:
            self.loss_instance = None

        if hasattr(model_obj, '_get_loss'):
            self.get_loss = model_obj._get_loss
        else:
            self._get_loss = theano.function([self.X, self.Y] + self.Y_aux, [self.top_loss, self.loss_instance])

        if model_obj is not None:
            model_obj._last_grads = self.last_grads  # share last grads in model
            model_obj.get_loss = self.get_loss
            self.model_obj = model_obj

    def updateOptimizerParams(self, optimizer_params):
        """
        Update the hyper-parameter dictionary
        """
        self.optimizer_params.update(optimizer_params)

    def get_loss(self, *args):
        """
        [data, labels(, *aux)] --> [loss, loss_instance]
        loss_instance is the loss per instance (e.g. batch-item or pixel)
        """
        loss, nloss_instance = self._get_loss(*args)
        return np.float32(loss), nloss_instance

    def __call__(self, *args):
        """
        Perform an update step

        [data, labels(, *aux)] --> [loss, loss_instance]
        loss_instance is the loss per instance (e.g. batch-item or pixel)
        """
        ret = list(self.step(*args))
        ret[0] = np.float32(ret[0])  # the scalar nll
        return ret

    def compileGradients(self):
        """
        Compile and return a function that returns list of gradients
        """
        print "Compiling getGradients"
        getGradients = theano.function(
            [self.X, self.Y] + self.Y_aux,
            self.gradients,
            on_unused_input='warn')
        return getGradients

##############################################################################################################
#######################################       Stochastic Gradient Descent     ################################
##############################################################################################################


class compileSGD(Optimizer):
    """
    Stochastic Gradient Descent
    """

    def __init__(self,
                 optimizer_params,
                 model_obj=None,
                 X=None,
                 Y=None,
                 Y_aux=[],
                 top_loss=None,
                 params=None):
        print "Compiling SGD"
        super(compileSGD, self).__init__(model_obj, X, Y, Y_aux, top_loss, params)

        self.LR = optimizer_params['LR']
        self.momentum = optimizer_params['momentum']

        grad_updates = []
        param_updates = []
        for param_i, grad_i, last_grad_i in zip(self.params, self.gradients, self.last_grads):
            new_grad_i = grad_i + last_grad_i * self.momentum
            grad_updates.append((last_grad_i, new_grad_i))  # use this if you want to use the gradient magnitude
            # For no weight decay weightdecay is just 0
            param_updates.append((param_i, param_i - (self.LR) * new_grad_i - self.LR * self.weightdecay * param_i))

        assert len(grad_updates) == len(param_updates), str(len(grad_updates)) + " != " + str(len(param_updates))
        # This updates last_grads with the current grad and returns the loss before any parameter change
        self.step = theano.function(
            [self.X, self.Y] + self.Y_aux,
            [self.top_loss, self.loss_instance],
            updates=param_updates + grad_updates,
            on_unused_input='warn')

        print " Compiling done  - in %.3f s!" % (time.time() - self.t_init)

##############################################################################################################
#######################################                Ada Grad               ################################
##############################################################################################################


class compileAdam(Optimizer):
    """
    Stochastic Gradient Descent
    """

    def __init__(self,
                 optimizer_params,
                 model_obj=None,
                 X=None,
                 Y=None,
                 Y_aux=[],
                 top_loss=None,
                 params=None):
        print "Compiling Adam"
        super(compileAdam, self).__init__(model_obj, X, Y, Y_aux, top_loss, params)

        self.LR = optimizer_params['LR']
        beta1 = optimizer_params['beta1']
        beta2 = optimizer_params['beta2']
        epsilon = optimizer_params['epsilon']

        updates = OrderedDict()
        t_prev = theano.shared(np.float32(0.0))
        t = t_prev + 1.0
        a_t = self.LR * T.sqrt(1 - beta2**t) / (1 - beta1**t)

        for param, g_t in zip(self.params, self.gradients):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

            m_t = beta1 * m_prev + (1 - beta1) * g_t
            v_t = beta2 * v_prev + (1 - beta2) * g_t**2
            step = a_t * m_t / (T.sqrt(v_t) + epsilon)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[param] = param - step

        updates[t_prev] = t
        self.step = theano.function(
            [self.X, self.Y] + self.Y_aux,
            [self.top_loss, self.loss_instance],
            updates=updates,
            on_unused_input='warn')

        print " Compiling done  - in %.3f s!" % (time.time() - self.t_init)

##############################################################################################################
#######################################       Resilient backPROPagation      #################################
##############################################################################################################


class compileRPROP(Optimizer):
    """ Resilient backPROPagation """

    def __init__(self,
                 optimizer_params,
                 model_obj=None,
                 X=None,
                 Y=None,
                 Y_aux=[],
                 top_loss=None,
                 params=None):
        print "Compiling RPROP..."
        super(compileRPROP, self).__init__(model_obj, X, Y, Y_aux, top_loss,
                                           params)

        self.LRs = []
        RPROP_updates = []

        # Initialise shared variables for the Training algos
        for i, para in enumerate(self.params):
            if para in self.params[:i]:
                print "Detected RNN or shared param @index =", i
            else:
                self.LRs.append(theano.shared(
                    np.float32(optimizer_params['initial_update_size']) *
                    np.ones(para.get_value().shape, dtype='float32'),
                    name=para.name + str('_RPROP'),
                    borrow=0))

        print "RPROP: missing backtracking handling "  ###TODO ???
        for param_i, grad_i, last_grad_i, pLR_i in zip(
                self.params, self.gradients, self.last_grads, self.LRs):
            # Commented code on next 4 lines is theano-incapable and just illustration!!!
            # if   ((last_grad_i*grad_i) < -1e-9): # sign-change & significant magnitude of last two gradients
            #   pLR_i_new = pLR_i * (1 - np.float32(RPROP_penalty)) # decrease this LR
            # elif ((last_grad_i*grad_i) > 1e-11): # no sign-change & and last two gradients were sufficiently big
            #   pLR_i_new = pLR_i * (1 + np.float32(RPORP_gain))    # increase this LR

            # capping RPROP-LR inside [1e-7,2e-3]
            RPROP_updates.append((pLR_i, T.minimum(T.maximum(
                    pLR_i * (1 - np.float32(optimizer_params['penalty']) * ((last_grad_i * grad_i) < -1e-9)
                             + np.float32(optimizer_params['gain']) * ((last_grad_i * grad_i) > 1e-11)),
                    1e-7 * T.ones_like(pLR_i)), 2e-3 * T.ones_like(pLR_i))))
            RPROP_updates.append((param_i, param_i - pLR_i * grad_i / (T.abs_(
                grad_i) + 1e-6) - (self.weightdecay * param_i)))
            RPROP_updates.append((last_grad_i, grad_i))

        self.step = theano.function([self.X, self.Y] + self.Y_aux,
                                    [self.top_loss, self.loss_instance],
                                    updates=RPROP_updates,
                                    on_unused_input='warn')
        print " Compiling done  - in %.3f s!" % (time.time() - self.t_init)

##############################################################################################################
###########################################       Conjugate Gradient      ####################################
##############################################################################################################


class compileCG(Optimizer):
    """ Conjugate Gradient """

    def __init__(self,
                 optimizer_params,
                 model_obj=None,
                 X=None,
                 Y=None,
                 Y_aux=[],
                 top_loss=None,
                 params=None):
        print "Compiling CG"
        super(compileCG, self).__init__(model_obj, X, Y, Y_aux, top_loss,
                                        params)

        self.optimizer_params = optimizer_params
        self.direc = []
        CG_updates_direc = []
        CG_updates_grads = []
        CG_updates = []  # params
        self.model_obj = model_obj

        # Initialise shared variables for the Training algos
        for para in self.params:
            para_shape = para.get_value().shape
            self.direc.append(theano.shared(
                np.zeros(para_shape, dtype='float32'),
                name=para.name + '_CG_direc',
                borrow=False))

        ### Kickstart of CG, initialise first direction to current gradient ###
        for grad_i, last_grad_i, direc_i in zip(self.gradients, self.last_grads, self.direc):
            CG_updates_grads.append((last_grad_i, grad_i))
            CG_updates_direc.append((direc_i, -grad_i))

        # update direc & last-grad
        updates = CG_updates_grads + CG_updates_direc
        self.CG_kickstart = theano.function(
            [self.X, self.Y] + self.Y_aux,
            None,
            updates=updates,
            on_unused_input='ignore')

        ### Regular CG-step ###
        CG_updates_direc = []  # clear update-list
        CG_updates_grads = []  # clear update-list

        # Compute Polak-Ribiere coefficient b for updating search direction
        denom = theano.shared(np.float32(0))
        num = theano.shared(np.float32(0))

        for grad_i, last_grad_i in zip(self.gradients, self.last_grads):
            num = num + T.sum(-grad_i * (-grad_i + last_grad_i))
            denom = denom + T.sum(last_grad_i * last_grad_i)
        coeff = num / denom
        coeff = T.max(T.stack([coeff, theano.shared(np.float32(0))]))  # select

        # Search-direction and last-grad update
        for grad_i, last_grad_i, direc_i in zip(self.gradients, self.last_grads, self.direc):
            CG_updates_grads.append((last_grad_i, grad_i))
            CG_updates_direc.append((direc_i, -grad_i + direc_i * coeff))

        updates = CG_updates_grads + CG_updates_direc
        #    if self.Y_aux is None:
        #      self.CG_step = theano.function([self.X, self.Y],
        #                                     coeff, updates=updates, on_unused_input='ignore')
        #    else:
        self.CG_step = theano.function([self.X, self.Y] + self.Y_aux, coeff, updates=updates, on_unused_input='ignore')

        # Weights update (Line search), no input needed, as only params are changed
        delta = T.fscalar('delta')  # used to parametrise the ray along we search (=0 at current params)
        self.t = theano.shared(np.float32(0))  # Internal update step indicator
        for param_i, search_direc_i in zip(self.params, self.direc):
            CG_updates.append((param_i, param_i + search_direc_i * delta))

        CG_updates.append((self.t, self.t + delta))
        self.CG_update_params = theano.function([delta], self.t + delta, updates=CG_updates)

        # Linear-Approximation (from shared last_(grad|direc))
        linear_approx = theano.shared(np.float32(0))
        for grad_i, last_direc_i in zip(self.last_grads, self.direc):
            linear_approx = linear_approx + T.sum(grad_i * last_direc_i)

        self.CG_linear_approx = theano.function([], linear_approx, updates=None)

        print " Compiling done  - in %.3f s!" % (time.time() - self.t_init)

    def __call__(self, *args):  # i.e. trainingStepCG
        """
        Perform an update step

        [data, labels(, *aux)] --> [loss, loss_instance]
        loss_instance is the loss per instance (e.g. batch-item or pixel)
        """
        self.CG_kickstart(*args)
        timeline = []
        loss, loss_instance, t, count = self.lineSearch(*args)
        timeline.append([loss, t, 0, count])
        self.t.set_value(np.float32(0))  # DBG: reset internal update-magnitude
        n_steps = self.optimizer_params['n_steps']
        for i in xrange(n_steps - 1):
            if self.optimizer_params['only_descent']:
                self.CG_kickstart(*args)
                coeff = 0
            else:  # use actual CG
                coeff = self.CG_step(*args)

            loss, loss_instance, t, count = self.lineSearch(*args)
            self.t.set_value(np.float32(0))
            timeline.append([loss, t, coeff, count])

        if hasattr(self.model_obj, 'CG_timeline'):
            self.model_obj.CG_timeline.extend(timeline)
        return loss, loss_instance

    def lineSearch(self, *args):
        """ Needed for CG """
        loss_0, _ = self.get_loss(*args)
        loss_0 = np.float32(loss_0)
        linear_approx = self.CG_linear_approx()
        counter = 0
        if linear_approx > 0:  # if algorithm gets stuck, reset
            loss, loss_instance = self.get_loss(*args)
            return loss_0, loss_instance, 0, counter
        max_step = self.optimizer_params['max_step']
        min_step = self.optimizer_params['min_step']
        beta = self.optimizer_params['beta']

        max_count = int(np.log(min_step / (max_step)) / np.log(beta))  # limit iterations to lower bound
        points = []  ### For Plotting
        t = max_step
        # The next search points DEcrement the search ray by the decaying negative factor delta
        # Thus parameters needn't to be reset before updating, we directly change
        # the parameters by the desired amount
        # The first search point: max_step along the current search direction
        self.CG_update_params(np.float32(max_step))
        loss, loss_instance = self.get_loss(*args)
        loss = np.float32(loss)
        counter += 1
        delta = max_step * (beta - 1.0)
        last_loss = 1000000
        for i in xrange(max_count):
            chord = loss_0 + t * linear_approx * self.optimizer_params['alpha']
            points.append([t, loss, chord])
            # The second condition is a deviation from regular BT-LineSearch
            if (loss <= chord) or (loss > last_loss):
                break
            self.CG_update_params(np.float32(delta))
            last_loss = loss
            loss, loss_instance = self.get_loss(*args)
            loss = np.float32(loss)
            delta = delta * beta
            t = t * beta
            counter += 1

        if self.optimizer_params['show']:
            points.append([0, loss_0, loss_0])
            points = np.array(points)
            plt.figure()
            plt.plot(points[:, 0], points[:, 1:])
            plt.scatter(points[:, 0], points[:, 1])
            plt.legend(('fu', 'chord'))
            plt.draw()
            plt.pause(0.0001)

        return loss, loss_instance, np.float(self.t.eval()), counter

##############################################################################################################
###########################################             L-BFGS            ####################################
##############################################################################################################


class compileLBFGS(Optimizer):
    """
    L-BFGS (fast, full-batch method)

    References (cite one):
      R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization,
      (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5, pp. 1190-1208.

      C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large scale
      bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.

      J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for large
      scale bound constrained optimization (2011), ACM Transactions on Mathematical Software, 38, 1.

    """

    def __init__(self,
                 optimizer_params,
                 model_obj=None,
                 X=None,
                 Y=None,
                 Y_aux=[],
                 top_loss=None,
                 params=None,
                 debug=False):
        print "Compiling lBFGS"
        super(compileLBFGS, self).__init__(model_obj, X, Y, Y_aux, top_loss, params)

        self.optimizer_params = optimizer_params
        ret = [self.top_loss] + self.gradients
        self._loss_and_grad = theano.function([self.X, self.Y] + self.Y_aux, ret, on_unused_input='warn')

        self.params_values = [p.get_value() for p in self.params]
        self.shapes = [p.shape for p in self.params_values]  # list of param-shapes
        self.sizes = map(np.prod, self.shapes)  # list of param-sizes
        self.total_size = np.sum(self.sizes)  # length of vectorized parameters
        self.params_vec = np.zeros(self.total_size, "float32")
        self.gradients_vec = np.zeros(self.total_size, "float32")

        self.debug = debug
        print " Compiling done  - in %.3f s!" % (time.time() - self.t_init)

    def vec2list(self, vec, target_list):
        pos = 0
        for a, shape, size in zip(target_list, self.shapes, self.sizes):
            a[...] = vec[pos:pos + size].reshape(shape)
            pos += size

    def list2vect(self, src_list, target_vec):
        pos = 0
        for a, size in zip(src_list, self.sizes):
            target_vec[pos:pos + size] = np.asarray(a).flatten()
            pos += size

    def loss_and_grad(self, params_vect_new, *args):
        """ internal use, updates self.params"""
        self.vec2list(params_vect_new, self.params_values)  # Update param values in list and then on GPU
        for p, val in zip(self.params, self.params_values):
            p.set_value(val, borrow=False)

        ret = self._loss_and_grad(*args)
        loss, grads = np.float32(ret[0]), ret[1:]
        if len(grads) == 1:
            grads = grads[0]

        self.list2vect(grads, self.gradients_vec)
        grad_vec = np.asarray(self.gradients_vec, dtype="float64")

        if self.debug:
            print "loss_and_grad:: x_min,max =",np.min(args[0]),np.max(args[0]),\
            "y_min,max =",np.min(args[1]),np.max(args[1])
            print "loss_and_grad (av(g) =", np.mean(
                grad_vec), ", av(abs(g)) =", np.mean(np.abs(grad_vec)), ")"

        return loss, grad_vec

    def __call__(self, *args):
        """
        Perform an update step

        [data, labels(, *aux)] --> [loss, loss_instance]
        loss_instance is the loss per instance (e.g. batch-item or pixel)
        """
        if self.debug:
            print "\nlbfgs->optimize::"
            print "__optimize:: x_min,max =", np.min(args[0]), np.max(args[0]), "y_min,max =",\
                  np.min(args[1]), np.max(args[1])

        self.params_values = [p.get_value() for p in self.params]
        self.list2vect(self.params_values, self.params_vec)

        params, loss, info_dict = fmin_l_bfgs_b(func=self.loss_and_grad,
                                                x0=self.params_vec,
                                                args=args,
                                                **self.optimizer_params)
        aborted = info_dict["warnflag"] != 1
        if aborted:
            print info_dict
            print "L-BFGS aborted!"
        return loss
