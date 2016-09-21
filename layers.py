from __future__ import division
import numpy as np

from keras.layers.convolutional import conv_output_length
from keras.layers.recurrent import Recurrent
from keras import initializations
from keras import activations
from keras import backend as K
from keras.engine import InputSpec
from keras import regularizers


def time_distributed_conv(x, w, nb_filter, subsample, border_mode, W_shape,
                          dim_ordering, b=None,
                          reshape_dim=None, output_dim=None, timesteps=None):
    '''Apply conv(y, w) + b for every temporal slice y of x.
    '''
    flat_output_dim = np.prod(output_dim)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, )+reshape_dim)

    output = K.conv2d(x, w, strides=subsample,
                      border_mode=border_mode,
                      dim_ordering=dim_ordering,
                      filter_shape=W_shape)
    if b:
        if dim_ordering == 'th':
            output += K.reshape(b, (1, nb_filter, 1, 1))
        elif dim_ordering == 'tf':
            output += K.reshape(b, (1, 1, 1, nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    # reshape to 3D tensor
    x = K.reshape(x, (-1, timesteps, flat_output_dim))
    return x


class ConvRNN(Recurrent):
    """RNN with all convolutional connections:
    H_t = activation(conv(H_tm1, W_hh) + conv(X_t, W_ih) + b)
    with H_t and X_t being images and W being filters.
    We use Keras' RNN API, thus input and outputs should be 3-way tensors.
    Assuming that your input video have frames of size
    [nb_channels, nb_rows, nb_cols], the input of this layer should be reshaped
    to [batch_size, time_length, nb_channels*nb_rows*nb_cols]. Thus, you have to
    pass the original images shape to the ConvRNN layer.
        self.input = K.placeholder(shape=(batch_size, input_dim[1], input_dim[2]))

    Parameters:
    -----------
    nb_filters, nb_row, nb_col: convolutional filter dimensions
    reshape_dim: list [nb_channels, nb_row, nb_col] original dimensions of a
        frame.
    batch_size: int, batch_size is useful for TensorFlow backend.
    time_length: int, optional for Theano, mandatory for TensorFlow
    subsample: (int, int), just keras.layers.Convolutional2D.subsample
    """
    '''Fully-connected RNN where the output is to be fed back to input.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, nb_filter, nb_row, nb_col, reshape_dim, subsample=(1, 1),
                 dim_ordering='th',
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', border_mode='same',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.subsample = subsample
        assert dim_ordering in {'tf', 'th'}
        self.dim_ordering = dim_ordering
        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for ConvRNN:', border_mode)
        self.border_mode = border_mode
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        if len(reshape_dim) is not 4:
            raise Exception("`reshape_dim` must contain 4 ints, ex: (batch_size, row, col, stack)")
        self.reshape_dim = reshape_dim
        self.output_dim = self._get_output_dim(reshape_dim)
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        super(ConvRNN, self).__init__(**kwargs)

    def _get_output_dim(self, input_shape):
        if self.dim_ordering == 'th':
            rows = self.reshape_dim[2]
            cols = self.reshape_dim[3]
        elif self.dim_ordering == 'tf':
            rows = self.reshape_dim[1]
            cols = self.reshape_dim[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.dim_ordering == 'th':
            stack_size = self.reshape_dim[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
            self.U_shape = (self.nb_filter, self.nb_filter, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = self.reshape_dim[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
            self.U_shape = (self.nb_row, self.nb_col, self.nb_filter, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.U = self.init(self.U_shape, name='{}_U'.format(self.name))
        self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_initial_states(self, x):
        initial_state = K.zeros((self.output_dim[0], np.prod(self.output_dim[1:])))
        return [initial_state, ]

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        flat_output_dim = np.prod(self.output_dim[1:])
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], flat_output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], flat_output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            timesteps = input_shape[1]
            return time_distributed_conv(x, self.W, self.nb_filter, self.subsample, self.border_mode, self.W_shape,
                                         self.dim_ordering, self.b,
                                         self.reshape_dim[1:], self.output_dim[1:], timesteps)
        else:
            return x

    def step(self, x, states):
        prev_output = K.reshape(states[0], self.output_dim)

        if self.consume_less == 'cpu':
            h = K.reshape(x, self.reshape_dim)
        else:
            x_t = K.reshape(x, self.reshape_dim)
            h = K.conv2d(x_t, self.W, strides=self.subsample,
                         border_mode=self.border_mode,
                         dim_ordering=self.dim_ordering,
                         filter_shape=self.W_shape)
            if self.dim_ordering == 'th':
                h += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                h += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        output = K.conv2d(prev_output, self.U, strides=(1, 1),
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          filter_shape=self.U_shape)
        output = self.activation(h + output)
        output = K.batch_flatten(output)
        return output, [output]

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return self.output_dim
        else:
            return (input_shape[0], ) + np.prod(self.output_dim[1:])

    def get_config(self):
        config = {'output_dim': np.prod(self.output_dim[1:]),
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None}
        base_config = super(ConvRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
