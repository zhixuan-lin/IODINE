import torch
import torch.nn as nn
from torch.nn import init, Parameter
from torch.autograd import Variable
from .distributions import DiagonalGaussian, PointEstimate


class Dense(nn.Module):

    """Fully-connected (dense) layer with optional batch normalization, non-linearity, weight normalization, and dropout."""

    def __init__(self, n_in, n_out, non_linearity=None, batch_norm=False, weight_norm=False, dropout=0., initialize='glorot_uniform'):
        super(Dense, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm1d(n_out, momentum=0.99)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear, name='weight')

        init_gain = 1.

        if non_linearity is None or non_linearity == 'linear':
            self.non_linearity = None
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
            init_gain = init.calculate_gain('relu')
        elif non_linearity == 'elu':
            self.non_linearity = nn.ELU()
        elif non_linearity == 'selu':
            self.non_linearity = nn.SELU()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
            init_gain = init.calculate_gain('tanh')
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        else:
            raise Exception('Non-linearity ' + str(non_linearity) + ' not found.')

        self.dropout = None
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)

        if initialize == 'normal':
            init.normal(self.linear.weight)
        elif initialize == 'glorot_uniform':
            init.xavier_uniform(self.linear.weight, gain=init_gain)
        elif initialize == 'glorot_normal':
            init.xavier_normal(self.linear.weight, gain=init_gain)
        elif initialize == 'kaiming_uniform':
            init.kaiming_uniform(self.linear.weight)
        elif initialize == 'kaiming_normal':
            init.kaiming_normal(self.linear.weight)
        elif initialize == 'orthogonal':
            init.orthogonal(self.linear.weight, gain=init_gain)
        elif initialize == '':
            pass
        else:
            raise Exception('Parameter initialization ' + str(initialize) + ' not found.')

        if batch_norm:
            init.normal(self.bn.weight, 1, 0.02)
            init.constant(self.bn.bias, 0.)

        init.constant(self.linear.bias, 0.)

    def random_re_init(self, re_init_fraction):
        pass

    def forward(self, input):
        *A, _ = input.size()
        input = input.view(-1, self.n_in)
        output = self.linear(input)
        if self.bn:
            output = self.bn(output)
        if self.non_linearity:
            output = self.non_linearity(output)
        if self.dropout:
            output = self.dropout(output)
        output = output.view(*A, self.n_out)
        return output


class Conv(nn.Module):

    """Basic convolutional layer with optional batch normalization, non-linearity, weight normalization and dropout."""

    def __init__(self, n_in, filter_size, n_out, non_linearity=None, batch_norm=False, weight_norm=False, dropout=0., initialize='glorot_uniform'):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(n_in, n_out, filter_size, padding=int(np.ceil(filter_size/2)))
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm2d(n_out)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv, name='weight')

        if non_linearity is None:
            self.non_linearity = None
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'elu':
            self.non_linearity = nn.ELU()
        elif non_linearity == 'selu':
            self.non_linearity = nn.SELU()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        else:
            raise Exception('Non-linearity ' + str(non_linearity) + ' not found.')

        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)

        if initialize == 'normal':
            init.normal(self.conv.weight)
        elif initialize == 'glorot_uniform':
            init.xavier_uniform(self.conv.weight)
        elif initialize == 'glorot_normal':
            init.xavier_normal(self.conv.weight)
        elif initialize == 'kaiming_uniform':
            init.kaiming_uniform(self.conv.weight)
        elif initialize == 'kaiming_normal':
            init.kaiming_normal(self.conv.weight)
        elif initialize == 'orthogonal':
            init.orthogonal(self.conv.weight)
        elif initialize == '':
            pass
        else:
            raise Exception('Parameter initialization ' + str(initialize) + ' not found.')

        if batch_norm:
            init.constant(self.bn.weight, 1.)
            init.constant(self.bn.bias, 0.)

        init.constant(self.conv.bias, 0.)

    def forward(self, input):
        output = self.conv(input)
        if self.bn:
            output = self.bn(output)
        if self.non_linearity:
            output = self.non_linearity(output)
        if self.dropout:
            output = self.dropout(output)
        return output


class Recurrent(nn.Module):

    def __init__(self, n_in, n_units):
        super(Recurrent, self).__init__()
        self.lstm = nn.LSTMCell(n_in, n_units)
        self.initial_hidden = Parameter(torch.zeros(1, n_units))
        self.initial_cell = Parameter(torch.zeros(1, n_units))
        self.hidden_state = None
        self.cell_state = None

    def forward(self, input):
        if self.hidden_state is None:
            self.hidden_state = self.initial_hidden.repeat(input.data.shape[0], 1)
        if self.cell_state is None:
            self.cell_state = self.initial_cell.repeat(input.data.shape[0], 1)
        self.hidden_state, self.cell_state = self.lstm.forward(input, (self.hidden_state, self.cell_state))
        return self.hidden_state

    def reset(self):
        self.hidden_state = None
        self.cell_state = None


class DenseInverseAutoRegressive(nn.Module):

    def __init__(self, n):
        super(DenseInverseAutoRegressive, self).__init__()
        self.mean = Dense(n, n)
        self.std = Dense(n, n)

    def forward(self, input):
        return (input - self.mean(input)) / self.std(input)


class MultiLayerPerceptron(nn.Module):

    """Multi-layered perceptron."""

    def __init__(self, n_in, n_units, n_layers, non_linearity=None, connection_type='sequential', batch_norm=False, weight_norm=False, dropout=0.):

        super(MultiLayerPerceptron, self).__init__()
        assert connection_type in ['sequential', 'residual', 'highway', 'concat_input', 'concat'], 'Connection type not found.'
        self.connection_type = connection_type
        self.layers = nn.ModuleList([])
        self.gates = nn.ModuleList([])

        self.n_in = n_in
        n_in_orig = n_in

        if self.connection_type in ['residual', 'highway']:
            self.initial_dense = Dense(n_in, n_units, batch_norm=batch_norm, weight_norm=weight_norm)

        output_size = 0

        for _ in range(n_layers):
            layer = Dense(n_in, n_units, non_linearity=non_linearity, batch_norm=batch_norm, weight_norm=weight_norm, dropout=dropout)
            self.layers.append(layer)

            if self.connection_type == 'highway':
                gate = Dense(n_in, n_units, non_linearity='sigmoid', batch_norm=batch_norm, weight_norm=weight_norm)
                self.gates.append(gate)
            if self.connection_type in ['sequential', 'residual', 'highway']:
                n_in = n_units
            elif self.connection_type == 'concat_input':
                n_in = n_units + n_in_orig
            elif self.connection_type == 'concat':
                n_in += n_units

            output_size = n_in

        self.n_out = output_size

    def random_re_init(self, re_init_fraction):
        for layer in self.layers:
            layer.random_re_init(re_init_fraction)
        if self.connection_type == 'highway':
            for gate in self.gates:
                gate.random_re_init(re_init_fraction)

    def forward(self, input):
    
        *A, _ = input.size()
        input = input.view(-1, self.n_in)
        input_orig = input.clone()

        for layer_num, layer in enumerate(self.layers):
            if self.connection_type == 'sequential':
                input = layer(input)
            elif self.connection_type == 'residual':
                if layer_num == 0:
                    input = self.initial_dense(input) + layer(input)
                else:
                    input = input + layer(input)
            elif self.connection_type == 'highway':
                gate = self.gates[layer_num]
                if layer_num == 0:
                    input = gate(input) * self.initial_dense(input) + (1 - gate(input)) * layer(input)
                else:
                    input = gate(input) * input + (1 - gate(input)) * layer(input)
            elif self.connection_type == 'concat_input':
                input = torch.cat((input_orig, layer(input)), dim=1)
            elif self.connection_type == 'concat':
                input = torch.cat((input, layer(input)), dim=1)

        input = input.view(*A, self.n_out)
        return input


class MultiLayerConv(nn.Module):

    """Multi-layer convolutional network."""

    def __init__(self, n_in, n_filters, filter_size, n_layers, non_linearity=None, connection_type='sequential', batch_norm=False, weight_norm=False, dropout=0.):
        super(MultiLayerConv, self).__init__()
        assert connection_type in ['sequential', 'residual', 'highway', 'concat_input', 'concat'], 'Connection type not found.'
        self.connection_type = connection_type
        self.layers = nn.ModuleList([])
        self.gates = nn.ModuleList([])

        n_in_orig = n_in

        if self.connection_type in ['residual', 'highway']:
            self.initial_conv = Conv(n_in, filter_size, n_filters, batch_norm=batch_norm, weight_norm=weight_norm)

        for _ in range(n_layers):
            layer = Conv(n_in, n_units, non_linearity=non_linearity, batch_norm=batch_norm, weight_norm=weight_norm, dropout=dropout)
            self.layers.append(layer)

            if self.connection_type == 'highway':
                gate = Conv(n_in, filter_size, n_filters, non_linearity='sigmoid', batch_norm=batch_norm, weight_norm=weight_norm)
                self.gates.append(gate)

            if self.connection_type in ['sequential', 'residual', 'highway']:
                n_in = n_filters
            elif self.connection_type == 'concat_input':
                n_in = n_filters + n_in_orig
            elif self.connection_type == 'concat':
                n_in += n_filters

    def forward(self, input):

        input_orig = input.clone()

        for layer_num, layer in enumerate(self.layers):
            if self.connection_type == 'sequential':
                input = layer(input)

            elif self.connection_type == 'residual':
                if layer_num == 0:
                    input = self.initial_conv(input) + layer(input)
                else:
                    input = input + layer(input)

            elif self.connection_type == 'highway':
                gate = self.gates[layer_num]
                if layer_num == 0:
                    input = gate * self.initial_conv(input) + (1 - gate) * layer(input)
                else:
                    input = gate * input + (1 - gate) * layer(input)

            elif self.connection_type == 'concat_input':
                input = torch.cat((input_orig, layer(input)), dim=1)

            elif self.connection_type == 'concat':
                input = torch.cat((input, layer(input)), dim=1)

        return input


class MultiLayerRecurrent(nn.Module):

    def __init__(self, n_in, n_layers, n_units, connection_type='sequential', **kwargs):
        super(MultiLayerRecurrent, self).__init__()
        self.n_layers = n_layers
        self.connection_type = connection_type
        self.layers = nn.ModuleList([])
        self.layers.append(Recurrent(n_in, n_units))
        for _ in range(1, self.n_layers):
            self.layers.append(Recurrent(n_units, n_units))
        if connection_type == 'residual':
            self.input_map = Dense(n_in, n_units)
        if connection_type == 'highway':
            self.gates = nn.ModuleList([])
            self.input_map = Dense(n_in, n_units)
            self.gates.append(Dense(n_in, n_units, non_linearity='sigmoid'))
            for _ in range(1, self.n_layers):
                self.gates.append(Dense(n_units, n_units, non_linearity='sigmoid'))

    def forward(self, input):
        input_orig = input.clone()
        for layer_num, layer in enumerate(self.layers):
            layer_output = layer(input)
            if self.connection_type == 'sequential':
                input = layer_output
            elif self.connection_type == 'residual':
                if layer_num == 0:
                    input = self.input_map(input) + layer_output
                else:
                    input = input + layer_output
            elif self.connection_type == 'highway':
                gate = self.gates[layer_num](input)
                if layer_num == 0:
                    input = self.input_map(input)
                input = gate * input + (1. - gate) * layer_output
            elif self.connection_type == 'concat':
                input = torch.cat((input, layer_output), dim=1)
            elif self.connection_type == 'concat_input':
                input = torch.cat((input_orig, layer_output), dim=1)
        return input

    def reset(self):
        for layer in self.layers:
            layer.reset()


class DenseGaussianVariable(object):

    def __init__(self, batch_size, n_variables, const_prior_var, n_input, update_form, posterior_form='gaussian', learn_prior=True):

        self.batch_size = batch_size
        self.n_variables = n_variables
        assert update_form in ['direct', 'highway'], 'Latent variable update form not found.'
        self.update_form = update_form
        self.posterior_form = posterior_form
        self.learn_prior = learn_prior

        if self.learn_prior:
            self.prior_mean = Dense(n_input[1], self.n_variables)
            self.prior_log_var = None
            if not const_prior_var:
                self.prior_log_var = Dense(n_input[1], self.n_variables)
        self.posterior_mean = Dense(n_input[0], self.n_variables)
        if self.posterior_form == 'gaussian':
            self.posterior_log_var = Dense(n_input[0], self.n_variables)

        if self.update_form == 'highway':
            self.posterior_mean_gate = Dense(n_input[0], self.n_variables, 'sigmoid')
            if self.posterior_form == 'gaussian':
                self.posterior_log_var_gate = Dense(n_input[0], self.n_variables, 'sigmoid')

        self.posterior = self.init_dist(self.posterior_form)
        self.prior = self.init_dist()
        if self.learn_prior and const_prior_var:
            self.prior.log_var_trainable()

    def init_dist(self, form='gaussian'):
        """
        Initializes a distribution.
        :param form: the form of the distribution, either Gaussian or point estimate (Dirac delta).
        :return: the initialized distribution
        """
        if form == 'gaussian':
            return DiagonalGaussian(self.n_variables, Variable(torch.zeros(self.batch_size, self.n_variables)),
                                    Variable(torch.zeros(self.batch_size, self.n_variables)))
        elif form == 'point_estimate':
            return PointEstimate(Variable(torch.zeros(self.batch_size, self.n_variables)))
        else:
            raise Exception('Distribution form not found.')

    def encode(self, input):
        """
        Encode the input into an estimate of / update to the approximate posterior parameters.
        :param input: the input to the variable
        :return: tensor of approximate posterior samples of size (batch_size x 1 x n_variables)
        """
        # encode the mean and log variance, update, return sample
        mean = self.posterior_mean(input)
        if self.posterior_form == 'gaussian':
            log_var = torch.clamp(self.posterior_log_var(input), -15., 15.)
        if self.update_form == 'highway':
            mean_gate = self.posterior_mean_gate(input)
            if self.posterior_form == 'gaussian':
                log_var_gate = self.posterior_log_var_gate(input)
            mean = mean_gate * self.posterior.mean.detach() + (1 - mean_gate) * mean
            if self.posterior_form == 'gaussian':
                log_var = torch.clamp(log_var_gate * self.posterior.log_var.detach() + (1 - log_var_gate) * log_var, -15., 15.)
        self.posterior.mean = mean
        self.posterior.mean.retain_grad()
        if self.posterior_form == 'gaussian':
            self.posterior.log_var = log_var
            self.posterior.log_var.retain_grad()
        return self.posterior.sample(resample=True)

    def decode(self, input, n_samples, generate=False):
        """
        Generates a sample from the prior or the approximate posterior.
        :param input: the input from above if learning the prior
        :param n_samples: number of samples to draw
        :param generate: whether to sample from the prior or the approximate posterior
        :return: tensor of samples of size (batch_size x n_samples x n_variables)
        """
        if self.learn_prior:
            # reshape samples into batch dimension
            batch_size = input.size()[0]
            sample_size = input.size()[1]
            data_size = input.size()[2]
            input = input.view(-1, data_size)
            mean = self.prior_mean(input).view(batch_size, sample_size, -1)
            self.prior.mean = mean
            if self.prior_log_var is not None:
                log_var = self.prior_log_var(input).view(batch_size, sample_size, -1)
                self.prior.log_var = log_var
        if generate:
            sample = self.prior.sample(n_samples=n_samples, resample=True)
        else:
            sample = self.posterior.sample(n_samples=n_samples, resample=True)
        return sample

    def error(self, averaged=True):
        """
        Calculates the error for this variable (sample - prior_mean)
        :param averaged: whether to average over samples
        :return: the error
        """
        sample = self.posterior.sample()
        n_samples = sample.size()[1]
        prior_mean = self.prior.mean.detach()
        if len(prior_mean.data.shape) == 2:
            prior_mean = prior_mean.unsqueeze(1).repeat(1, n_samples, 1)
        if averaged:
            return (sample - prior_mean).mean(dim=1)
        else:
            return sample - prior_mean

    def norm_error(self, averaged=True):
        """
        Calculates the normalized error for this variable (sample - prior_mean) / prior_variance
        :param averaged: whether to average over samples
        :return: the normalized error
        """
        sample = self.posterior.sample()
        n_samples = sample.size()[1]
        prior_mean = self.prior.mean.detach()
        if len(prior_mean.data.shape) == 2:
            prior_mean = prior_mean.unsqueeze(1).repeat(1, n_samples, 1)
        prior_log_var = self.prior.log_var.detach()
        if len(prior_log_var.data.shape) == 2:
            prior_log_var = prior_log_var.unsqueeze(1).repeat(1, n_samples, 1)
        n_error = (sample - prior_mean) / torch.exp(prior_log_var + 1e-7)
        if averaged:
            n_error = n_error.mean(dim=1)
        return n_error

    def kl_divergence(self):
        """
        Calculates an estimate of the KL divergence for this variable.
        Using the current estimate of the distributions and approximate posterior sample
        :return: KL divergence estimate of size (batch_size x n_samples x n_variables)
        """
        return self.posterior.log_prob(self.posterior.sample()) - self.prior.log_prob(self.posterior.sample())

    def analytical_kl(self):
        """
        Calculates the analytical KL divergence between two Gaussian distributions.
        WARNING: currently assumes a standard Normal for the prior
        :return: KL divergence of size (batch_size x n_samples x n_variables)
        """
        n_samples = self.posterior.sample().size()[1]
        kl = -0.5 * (1 + self.posterior.log_var - torch.pow(self.posterior.mean, 2) - torch.exp(self.posterior.log_var))
        return kl.unsqueeze(1).repeat(1, n_samples, 1)

    def reset(self, mean=None, log_var=None, from_prior=True):
        """
        Resets the approximate posterior estimate.
        :param mean: value to set as the new mean
        :param log_var: value to set as the new log variance
        :param from_prior: whether to initialize using the prior
        :return: None
        """
        if from_prior:
            mean = self.prior.mean.data.clone()
            log_var = self.prior.log_var.data.clone()
            if len(mean.shape) == 3:
                mean = mean.mean(dim=1)
            if len(log_var.shape) == 3:
                log_var = log_var.mean(dim=1)
        self.reset_mean(mean)
        if self.posterior_form == 'gaussian':
            self.reset_log_var(log_var)

    def reset_mean(self, value):
        self.posterior.reset_mean(value)

    def reset_log_var(self, value):
        self.posterior.reset_log_var(value)

    def trainable_mean(self):
        self.posterior.mean_trainable()

    def trainable_log_var(self):
        self.posterior.log_var_trainable()

    def not_trainable_mean(self):
        self.posterior.mean_not_trainable()

    def not_trainable_log_var(self):
        self.posterior.log_var_not_trainable()

    def eval(self):
        """
        Puts the variable into eval mode.
        :return: None
        """
        if self.learn_prior:
            self.prior_mean.eval()
            if self.prior_log_var is not None:
                self.prior_log_var.eval()
        self.posterior_mean.eval()
        if self.posterior_form == 'gaussian':
            self.posterior_log_var.eval()
        if self.update_form == 'highway':
            self.posterior_mean_gate.eval()
            if self.posterior_form == 'gaussian':
                self.posterior_log_var_gate.eval()

    def train(self):
        """
        Puts the variable into train mode.
        :return: None
        """
        if self.learn_prior:
            self.prior_mean.train()
            if self.prior_log_var is not None:
                self.prior_log_var.train()
        self.posterior_mean.train()
        if self.posterior_form == 'gaussian':
            self.posterior_log_var.train()
        if self.update_form == 'highway':
            self.posterior_mean_gate.train()
            if self.posterior_form == 'gaussian':
                self.posterior_log_var_gate.train()

    def cuda(self, device_id=0):
        """
        Places the variable on the GPU.
        :param device_id: device on which to place the variable.
        :return: None
        """
        if self.learn_prior:
            self.prior_mean.cuda(device_id)
            if self.prior_log_var is not None:
                self.prior_log_var.cuda(device_id)
        self.posterior_mean.cuda(device_id)
        if self.posterior_form == 'gaussian':
            self.posterior_log_var.cuda(device_id)
        if self.update_form == 'highway':
            self.posterior_mean_gate.cuda(device_id)
            if self.posterior_form == 'gaussian':
                self.posterior_log_var_gate.cuda(device_id)
        self.prior.cuda(device_id)
        self.posterior.cuda(device_id)

    def parameters(self):
        """
        Gets all parameters for this variable.
        :return: List of all parameters.
        """
        return self.encoder_parameters() + self.decoder_parameters()

    def encoder_parameters(self):
        """
        Gets all encoder parameters for this variable.
        :return: List of all encoder parameters.
        """
        encoder_params = []
        encoder_params.extend(list(self.posterior_mean.parameters()))
        if self.posterior_form == 'gaussian':
            encoder_params.extend(list(self.posterior_log_var.parameters()))
        if self.update_form == 'highway':
            encoder_params.extend(list(self.posterior_mean_gate.parameters()))
            if self.posterior_form == 'gaussian':
                encoder_params.extend(list(self.posterior_log_var_gate.parameters()))
        return encoder_params

    def decoder_parameters(self):
        """
        Gets all decoder parameters for this variable.
        :return: List of all decoder parameters.
        """
        decoder_params = []
        if self.learn_prior:
            decoder_params.extend(list(self.prior_mean.parameters()))
            if self.prior_log_var is not None:
                decoder_params.extend(list(self.prior_log_var.parameters()))
            else:
                decoder_params.append(self.prior.log_var)
        return decoder_params

    def state_parameters(self):
        """
        Gets the state (approximate posterior) parameters.
        :return: List of state parameters.
        """
        return self.posterior.state_parameters()

    def state_gradients(self):
        """
        Gets the state (approximate posterior) gradients.
        :return: List containing approximate posterior mean (and possibly log variance) gradients.
        """
        assert self.posterior.mean.grad is not None, 'State gradients are None.'
        grads = [self.posterior.mean.grad.detach()]
        if self.posterior_form == 'gaussian':
            grads += [self.posterior.log_var.grad.detach()]
        for grad in grads:
            grad.volatile = False
        return grads


class ConvGaussianVariable(object):

    def __init__(self, batch_size, n_variable_channels, filter_size, const_prior_var, n_input, update_form, learn_prior=True):

        self.batch_size = batch_size
        self.n_variable_channels = n_variable_channels
        assert update_form in ['direct', 'highway'], 'Variable update form not found.'
        self.update_form = update_form
        self.learn_prior = learn_prior

        if self.learn_prior:
            self.prior_mean = Conv(n_input[1], filter_size, self.n_variable_channels)
            self.prior_log_var = None
            if not const_prior_var:
                self.prior_log_var = Conv(n_input[1], filter_size, self.n_variable_channels)
        self.posterior_mean = Conv(n_input[0], filter_size, self.n_variable_channels)
        self.posterior_log_var = Conv(n_input[0], filter_size, self.n_variable_channels)

        if self.update_form == 'highway':
            self.posterior_mean_gate = Conv(n_input[0], filter_size, self.n_variable_channels, 'sigmoid')
            self.posterior_log_var_gate = Conv(n_input[0], filter_size, self.n_variable_channels, 'sigmoid')

        self.posterior = DiagonalGaussian()
        self.prior = DiagonalGaussian()
        if self.learn_prior and const_prior_var:
            self.prior.log_var_trainable()

    def encode(self, input):
        # encode the mean and log variance, update, return sample
        mean, log_var = self.posterior_mean(input), self.posterior_log_var(input)
        if self.update_form == 'highway':
            mean_gate = self.posterior_mean_gate(input)
            log_var_gate = self.posterior_log_var_gate(input)
            mean = mean_gate * self.posterior.mean + (1 - mean_gate) * mean
            log_var = log_var_gate * self.posterior.log_var + (1 - log_var_gate) * log_var
        self.posterior.mean, self.posterior.log_var = mean, log_var
        return self.posterior.sample()

    def decode(self, input, generate=False):
        # decode the mean and log variance, update, return sample
        mean, log_var = self.prior_mean(input), self.prior_log_var(input)
        self.prior.mean, self.prior.log_var = mean, log_var
        sample = self.prior.sample() if generate else self.posterior.sample()
        return sample

    def error(self):
        return self.posterior.sample() - self.prior.mean

    def norm_error(self):
        return self.error() / (torch.exp(self.prior.log_var) + 1e-7)

    def KL_divergence(self):
        return self.posterior.log_prob(self.posterior.sample()) - self.prior.log_prob(self.posterior.sample())

    def reset(self):
        self.reset_mean()
        self.reset_log_var()

    def reset_mean(self):
        self.posterior.reset_mean()

    def reset_log_var(self):
        self.posterior.reset_log_var()

    def trainable_mean(self, trainable=True):
        self.posterior.mean_trainable(trainable)

    def trainable_log_var(self, trainable=True):
        self.posterior.log_var_trainable(trainable)

    def cuda(self, device_id=0):
        # place all modules on the GPU
        self.prior_mean.cuda(device_id)
        self.prior_log_var.cuda(device_id)
        self.posterior_mean.cuda(device_id)
        self.posterior_log_var.cuda(device_id)
        if self.update_form == 'highway':
            self.posterior_mean_gate.cuda(device_id)
            self.posterior_log_var_gate.cuda(device_id)
        self.prior.cuda(device_id)
        self.posterior.cuda(device_id)

    def parameters(self):
        pass

    def encoder_parameters(self):
        pass

    def decoder_parameters(self):
        pass

    def state_parameters(self):
        return self.posterior.state_parameters()


class DenseLatentLevel(object):

    def __init__(self, batch_size, encoder_arch, decoder_arch, n_latent, n_det, encoding_form, const_prior_var,
                 variable_update_form, posterior_form='gaussian', learn_prior=True):

        self.batch_size = batch_size
        self.n_latent = n_latent
        self.encoding_form = encoding_form

        self.encoder = None
        if encoder_arch is not None:
            self.encoder = MultiLayerPerceptron(**encoder_arch)
        self.decoder = MultiLayerPerceptron(**decoder_arch)

        variable_input_sizes = (encoder_arch['n_units'], decoder_arch['n_units'])

        self.latent = DenseGaussianVariable(self.batch_size, self.n_latent, const_prior_var, variable_input_sizes,
                                            variable_update_form, posterior_form, learn_prior)
        self.deterministic_encoder = Dense(variable_input_sizes[0], n_det[0]) if n_det[0] > 0 else None
        self.deterministic_decoder = Dense(variable_input_sizes[1], n_det[1]) if n_det[1] > 0 else None

    def get_encoding(self, input, in_out):
        # 'encoding_form': ['posterior', 'log_gradient', 'sign_gradient', 'mean', 'log_var'],
        encoding = input if in_out == 'in' else None
        if 'posterior' in self.encoding_form and in_out == 'out':
            encoding = input
        if ('top_error' in self.encoding_form and in_out == 'in') or ('bottom_error' in self.encoding_form and in_out == 'out'):
            error = self.latent.error()
            encoding = error if encoding is None else torch.cat((encoding, error), 1)
        if ('l2_norm_top_error' in self.encoding_form and in_out == 'in') or ('l2_norm_bottom_error' in self.encoding_form and in_out == 'out'):
            error = self.latent.error()
            norm_error = error / torch.norm(error, 2, 1, True)
            encoding = norm_error if encoding is None else torch.cat((encoding, norm_error), 1)
        if ('layer_norm_top_error' in self.encoding_form and in_out == 'in') or ('layer_norm_bottom_error' in self.encoding_form and in_out == 'out'):
            # TODO
            pass
        if ('top_norm_error' in self.encoding_form and in_out == 'in') or ('bottom_norm_error' in self.encoding_form and in_out == 'out'):
            norm_error = self.latent.norm_error()
            encoding = norm_error if encoding is None else torch.cat((encoding, norm_error), 1)
        if ('l2_norm_top_norm_error' in self.encoding_form and in_out == 'in') or ('l2_norm_bottom_norm_error' in self.encoding_form and in_out == 'out'):
            norm_error = self.latent.norm_error()
            norm_norm_error = norm_error / torch.norm(norm_error, 2, 1, True)
            encoding = norm_norm_error if encoding is None else torch.cat((encoding, norm_norm_error), 1)
        if ('layer_norm_top_norm_error' in self.encoding_form and in_out == 'in') or ('layer_norm_bottom_norm_error' in self.encoding_form and in_out == 'out'):
            # TODO
            pass
        if ('log_top_error' in self.encoding_form and in_out == 'in') or ('log_bottom_error' in self.encoding_form and in_out == 'out'):
            log_error = torch.log(torch.abs(self.latent.error()))
            encoding = log_error if encoding is None else torch.cat((encoding, log_error), 1)
        if ('sign_top_error' in self.encoding_form and in_out == 'in') or ('sign_bottom_error' in self.encoding_form and in_out == 'out'):
            sign_error = torch.sign(self.latent.error())
            encoding = sign_error if encoding is None else torch.cat((encoding, sign_error), 1)
        if 'mean' in self.encoding_form and in_out == 'in':
            approx_post_mean = self.latent.posterior.mean.detach()
            if len(approx_post_mean.data.shape) == 3:
                approx_post_mean = approx_post_mean.mean(dim=1)
            encoding = approx_post_mean if encoding is None else torch.cat((encoding, approx_post_mean), 1)
        if 'l2_norm_mean' in self.encoding_form and in_out == 'in':
            mean_norm = torch.norm(self.latent.posterior.mean.detach(), 2, 1, True)
            norm_mean = self.latent.posterior.mean.detach()/mean_norm
            if len(norm_mean.data.shape) == 3:
                norm_mean = norm_mean.mean(dim=1)
            encoding = norm_mean if encoding is None else torch.cat((encoding, norm_mean), 1)
        if 'layer_norm_mean' in self.encoding_form and in_out == 'in':
            post_mean = self.latent.posterior.mean.detach()
            layer_mean = post_mean.mean(dim=0, keepdim=True)
            # layer_std = post_mean.std(dim=0, keepdim=True)
            norm_mean = post_mean - layer_mean
            if len(norm_mean.data.shape) == 3:
                norm_mean = norm_mean.mean(dim=1)
            encoding = norm_mean if encoding is None else torch.cat((encoding, norm_mean), 1)
        if 'log_var' in self.encoding_form and in_out == 'in':
            approx_post_log_var = self.latent.posterior.log_var.detach()
            if len(approx_post_log_var.data.shape) == 3:
                approx_post_log_var = approx_post_mean.mean(dim=1)
            encoding = approx_post_log_var if encoding is None else torch.cat((encoding, approx_post_log_var), 1)
        if 'l2_norm_log_var' in self.encoding_form and in_out == 'in':
            log_var_norm = torch.norm(self.latent.posterior.log_var.detach(), 2, 1, True)
            norm_log_var = self.latent.posterior.log_var.detach()/log_var_norm
            if len(norm_log_var.data.shape) == 3:
                norm_log_var = norm_log_var.mean(dim=1)
            encoding = norm_log_var if encoding is None else torch.cat((encoding, norm_log_var), 1)
        if 'layer_norm_log_var' in self.encoding_form and in_out == 'in':
            post_log_var = self.latent.posterior.log_var.detach()
            layer_mean = post_log_var.mean(dim=0, keepdim=True)
            # layer_std = post_mean.std(dim=0, keepdim=True)
            norm_log_var = post_log_var - layer_mean
            if len(norm_log_var.data.shape) == 3:
                norm_log_var = norm_log_var.mean(dim=1)
            encoding = norm_log_var if encoding is None else torch.cat((encoding, norm_log_var), 1)
        if 'var' in self.encoding_form and in_out == 'in':
            approx_post_var = torch.exp(self.latent.posterior.log_var.detach())
            if len(approx_post_var.data.shape) == 3:
                approx_post_var = approx_post_mean.mean(dim=1)
            encoding = approx_post_var if encoding is None else torch.cat((encoding, approx_post_var), 1)
        if 'mean_gradient' in self.encoding_form and in_out == 'in':
            encoding = self.state_gradients()[0] if encoding is None else torch.cat((encoding, self.state_gradients()[0]), 1)
        if 'l2_norm_mean_gradient' in self.encoding_form and in_out == 'in':
            grad_norm = torch.norm(self.state_gradients()[0], 2, 1, True)
            norm_mean_grad = self.state_gradients()[0]/grad_norm
            encoding = norm_mean_grad if encoding is None else torch.cat((encoding, norm_mean_grad), 1)
        if 'layer_norm_mean_gradient' in self.encoding_form and in_out == 'in':
            mean_grad = self.state_gradients()[0]
            layer_mean = mean_grad.mean(dim=0, keepdim=True)
            layer_std = mean_grad.std(dim=0, keepdim=True)
            norm_mean_grad = (mean_grad - layer_mean) / (layer_std + 1e-5)
            encoding = norm_mean_grad if encoding is None else torch.cat((encoding, norm_mean_grad), 1)
        if 'log_var_gradient' in self.encoding_form and in_out == 'in':
            encoding = self.state_gradients()[1] if encoding is None else torch.cat((encoding, self.state_gradients()[1]), 1)
        if 'l2_norm_log_var_gradient' in self.encoding_form and in_out == 'in':
            #log_var_grad = self.state_gradients()[1]
            #log_var_grad = log_var_grad - log_var_grad.mean(dim=0, keepdim=True)
            grad_norm = torch.norm(self.state_gradients()[1], 2, 1, True)
            norm_log_var_grad = self.state_gradients()[1]/grad_norm
            encoding = norm_log_var_grad if encoding is None else torch.cat((encoding, norm_log_var_grad), 1)
        if 'layer_norm_log_var_gradient' in self.encoding_form and in_out == 'in':
            log_var_grad = self.state_gradients()[1]
            layer_mean = log_var_grad.mean(dim=0, keepdim=True)
            layer_std = log_var_grad.std(dim=0, keepdim=True)
            norm_log_var_grad = (log_var_grad - layer_mean) / (layer_std + 1e-5)
            encoding = norm_log_var_grad if encoding is None else torch.cat((encoding, norm_log_var_grad), 1)
        if 'gradient' in self.encoding_form and in_out == 'in':
            encoding = torch.cat(self.state_gradients(), 1) if encoding is None else torch.cat([encoding] + self.state_gradients(), 1)
        if 'l2_norm_gradient' in self.encoding_form and in_out == 'in':
            norm_mean_grad = self.state_gradients()[0] / torch.norm(self.state_gradients()[0], 2, 1, True)
            norm_log_var_grad = self.state_gradients()[1] / torch.norm(self.state_gradients()[1], 2, 1, True)
            norm_state_gradients = [norm_mean_grad, norm_log_var_grad]
            encoding = torch.cat(norm_state_gradients, 1) if encoding is None else torch.cat([encoding] + norm_state_gradients, 1)
        if 'log_gradient' in self.encoding_form and in_out == 'in':
            log_grads = torch.log(torch.cat(self.state_gradients(), 1).abs() + 1e-5)
            encoding = log_grads if encoding is None else torch.cat((encoding, log_grads), 1)
        if 'scaled_log_gradient' in self.encoding_form and in_out == 'in':
            log_grads = torch.clamp(torch.log(torch.cat(self.state_gradients(), 1).abs() + 1e-5) * 10., min=-5.)
            encoding = log_grads if encoding is None else torch.cat((encoding, log_grads), 1)
        if 'sign_gradient' in self.encoding_form and in_out == 'in':
            sign_grad = torch.sign(torch.cat(self.state_gradients(), 1))
            encoding = sign_grad if encoding is None else torch.cat((encoding, sign_grad), 1)
        return encoding

    def encode(self, input):
        # encode the input, possibly with errors, concatenate any deterministic units
        # 'encoding_form': ['posterior', 'log_gradient', 'sign_gradient', 'mean', 'log_var'],
        encoded = self.encoder(self.get_encoding(input, 'in'))
        output = self.get_encoding(self.latent.encode(encoded).mean(dim=1), 'out')
        if self.deterministic_encoder:
            det = self.deterministic_encoder(encoded)
            output = torch.cat((det, output), 1)
        return output

    def decode(self, input, n_samples, generate=False):
        # decode the input, sample the latent variable, concatenate any deterministic units
        # reshape input to put samples in batch dimension
        batch_size = input.size()[0]
        input = input.view(-1, input.size()[2])
        decoded = self.decoder(input)
        # reshape back into samples in dim 1
        decoded = decoded.view(batch_size, n_samples, -1)
        sample = self.latent.decode(decoded, n_samples, generate=generate)
        if self.deterministic_decoder:
            decoded = decoded.view(-1, decoded.size()[2])
            det = self.deterministic_decoder(decoded)
            det = det.view(batch_size, n_samples, -1)
            sample = torch.cat((sample, det), dim=2)
        return sample

    def kl_divergence(self):
        return self.latent.kl_divergence()

    def reset(self, mean=None, log_var=None, from_prior=True):
        self.latent.reset(mean=mean, log_var=log_var, from_prior=from_prior)

    def trainable_state(self):
        self.latent.trainable_mean()
        if self.latent.posterior_form == 'gaussian':
            self.latent.trainable_log_var()

    def not_trainable_state(self):
        self.latent.not_trainable_mean()
        if self.latent.posterior_form == 'gaussian':
            self.latent.not_trainable_log_var()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.latent.eval()

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.latent.train()

    def cuda(self, device_id=0):
        # place all modules on the GPU
        self.encoder.cuda(device_id)
        self.decoder.cuda(device_id)
        self.latent.cuda(device_id)
        if self.deterministic_encoder:
            self.deterministic_encoder.cuda(device_id)
        if self.deterministic_decoder:
            self.deterministic_decoder.cuda(device_id)

    def parameters(self):
        return self.encoder_parameters() + self.decoder_parameters()

    def encoder_parameters(self):
        encoder_params = []
        encoder_params.extend(list(self.encoder.parameters()))
        if self.deterministic_encoder:
            encoder_params.extend(list(self.deterministic_encoder.parameters()))
        encoder_params.extend(list(self.latent.encoder_parameters()))
        return encoder_params

    def decoder_parameters(self):
        decoder_params = []
        decoder_params.extend(list(self.decoder.parameters()))
        if self.deterministic_decoder:
            decoder_params.extend(list(self.deterministic_decoder.parameters()))
        decoder_params.extend(list(self.latent.decoder_parameters()))
        return decoder_params

    def state_parameters(self):
        return self.latent.state_parameters()

    def state_gradients(self):
        return self.latent.state_gradients()


class ConvLatentLevel(object):
    pass


class RecurrentLatentLevel(object):

    def __init__(self, batch_size, encoder_arch, decoder_arch, n_latent, n_det, encoding_form, const_prior_var,
                 variable_update_form, posterior_form='gaussian', learn_prior=True):

        self.batch_size = batch_size
        self.n_latent = n_latent
        self.encoding_form = encoding_form

        self.encoder = MultiLayerRecurrent(**encoder_arch)
        self.decoder = MultiLayerPerceptron(**decoder_arch)

        variable_input_sizes = (encoder_arch['n_units'], decoder_arch['n_units'])

        self.latent = DenseGaussianVariable(self.batch_size, self.n_latent, const_prior_var, variable_input_sizes,
                                            variable_update_form, posterior_form, learn_prior)
        self.deterministic_encoder = Dense(variable_input_sizes[0], n_det[0]) if n_det[0] > 0 else None
        self.deterministic_decoder = Dense(variable_input_sizes[1], n_det[1]) if n_det[1] > 0 else None

    def get_encoding(self, input, in_out):
        encoding = input if in_out == 'in' else None
        if 'posterior' in self.encoding_form and in_out == 'out':
            encoding = input
        if ('top_error' in self.encoding_form and in_out == 'in') or ('bottom_error' in self.encoding_form and in_out == 'out'):
            error = self.latent.error()
            encoding = error if encoding is None else torch.cat((encoding, error), 1)
        if ('top_norm_error' in self.encoding_form and in_out == 'in') or ('bottom_norm_error' in self.encoding_form and in_out == 'out'):
            norm_error = self.latent.norm_error()
            encoding = norm_error if encoding is None else torch.cat((encoding, norm_error), 1)
        if 'gradient' in self.encoding_form and in_out == 'in':
            pass
        return encoding

    def encode(self, input):
        encoded = self.encoder(self.get_encoding(input, 'in'))
        output = self.get_encoding(self.latent.encode(encoded), 'out')
        if self.deterministic_encoder:
            det = self.deterministic_encoder(encoded)
            output = torch.cat((det, output), 1)
        return output

    def decode(self, input, generate=False):
        # decode the input, sample the latent variable, concatenate any deterministic units
        decoded = self.decoder(input)
        sample = self.latent.decode(decoded, generate=generate)
        if self.deterministic_decoder:
            det = self.deterministic_decoder(decoded)
            sample = torch.cat((sample, det), 1)
        return sample

    def kl_divergence(self):
        return self.latent.kl_divergence()

    def reset(self, from_prior=True):
        self.latent.reset(from_prior)
        self.encoder.reset()

    def trainable_state(self):
        self.latent.trainable_mean()
        self.latent.trainable_log_var()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.latent.eval()

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.latent.train()

    def cuda(self, device_id=0):
        # place all modules on the GPU
        self.encoder.cuda(device_id)
        self.decoder.cuda(device_id)
        self.latent.cuda(device_id)
        if self.deterministic_encoder:
            self.deterministic_encoder.cuda(device_id)
        if self.deterministic_decoder:
            self.deterministic_decoder.cuda(device_id)

    def parameters(self):
        return self.encoder_parameters() + self.decoder_parameters()

    def encoder_parameters(self):
        encoder_params = []
        encoder_params.extend(list(self.encoder.parameters()))
        if self.deterministic_encoder:
            encoder_params.extend(list(self.deterministic_encoder.parameters()))
        encoder_params.extend(list(self.latent.encoder_parameters()))
        return encoder_params

    def decoder_parameters(self):
        decoder_params = []
        decoder_params.extend(list(self.decoder.parameters()))
        if self.deterministic_decoder:
            decoder_params.extend(list(self.deterministic_decoder.parameters()))
        decoder_params.extend(list(self.latent.decoder_parameters()))
        return decoder_params

    def state_parameters(self):
        return self.latent.state_parameters()
