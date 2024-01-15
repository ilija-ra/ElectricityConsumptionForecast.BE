from re import VERBOSE


EPOCH_NUMBER = 100
BATCH_SIZE_NUMBER = 1
COST_FUNCTION = 'mean_squared_error'
OPTIMIZER = 'adam'
KERNEL_INITIALIZER = 'normal'
ACTIVATION_FUNCTION = 'sigmoid'
NUMBER_OF_HIDDEN_LAYERS = 2
NUMBER_OF_NEURONS_IN_FIRST_HIDDEN_LAYER = 4
NUMBER_OF_NEURONS_IN_OTHER_HIDDEN_LAYERS = 2
VERBOSE = 2

class AnnBase:

    def __init__(self):
        self.epoch_number = EPOCH_NUMBER
        self.batch_size_number = BATCH_SIZE_NUMBER
        self.cost_function = COST_FUNCTION
        self.optimizer = OPTIMIZER
        self.kernel_initializer = KERNEL_INITIALIZER
        self.activation_function = ACTIVATION_FUNCTION        
        self.number_of_hidden_layers = NUMBER_OF_HIDDEN_LAYERS
        self.number_of_neurons_in_first_hidden_layer = 20
        self.number_of_neurons_in_other_hidden_layers = 8
        self.verbose = VERBOSE

    @property
    def epoch_number(self):
        return self._epoch_number

    @epoch_number.setter
    def epoch_number(self, value):
        self._epoch_number = value

    @property
    def batch_size_number(self):
        return self._batch_size_number

    @batch_size_number.setter
    def batch_size_number(self, value):
        self._batch_size_number = value
    
    #The purpose of loss functions is to compute the quantity that a model should seek to minimize during training.
    #mean_squared_error
    #mean_absolute_error
    #mean_absolute_percentage_error
    #mean_squared_logarithmic_error
    #cosine_similarity
    #huber_loss
    #mean_squared_logarithmic_error
    @property
    def cost_function(self):
        return self._cost_function

    @cost_function.setter
    def cost_function(self, value):
        self._cost_function = value

    #SGD
    #RMSprop
    #adam
    #adadelta
    #adagrad
    #adamax
    #nadam
    #ftrl
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
    #Initializers define the way to set the initial random weights of Keras layers.
    #random_normal
    #normal
    #random_uniform
    #zeros
    #ones
    #glorot_normal
    #glorot_uniform
    #he_normal
    #he_uniform
    #identity
    @property
    def kernel_initializer(self):
        return self._kernel_initializer

    @kernel_initializer.setter
    def kernel_initializer(self, value):
        self._kernel_initializer = value

    @property
    def activation_function(self):
        return self._activation_function

    #relu 
    #sigmoid
    #softmax
    #softplus 
    #softsign
    #tanh 
    #selu
    #elu
    #exponential

    @activation_function.setter
    def activation_function(self, value):
        self._activation_function = value

    @property
    def number_of_hidden_layers(self):
        return self._number_of_hidden_layers

    @number_of_hidden_layers.setter
    def number_of_hidden_layers(self, value):
        self._number_of_hidden_layers = value

    @property
    def number_of_neurons_in_first_hidden_layer(self):
        return self._number_of_neurons_in_first_hidden_layer

    @number_of_neurons_in_first_hidden_layer.setter
    def number_of_neurons_in_first_hidden_layer(self, value):
        self._number_of_neurons_in_first_hidden_layer = value

    @property
    def number_of_neurons_in_other_hidden_layers(self):
        return self._number_of_neurons_in_other_hidden_layers

    @number_of_neurons_in_other_hidden_layers.setter
    def number_of_neurons_in_other_hidden_layers(self, value):
        self._number_of_neurons_in_other_hidden_layers = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value