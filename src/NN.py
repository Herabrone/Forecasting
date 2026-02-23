import einops
import numpy as np
import torch


#NOTE: This class was taken from the repo in our paper
#via https://github.com/LoryPack/GenerativeNetworksScoringRulesProbabilisticForecasting 
def createGenerativeFCNN(input_size, output_size, hidden_sizes=None, nonlinearity=None):
    """Function returning a fully connected neural network class with a given input and output size, and optionally
    given hidden layer sizes (if these are not given, they are determined from the input and output size with some
    expression. With respect to the one above, the NN here has two inputs,
    which are the context x and the auxiliary variable z in the case of this being used for a generative model.

    In order to instantiate the network, you need to write: createGenerativeFCNN(input_size, output_size)() as the function
    returns a class, and () is needed to instantiate an object.

    Note that the nonlinearity here is as an object or a functional, not a class, eg:
        nonlinearity =  nn.Softplus()
    or:
        nonlinearity =  nn.functional.softplus
    """

    class GenerativeFCNN(nn.Module):
        """Neural network class with sizes determined by the upper level variables."""

        def __init__(self):
            super(GenerativeFCNN, self).__init__()
            # put some fully connected layers:

            if hidden_sizes is not None and len(hidden_sizes) == 0:
                # it is effectively a linear network
                self.fc_in = nn.Linear(input_size, output_size)

            else:
                if hidden_sizes is None:
                    # then set some default values for the hidden layers sizes; is this parametrization reasonable?
                    hidden_sizes_list = [int(input_size * 1.5), int(input_size * 0.75 + output_size * 3),
                                         int(output_size * 5)]

                else:
                    hidden_sizes_list = hidden_sizes

                self.fc_in = nn.Linear(input_size, hidden_sizes_list[0])

                # define now the hidden layers
                self.fc_hidden = nn.ModuleList()
                for i in range(len(hidden_sizes_list) - 1):
                    self.fc_hidden.append(nn.Linear(hidden_sizes_list[i], hidden_sizes_list[i + 1]))
                self.fc_out = nn.Linear(hidden_sizes_list[-1], output_size)

            self.nonlinearity_fcn = F.relu if nonlinearity is None else nonlinearity

        def forward(self, context: TensorType["batch_size", "window_size", "data_size"],
                    z: TensorType["batch_size", "number_generations", "size_auxiliary_variable"]) \
                -> TensorType["batch_size", "number_generations", "output_size"]:
            # this network just flattens the context and concatenates it to the auxiliary variable, for each possible
            # auxiliary variable for each batch element, and then uses a FCNN.
            # input size of the FCNN must be equal to window_size * data_size + size_auxiliary_variable

            batch_size, window_size, data_size = context.shape
            batch_size, number_generations, size_auxiliary_variable = z.shape

            # can add argument `output_size=batch_size * number_generations` with torch 1.10
            repeated_context = torch.repeat_interleave(context, repeats=number_generations, dim=0).reshape(
                batch_size, number_generations, window_size * data_size)

            input_tensor = torch.cat((repeated_context, z), dim=-1)

            if not hasattr(self,
                           "fc_hidden"):  # it means that hidden sizes was provided and the length of the list was 0
                input_tensor = self.fc_in(input_tensor)
                return input_tensor

            input_tensor = self.nonlinearity_fcn(self.fc_in(input_tensor))
            for i in range(len(self.fc_hidden)):
                input_tensor = self.nonlinearity_fcn(self.fc_hidden[i](input_tensor))

            return self.fc_out(input_tensor)

    return GenerativeFCNN

#NOTE: This class was taken from the repo in our paper
#via https://github.com/LoryPack/GenerativeNetworksScoringRulesProbabilisticForecasting 
class ConditionalGenerativeModel(nn.Module):
    """This is a class wrapping a net which takes as an input a conditioning variable and an auxiliary variable,
    concatenates then and then feeds them through the NN. The auxiliary variable generation is done whenever the
    forward method is called."""

    def __init__(self, net, size_auxiliary_variable: Union[int, torch.Size], number_generations_per_forward_call: int,
                 seed: int = 42, base_measure: str = "normal"):
        super(ConditionalGenerativeModel, self).__init__()
        self.net = net  # net has to be able to take input size `size_auxiliary_variable + dim(x)`
        if isinstance(size_auxiliary_variable, int):
            size_auxiliary_variable = torch.Size([size_auxiliary_variable])
        self.size_auxiliary_variable = size_auxiliary_variable
        self.number_generations_per_forward_call = number_generations_per_forward_call
        # set the seed of the random number generator for ensuring reproducibility:
        torch.random.manual_seed(seed)
        distribution_dict = {"normal": normal.Normal, "laplace": laplace.Laplace, "cauchy": cauchy.Cauchy}
        if base_measure not in distribution_dict.keys():
            raise NotImplementedError("Base measure not available")
        self.distribution = distribution_dict[base_measure](loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))

    def forward(self, context: Union[TensorType["batch_size", "window_size", "data_size"], TensorType[
        "batch_size", "window_size", "height", "width", "fields"]],
                number_generations: int = None) -> Union[
        TensorType["batch_size", "number_generations", "data_size"], TensorType[
            "batch_size", "number_generations", "height", "width", "fields"]]:
        """This returns a stacked torch tensor for outputs. For now, I use different random auxiliary variables for
            each element in the batch size, but in principle you could take the same."""

        # you basically need to generate self.number_generations_per_forward_call forward simulations with a different
        # auxiliary variable z, all with the same conditioning variable x, for each conditioning variable x:
        if number_generations is None:
            number_generations = self.number_generations_per_forward_call

        if context.ndim == 3:
            batch_size, window_size, data_size = context.shape
        elif context.ndim == 5:
            batch_size, window_size, height, width, n_fields = context.shape
            # assume that the window_size is 1 for the moment now:
            if window_size != 1:
                raise NotImplementedError("We do not yet implement UNet for observation windows larger than 1")
            context = context.squeeze(1)
        else:
            raise NotImplementedError

        # generate the auxiliary variables; use different noise for each batch element
        z = self.distribution.sample(torch.Size([batch_size, number_generations]) + self.size_auxiliary_variable).to(
            device="cuda" if next(self.parameters()).is_cuda else "cpu").squeeze(-1)

        return self.net(context, z)

#NOTE: This method was taken from the repo in our paper
#via https://github.com/LoryPack/GenerativeNetworksScoringRulesProbabilisticForecasting 
def createGenerativeGRUNN(data_size, gru_hidden_size, noise_size, output_size, hidden_sizes=None, gru_layers=3,
                          nonlinearity=None):
    """Function returning a recurrent neural network with a given input and output size, and optionally
    given hidden layer sizes (if these are not given, they are determined from the input and output size with some
    expression). The NN here has two inputs,
    which are the context x and the auxiliary variable z in the case of this being used for a generative model.

    In order to instantiate the network, you need to write: createGenerativeGRUNN(input_size, output_size)() as the function
    returns a class, and () is needed to instantiate an object.

    Note that the nonlinearity here is as an object or a functional, not a class, eg:
        nonlinearity =  nn.Softplus()
    or:
        nonlinearity =  nn.functional.softplus
    """

    class GenerativeGRUNN(nn.Module):
        """Neural network class with sizes determined by the upper level variables."""

        def __init__(self):
            super(GenerativeGRUNN, self).__init__()

            # GRU layer:
            self.gru = nn.GRU(data_size, gru_hidden_size, gru_layers, batch_first=True)

            # put some fully connected layers after the gru
            fc_in_size = gru_hidden_size + noise_size

            # instantiate a generativeFCNN:
            self.fc_nn = createGenerativeFCNN(fc_in_size, output_size, hidden_sizes, nonlinearity)()

        def forward(self, context: TensorType["batch_size", "window_size", "data_size"],
                    z: TensorType["batch_size", "number_generations", "size_auxiliary_variable"]) \
                -> TensorType["batch_size", "number_generations", "output_size"]:
            gru_out, _ = self.gru(context)

            # this has shape [batch_size, window_size, gru_hidden_size]; take only the last temporal element:
            gru_out = gru_out[:, -1, :].unsqueeze(1)

            # now apply the FC generative net:
            out = self.fc_nn(gru_out, z)
            return out

    return GenerativeGRUNN


def ensemble_nll_loss(ensemble_preds: Tensor, targets: Tensor) -> Tensor:
    """
    Parameters:
        ensemble_preds : Tensor
        Shape (batch_size, number_generations, 1) – the collection of
        sampled forecasts produced by the generative model for each item in
        the batch.  Each “generation” is one stochastic draw, so the
        second dimension is the ensemble axis.

    targets : Tensor
        Shape `(batch_size, 1)` – the true next‑step sales value for each
        element of the batch.
    

    """
