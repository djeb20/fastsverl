import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np

class NN(nn.Module):
    """
    Base class for building generic MLP neural networks.
    """
    def __init__(self):
        super().__init__()

    def build_block(self, **kwargs):
        """
        Builds a block of layers.
        kwargs: dictionary defining the architecture.
        Each value is a list where the first element is the component type
        (e.g., 'Linear', 'ReLU') and the second element is a list of component 
        parameters (e.g., [in_features, out_features] for Linear).
        """

        layers = []

        for _, (component, value) in kwargs.items():
            if component == 'layer_init':
                self.layer_init(layers[-1], *value)
            elif hasattr(nn, component):
                layers.append(getattr(nn, component)(*value))
            else:
                raise ValueError(f'Network component: {component} not recognised.')
            
            # For debugging size problems.
            # layers.append(DebugLayer(f"After {component}"))

        return nn.Sequential(*layers)
    
    def layer_init(self, layer, std, bias_const):
        """Initializes the layer's weights and biases."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
        
class SimpleNN(NN):
    """
    Simple feedforward neural network.
    """
    def __init__(self, name, output_shape, **kwargs):
        super().__init__()

        # Adjust output layer size (flattened)
        kwargs['output'][1][1] = np.prod(output_shape)

        # Build the model
        self.model = self.build_block(**kwargs)

        # Named access to output
        self.outputTuple = namedtuple('OutputTuple', [name])
        
        # Store output shape
        self.output_shape = output_shape

    def forward(self, *args):
        """
        Forward pass.
        Reshapes output to match first dimensions of input batch 
        and given output shape, aligning outputs for Shapley models.
        e.g. (batch_size, num_features, num_actions) for behaviour Shapley.
        """
        out = self.model(torch.cat(args, dim=-1)).view(args[0].shape[:-1] + self.output_shape)
        return self.outputTuple(out)

class MultiHeadNN(NN):
    """
    Multi-head neural network.
    For combining agent, characteristics, and Shapley networks.
    """
    def __init__(self, shared_layers, heads, heads_output_shapes):
        """
        shared_layers: a dictionary of the shared layers' architecture
        heads: dictionary with keys as head names and values as head architecture dictionaries
        heads_output_shapes: list of output shapes for each head  
        """
        super().__init__()

        # Common hidden layers
        self.common_layers = self.build_block(**shared_layers)

        # Heads
        for head_arch, output_shape in zip(heads.values(), heads_output_shapes):
            head_arch['output'][1][1] = np.prod(output_shape)
        self.heads = nn.ModuleList([self.build_block(**head_arch) for head_arch in heads.values()])

        # Output shapes
        self.output_shapes = heads_output_shapes

        # Named access to head outputs
        self.outputTuple = namedtuple('OutputTuple', [name for name in heads])

    def forward(self, *args):

        # Pass input through common layers
        x = self.common_layers(torch.cat(args, dim=-1))

        # Get outputs from each head, reshaped following SimpleNN forward logic
        output = [
            head(x).view(args[0].shape[:-1] + output_shape) 
            for head, output_shape in zip(self.heads, self.output_shapes)
            ]
        return self.outputTuple(*output)

class DebugLayer(nn.Module):
    """
    Debugging layer to print the shape of the input tensor.
    """
    def __init__(self, name):
        super(DebugLayer, self).__init__()
        self.name = name

    def forward(self, x):
        print(f"{self.name}: {x.shape}")
        return x
    
class ImportanceWeightedMSELoss(nn.Module):
    """
    Computes the importance-weighted mean squared error loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, weights):
        # Ensure the weights are the same shape as predictions and targets
        loss = (weights.view(-1, *[1] * (predictions.dim() - 1)) * (predictions - targets) ** 2).mean()
        return loss