import torch
import torch.nn as nn


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        # x should have shape (batch_size, sequence_length, num_features)

        b, seq_len, num_features = x.size()

        n = seq_len - 1

        x_minus_mu_square = (x - x.mean(dim=1, keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=1, keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activation(y)


# Example usage:
# Define a time series model and apply SimAM to it
class TimeSeriesModel(nn.Module):
    def __init__(self):
        super(TimeSeriesModel, self).__init__()
        self.simam = SimAM()

    def forward(self, x):
        # You can add your time series prediction model here
        # Apply SimAM to the input x
        x = self.simam(x)
        return x



