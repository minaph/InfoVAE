import torch
from torch import nn
from torch.nn import functional as F

from .types_ import *
from .base import BaseVAE


class TensorShapeTracker:
    def __init__(self, initial_shape: tuple[int, int, int]):
        self.shape = initial_shape
        self.history: list[tuple[str, tuple[int, int, int]]] = []
        self.history.append(("initial", initial_shape))
        self.next_channels = initial_shape[0]

    def Conv2d(
        self,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
    ):
        self.next_channels = out_channels

        padding_0, padding_1 = padding
        stride_0, stride_1 = stride
        dilation_0, dilation_1 = dilation
        kernel_size_0, kernel_size_1 = kernel_size

        _, H_in, W_in = self.shape

        H_out = int(
            (H_in + 2 * padding_0 - dilation_0 * (kernel_size_0 - 1) - 1) / stride_0 + 1
        )
        W_out = int(
            (W_in + 2 * padding_1 - dilation_1 * (kernel_size_1 - 1) - 1) / stride_1 + 1
        )
        self.shape = (out_channels, H_out, W_out)
        self.history.append(("Conv2d", self.shape))

    def ConvTranspose2d(
        self,
        out_channels: int,
        kernel_size: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = 1,
        padding: tuple[int, int] = (0, 0),
        output_padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
    ):
        self.next_channels = out_channels

        stride_0, stride_1 = stride
        padding_0, padding_1 = padding
        dilation_0, dilation_1 = dilation
        kernel_size_0, kernel_size_1 = kernel_size
        output_padding_0, output_padding_1 = output_padding

        _, H_in, W_in = self.shape

        H_out = (
            (H_in - 1) * stride_0
            - 2 * padding_0
            + dilation_0 * (kernel_size_0 - 1)
            + output_padding_0
            + 1
        )
        W_out = (
            (W_in - 1) * stride_1
            - 2 * padding_1
            + dilation_1 * (kernel_size_1 - 1)
            + output_padding_1
            + 1
        )

        self.shape = (out_channels, H_out, W_out)
        self.history.append(("ConvTranspose2d", self.shape))

    def Linear(self, out_features: int):
        self.next_channels = out_features
        _, H_in, W_in = self.shape
        self.shape = (out_features, H_in, W_in)
        self.history.append(("Linear", self.shape))

    def Flatten(self):
        self.shape = (self.shape[0] * self.shape[1] * self.shape[2], 1, 1)
        self.next_channels = self.shape[0]
        self.history.append(("Flatten", self.shape))

    def reshape(self, c, h, w):
        self.shape = (c, h, w)
        self.next_channels = c
        self.history.append(("reshape", self.shape))

    @property
    def current_size(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    def log(self):
        print("Shape history:")
        for i, (action, shape) in enumerate(self.history):
            print(f"{i} {action}: {shape} = {shape[0] * shape[1] * shape[2]}")


class InfoVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        alpha: float = -0.5,
        beta: float = 5.0,
        reg_weight: int = 100,
        kernel_type: str = "imq",
        latent_var: float = 2.0,
        **kwargs,
    ) -> None:
        super(InfoVAE, self).__init__()

        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var

        assert alpha <= 0, "alpha must be negative or zero."

        self.alpha = alpha
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        H_in = W_in = 64
        tracker = TensorShapeTracker((in_channels, H_in, W_in))

        try:
            # Build Encoder
            for h_dim in hidden_dims:
                tracker.Conv2d(h_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

                modules.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            out_channels=h_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU(),
                    )
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            self.encoded_shape = tracker.shape
            encoded_size = tracker.current_size
            transformed_size = encoded_size

            tracker.Flatten()

            transform_size_list = []
            transform_modules = []
            while transformed_size > latent_dim:
                next_transformed_size = int(transformed_size / 4)
                transform_size_list.append(transformed_size)
                tracker.Linear(transformed_size)
                transform_modules.append(
                    nn.Sequential(
                        nn.Linear(transformed_size, next_transformed_size),
                        nn.BatchNorm1d(next_transformed_size),
                        nn.LeakyReLU(),
                    )
                )
                transformed_size = next_transformed_size

            self.fc_transform = nn.Sequential(*transform_modules)

            tracker.Linear(latent_dim)
            self.randn_shape = tracker.shape
            self.fc_mu = nn.Linear(transformed_size, latent_dim)
            self.fc_var = nn.Linear(transformed_size, latent_dim)

            # Build Decoder
            inv_transform_modules = []
            prev_transformed_size = latent_dim
            for transformed_size in reversed(transform_size_list):
                inv_transform_modules.append(
                    nn.Sequential(
                        nn.Linear(prev_transformed_size, transformed_size),
                        nn.BatchNorm1d(transformed_size),
                        nn.LeakyReLU(),
                    )
                )
                tracker.Linear(transformed_size)
                prev_transformed_size = transformed_size

            self.fc_transform_inv = nn.Sequential(*inv_transform_modules)

            modules = []
            tracker.Linear(encoded_size)
            self.decoder_input = nn.Linear(transformed_size, encoded_size)

            tracker.reshape(*self.encoded_shape)
            hidden_dims = hidden_dims[::-1]

            for i in range(len(hidden_dims) - 1):
                tracker.ConvTranspose2d(
                    hidden_dims[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    output_padding=(1, 1),
                )

                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            hidden_dims[i],
                            hidden_dims[i + 1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU(),
                    )
                )

            self.decoder = nn.Sequential(*modules)
            tracker.ConvTranspose2d(
                hidden_dims[-1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1),
            )
            tracker.Conv2d(
                out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(
                    hidden_dims[-1], out_channels=out_channels, kernel_size=3, padding=1
                ),
                nn.Tanh(),
            )
        except Exception as e:
            tracker.log()
            raise e

        tracker.log()

    @property
    def randn(self):
        result = self._randn[self._randn_i]
        self._randn_i += 1
        return result

    def set_randn(self, iteration: int, batch_size: int, device):
        self._randn_i = 0
        randn_shape = (iteration, batch_size, 3)
        self._randn = torch.randn(randn_shape, device=device)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.fc_transform(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result).clamp(torch.finfo().min, torch.finfo().max)

        if torch.is_anomaly_enabled():
            print("log_var", log_var.min().data.item(), log_var.max().data.item())
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        transformed = self.fc_transform_inv(z)
        result = self.decoder_input(transformed)
        result = result.view(-1, *self.encoded_shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=mu.device)
        # print("shapes: ", mu.shape, std.shape, eps.shape)
        r = eps * std + mu
        return r

    def _reparameterize_training(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = self.randn
        # print(f"std: shape {std.shape} dtype {std.dtype} layout {std.layout}, \neps: shape {eps.shape} dtype {eps.dtype} layout {eps.layout}")
        r = eps * std + mu
        return r

    def forward(self, input: Tensor, training: bool, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = (
            self._reparameterize_training(mu, log_var)
            if training
            else self.reparameterize(mu, log_var)
        )
        # z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, z, mu, log_var]

    # def training_step(self, input):
    #     mu, log_var = self.encode(input)
    #     z = self._reparameterize_training(mu, log_var)
    #     return [self.decode(z), input, z, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]
        # mu = args[3]
        # log_var = args[4]

        # batch_size = input.size(0)
        # bias_corr = batch_size * (batch_size - 1)
        # kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)
        mmd_loss = self.compute_mmd(z)
        # kld_loss = torch.mean(
        #     -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        # )

        # loss = (
        #     self.beta * recons_loss
        #     + (1.0 - self.alpha) * kld_weight * kld_loss
        #     + (self.alpha + self.reg_weight - 1.0) / bias_corr * mmd_loss
        # )
        loss = recons_loss + self.reg_weight * mmd_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "MMD": mmd_loss,
            # "KLD": kld_loss,
        }

    def compute_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == "rbf":
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == "imq":
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError("Undefined kernel type.")

        return result

    def compute_rbf(self, x1: Tensor, x2: Tensor, eps: float = 1e-7) -> Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2.0 * z_dim * self.z_var
        to_exp = (-((x1 - x2).pow(2).mean(-1) / sigma)).clamp(
            torch.finfo().min, torch.finfo().max
        )
        if torch.is_anomaly_enabled():
            print("compute_rbf", to_exp.min().data.item(), to_exp.max().data.item())
        result = torch.exp(to_exp)
        return result

    def compute_inv_mult_quad(
        self, x1: Tensor, x2: Tensor, eps: float = 1e-7
    ) -> Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z: Tensor) -> Tensor:
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z, device=z.device)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + z__kernel.mean() - 2 * priorz_z__kernel.mean()
        del prior_z__kernel, z__kernel, priorz_z__kernel, prior_z
        return mmd

    def sample(self, num_samples: int, current_device: "int | str", **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(
            num_samples, self.latent_dim, device=torch.device(current_device)
        )
        # z = z.to(torch.device(current_device))

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
