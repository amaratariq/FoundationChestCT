import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class ConvUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvUpsampling, self).__init__()
        
        self.scale_factor = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        return self.conv(x)
    
class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D] (Bx64x16x16)->(Bx16x16x64)
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D] (B*16*16 x 64(D))

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss, encoding_inds, encoding_one_hot # [B x D x H x W]

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int, 
                d_rate: float):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.Dropout(p=d_rate),
                                      nn.BatchNorm2d(out_channels, track_running_stats=False), #############################################
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)

class ResidualLayerWODrop(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayerWODrop, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      #nn.Dropout(p=d_rate),
                                      nn.BatchNorm2d(out_channels, track_running_stats=False), #############################################
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)

    
class ResNextLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResNextLayer, self).__init__()
        
        self.resblock = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
                            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),     
                        )

    def forward(self, input: Tensor) -> Tensor:
        return nn.ReLU(inplace=True)(input + self.resblock(input))
    
class VQVAE_BN(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 drop_rates: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 up_sample = False,
                 depth=6,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        self.in_channels = in_channels
        self.up_sample = up_sample
        self.depth=depth
        
        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]
        if drop_rates is None:
            drop_rates = [0.0 for i in range(len(hidden_dims))]

        # Build Encoder
        for h_dim_idx in range(len(hidden_dims)):
            h_dim = hidden_dims[h_dim_idx]
            d_rate = drop_rates[h_dim_idx]
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim, track_running_stats=False), 
                    nn.Dropout(p=d_rate),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels, track_running_stats=False), 
                nn.Dropout(p=d_rate),
                nn.LeakyReLU())
        )

        for _ in range(self.depth):
            modules.append(ResidualLayer(in_channels, in_channels, 0.0))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.BatchNorm2d(embedding_dim, track_running_stats=False), 
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(hidden_dims[-1], track_running_stats=False), 
                nn.LeakyReLU())
        )

        for _ in range(self.depth):
            #modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1], drop_rates[-1]))
            modules.append(ResidualLayerWODrop(hidden_dims[-1], hidden_dims[-1]))
        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()
        drop_rates.reverse()
        if self.up_sample==False:
            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=4,
                                           stride=2,
                                           padding=1),
                        nn.BatchNorm2d(hidden_dims[i+1], track_running_stats=False), 
                        nn.LeakyReLU())
                )

            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[-1],
                                       out_channels=self.in_channels,#3,#1#use original number of in_channesl
                                       kernel_size=4,
                                       stride=2, padding=1),
                    nn.BatchNorm2d(self.in_channels, track_running_stats=False), 
                    nn.Tanh()))
        else:
            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        ConvUpsampling(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=4,
                                           stride=2,
                                           padding=1),
                        nn.BatchNorm2d(hidden_dims[i+1], track_running_stats=False), 
                        nn.LeakyReLU())
                )

            modules.append(
                nn.Sequential(
                    ConvUpsampling(hidden_dims[-1],
                                       out_channels=self.in_channels,#3,#1#use original number of in_channesl
                                       kernel_size=4,
                                       stride=2, padding=1),
                    nn.BatchNorm2d(self.in_channels, track_running_stats=False), 
                    nn.Tanh()))
            

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
#         print('Inside:', input.size())
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss, encoding_inds, encoding_one_hot  = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]