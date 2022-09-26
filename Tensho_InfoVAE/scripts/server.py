from fastapi import FastAPI
# from txt2img import  get_grid_image, load_engines
from argparse import Namespace
# from GenerationBatch import T2IGenerationBatch, get_images
from pydantic import BaseModel
import io
# from fastapi.responses import StreamingResponse
from fastapi.responses import StreamingResponse, FileResponse
import numpy as np
from torch import nn
from torchvision import transforms
import torch
from PIL import Image

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:4000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

z_dim = 3
class ChannelsToLinear(nn.Linear):
    """Flatten a Variable to 2d and apply Linear layer"""
    def forward(self, x):
        b = x.size(0)
        return super().forward(x.view(b,-1))
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        n_filters = 64
        self.conv1 = nn.Conv2d(1, n_filters, 4,2,1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 4,2,1)
        self.conv3 = nn.Conv2d(n_filters, n_filters*2, 4, 2,1)
        self.conv4 = nn.Conv2d(n_filters*2, n_filters*4, 4, 2,1)

        self.toLinear1 =  ChannelsToLinear(n_filters*16, 1024)
        # self.toLinear1 =  ChannelsToLinear(n_filters*2*7*7, 1024)
        self.fc1 = nn.Linear(1024,z_dim)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        
    def forward(self,x):
        h1 = self.lrelu(self.conv1(x))
        h2 = self.lrelu(self.conv2(h1))
        h3 = self.lrelu(self.conv3(h2))
        h4 = self.lrelu(self.conv4(h3))
        h5 = self.lrelu(self.toLinear1(h4))
        h5 = self.fc1(h5)
        
        return h5

class LinearToChannels2d(nn.Linear):
    """Reshape 2d Variable to 4d after Linear layer"""
    def __init__(self, m, n, w=1, h=None, **kw):
        h = h or w
        super().__init__(m, n*w*h, **kw)
        self.w = w
        self.h = h
    def forward(self, x):
        b = x.size(0)
        return super().forward(x).view(b, -1, self.w, self.h)
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        n_filters = 64
        img_size = 32
        last_layer_channel_count = n_filters
        window_size = 2
        layer_count = 4
        # assert img_size**2 * 2**(layer_count-1) == last_layer_channel_count * window_size**2 , (img_size**2 * 2**(layer_count-1), first_layer_channel_count * window_size**2 )
        # なんらかの等式が成り立っていることは間違いない
        # img_size**2 = first_layer_channel_count * window_size**2 * 8
        # assert img_size == window_size*8

        self.fc1 = nn.Linear(z_dim,1024)
        # self.LineartoChannel = LinearToChannels2d(1024,n_filters*2,7,7)
        self.LineartoChannel = LinearToChannels2d(1024, n_filters*4, window_size, window_size)
        self.conv1 = nn.ConvTranspose2d(n_filters*4, n_filters*2,4,2,1)
        self.conv2 = nn.ConvTranspose2d(n_filters*2,n_filters,4,2,1)
        self.conv3 = nn.ConvTranspose2d(n_filters, n_filters,4,2,1)
        self.conv4 = nn.ConvTranspose2d(n_filters, 1,4,2,1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,z):
        h1 = self.relu(self.fc1(z))
        h2 = self.relu(self.LineartoChannel(h1))
        
        h3 = self.relu(self.conv1(h2))
        h4 = self.relu(self.conv2(h3))
        h5 = self.relu(self.conv3(h4))
        h6 = self.sigmoid(self.conv4(h5))
        
        return h6

encodermodel = Encoder()
encodermodel.load_state_dict(torch.load("/Users/life_mac_43/Projects/MediaDesign/MMD-Variational-Autoencoder-Pytorch-InfoVAE/encodermodel_epoch80_loss0.10727991163730621.pth"))

decodermodel = Decoder()
decodermodel.load_state_dict(torch.load("/Users/life_mac_43/Projects/MediaDesign/MMD-Variational-Autoencoder-Pytorch-InfoVAE/decodermodel_epoch80_loss0.10727991163730621.pth"))


class ChannelsToLinear(nn.Linear):
    """Flatten a Variable to 2d and apply Linear layer"""
    def forward(self, x):
        b = x.size(0)
        return super().forward(x.view(b,-1))
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        n_filters = 64
        self.conv1 = nn.Conv2d(1, n_filters, 4,2,1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 4,2,1)
        self.conv3 = nn.Conv2d(n_filters, n_filters*2, 4, 2,1)
        self.conv4 = nn.Conv2d(n_filters*2, n_filters*4, 4, 2,1)

        self.toLinear1 =  ChannelsToLinear(n_filters*16, 1024)
        # self.toLinear1 =  ChannelsToLinear(n_filters*2*7*7, 1024)
        self.fc1 = nn.Linear(1024,z_dim)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        
    def forward(self,x):
        h1 = self.lrelu(self.conv1(x))
        h2 = self.lrelu(self.conv2(h1))
        h3 = self.lrelu(self.conv3(h2))
        h4 = self.lrelu(self.conv4(h3))
        h5 = self.lrelu(self.toLinear1(h4))
        h5 = self.fc1(h5)
        
        return h5

class Item(BaseModel):
    pixels: str


@app.get("/")
async def root():
    return FileResponse("scripts/index.html")

# @app.post("/img/{prompt}")
# async def get_img(prompt: str):
#     # img = generator.generate(prompt)[0]
#     img_byte_arr = io.BytesIO()
#     img.save(img_byte_arr, format='PNG')
#     img_byte_arr.seek(0)
#     return StreamingResponse(img_byte_arr, media_type="image/png")


@app.post("/api/predict")
async def root(item: Item):
    input_img = [*map(int, item.pixels.split(","))]
    data = []
    for  i, d in enumerate(input_img):
        if i % 4 == 0:
            data.append(d)
    print(len(data))
    input_img = data
    # print(input_img)
    input_img = np.array(input_img).reshape(32, 32)
    z = encodermodel(torch.from_numpy(input_img).float().reshape(1, 1, 32, 32))
    output_img = decodermodel(z)
    print(output_img.shape)
    img = transforms.ToPILImage()((255 - output_img).reshape(32, 32))
    img = img.convert("L")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='BMP')
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type="image/bmp")
    

