from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from modules.stnm import STNM
from modules.gridgen import AffineGridGen, CylinderGridGen, CylinderGridGenV2, DenseAffine3DGridGen, DenseAffine3DGridGen_rotate

import time

nframes = 64
height = 64
width = 128
channels = 64

inputImages = torch.zeros(nframes, height, width, channels)
grids = torch.zeros(nframes, height, width, 2)
canvas = torch.zeros(nframes, height, width, channels)
masks = torch.zeros(nframes, height, width, 1)
canvas, fgimg, fggrid, fgmask = Variable(canvas, requires_grad=True), Variable(inputImages, requires_grad=True), Variable(grids, requires_grad=True), Variable(masks, requires_grad=True)

s = STNM()
canvas = canvas.cuda()
fgimg = fgimg.cuda()
fggrid = fggrid.cuda()
fgmask = fgmask.cuda()

start = time.time()
out = s(canvas, fgimg, fggrid, fgmask)
print(out.size(), 'time:', time.time() - start)
start = time.time()
out.backward(canvas.data)
print('time:', time.time() - start)
