# INSTALLATIONS OF SOME PACKAGES
'''
pip install lenstronomy
pip install corner
pip install image-similarity-measures
'''

# import of standard python libraries
import numpy as np
import os
import time
import corner
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import copy
from random import *
import pandas as pd
import os.path
from PIL import Image
import matplotlib.image

# rgb 2 gray 
from skimage.color import rgb2gray

#import the lens_plot module
from lenstronomy.Plots import lens_plot
from lenstronomy.Plots.model_plot import ModelPlot

## import Lens Model used named EPL
from lenstronomy.LensModel.Profiles import epl

# import main simulation class of lenstronomy
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.ImSim.image_model import ImageModel


# lenstronomy utility functions
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.simulation_util as sim_util
# import the ShapeletSet class
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
from lenstronomy.LightModel.Profiles.shapelets_polar import ShapeletSetPolar
