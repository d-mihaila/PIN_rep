import json
# from geolocation import *
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import csv
import re
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from pathlib import Path
from collections import defaultdict
from rasterio.transform import xy
from matplotlib import cm
from matplotlib.colors import Normalize


## here we read from worldpop
zip_dir = Path("/eos/jeodpp/home/users/mihadar/data/WorldPop")

# sex disaggregated and migration data
spatial_data_path = zip_dir / 'SexDisaggregated_Migration/SpatialData'
Centroids = spatial_data_path / 'Centroids.shp'
Flowlines_Internal = spatial_data_path / 'Flowlines_Internal.shp'
Flowlines_International = spatial_data_path / 'Flowlines_International.shp'

internal_migration_path = zip_dir / 'SexDisaggregated_Migration/MigrationEstimates/Metadata_MigrEst_internal_v4.txt'
international_migration_path = zip_dir / 'SexDisaggregated_Migration/MigrationEstimates/Metadata_MigrEst_international_v7.txt'

