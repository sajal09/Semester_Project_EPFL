### Follow the steps below:

#### Download the `MD.hdf5` from Zenodo:
```
wget -O MD.hdf5 https://zenodo.org/record/7711953/files/MD.hdf5
```

#### Then, create the `md_out.hdf5` file using the script:
```
python3 md_out.py
```
**Note**: The misato-dataset/src/data/processing/preprocessing_db.py is modified such that it stores the Aligned Trajectory Coordinates instead of Unaligned Trajectory Coordinates.

#### Now generate the graphs and save them in disk:
Change data_type to 'train' in save_data.py and run:
```
python3 save_data.py
```
Change data_type to 'val' in save_data.py and run:
```
python3 save_data.py
```
Change data_type to 'test' in save_data.py and run:
```
python3 save_data.py
```

#### For Node Featurization,
* **For relative_distance_feature**: First get the mean and standard deviation and then create the feature:
```
python3

from gnn_utils import *

mean_imputation, actual_mean, actual_std = get_imputationmean_actualmean_actualstd_of_locantdistance_feature()

relative_distance_feature(mean_imputation, actual_mean, actual_std)
```

* **For nodedegree_feature**: First get the mean and standard deviation and then create the feature:
```
python3

from gnn_utils import *

mean, std = get_actualmean_actualstd_of_nodedegree_feature()

create_nodedegree_feature(mean, std, 'train')

create_nodedegree_feature(mean, std, 'val')

create_nodedegree_feature(mean, std, 'test')
```
