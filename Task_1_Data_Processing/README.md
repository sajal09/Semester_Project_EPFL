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
Change data_type to 'val' in save_data.py and run:
```
python3 save_data.py
```
Change data_type to 'train' in save_data.py and run:
```
python3 save_data.py
```

