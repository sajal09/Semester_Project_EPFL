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


#### Download the `QM.hdf5` from Zenodo:
```
wget -O QM.hdf5 https://zenodo.org/records/7711953/files/QM.hdf5
```

---

#### Now generate the graphs and save them in disk:
Change data_type to 'train' in `save_data.py` and run:
```
python3 save_data.py
```
Change data_type to 'val' in `save_data.py` and run:
```
python3 save_data.py
```
Change data_type to 'test' in `save_data.py` and run:
```
python3 save_data.py
```

---

### Further Notes

##### Train, Val, Test Ids
Please refer to file `train_val_test_ids.py`

---

##### Official Test Ids
You can download the test data corresponding to core set (casp 2016) from here:
```
https://www.pdbbind-plus.org.cn/casf
```
You have to download the "The CASF-2016 benchmark package" and inside it, you will find the "coreset" folder which has the test set PDB ids (total 285).
Then, we only take the Ids which are present in Misato as well.

---

##### Origin of all_dataset_merged.csv
The all_dataset_merged.csv comes from the link below where you have to download "Index files":
```
https://www.pdbbind-plus.org.cn/download -> PDBbind v2020.R1 -> Index files
```
Inside it, you will find the Ki/Kd values of various PDB ids. You can then take -log(Ki) or -log(Kd) (take log after converting it to Molars) to get the Target Binding Affinity.

---

##### Origin of atom_classes.pickle
You can get atom_classes.pickle from the Misato-Affinity github repo:
```
https://github.com/kierandidi/misato-affinity/blob/main/data/atom_classes.pickle
```

