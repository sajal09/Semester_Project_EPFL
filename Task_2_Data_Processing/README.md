### Follow the steps below:

#### Install other packages and Pytraj (https://amber-md.github.io/pytraj/latest/installation.html) in your venv 
```
bash install.sh
```

#### Download the Restart files from Zenodo
```
wget https://zenodo.org/records/7711953/files/parameter_restart_files_MD.tar.gz
```

#### Then decompress it using:
```
tar -xvzf parameter_restart_files_MD.tar.gz
```

#### Then decompress each individual production.top.gz inside each their respective protein folder:
```
chmod +x /home/chaurasi/semesterproject/processing_task_2/decompress_topology.sh

./decompress_topology.sh /scratch/izar/chaurasi/restart_files/parameter_restart_files_MD
```

#### Then create the traj files(.nc and .top) using h5_to_traj.py and restart files
```
python3

from h5_to_traj import create_traj_files

create_traj_files()
```

#### Then add 'atoms_residue_number' field in md_out file using add_atom_residue_number.py
```
python3 add_atom_residue_number.py
```

#### Then add ('residue_ids', 'residue_binding_labels') fields in md_out file using add_binding_site_labels.py
```
python3 add_binding_site_labels.py
```

#### Now generate the graphs and save them in disk:
Change data_type to 'val' in `save_data.py` and run:
```
python3 save_data.py
```
Change data_type to 'train' in `save_data.py` and run:
```
python3 save_data.py
```


