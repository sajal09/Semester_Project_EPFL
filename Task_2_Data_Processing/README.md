## Follow the steps below:

### Install other packages and Pytraj (https://amber-md.github.io/pytraj/latest/installation.html) in your venv 
```
bash install.sh
```

### Download the Restart files from Zenodo
```
wget https://zenodo.org/records/7711953/files/parameter_restart_files_MD.tar.gz
```

### Then decompress it using:
```
tar -xvzf parameter_restart_files_MD.tar.gz
```

### Then decompress each individual production.top.gz inside each their respective protein folder:
```
chmod +x /home/chaurasi/semesterproject/processing_task_2/decompress_topology.sh

./decompress_topology.sh /scratch/izar/chaurasi/restart_files/parameter_restart_files_MD
```

