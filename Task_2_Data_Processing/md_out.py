import sys
import os
sys.path.insert(0,os.path.join(os.path.abspath('').split('misato-dataset')[0],'misato-dataset/src/'))
sys.path.insert(0,os.path.join(os.path.abspath('').split('misato-dataset')[0],'misato-dataset/src/data/components/'))

from data.processing import preprocessing_db

class Args:
    # input file
    datasetIn = "/scratch/izar/chaurasi/data/MD.hdf5"#"misato-dataset/data/MD/h5_files/tiny_md.hdf5"
    # Feature that should be stripped, e.g. atoms_element or atoms_type
    strip_feature = "atoms_element"
    # Value to strip, e.g. if strip_freature= atoms_element; 1 for H.
    strip_value = 1
    # Start index of structures
    begin = 0
    # End index of structures
    end = -1
    # We calculate the adaptability for each atom.
    # Default behaviour will also strip H atoms, if no stripping should be perfomed set strip_value to -1.
    Adaptability = True
    # If set to True this will create a new feature that combines one entry for each protein AA but all ligand entries;
    # e.g. for only ca set strip_feature = atoms_type and strip_value = 14
    Pres_Lat = False
    # We strip the complex by given distance (in Angstrom) from COG of molecule,
    # use e.g. 15.0. If default value is given (0.0) no pocket stripping will be applied.
    Pocket = 0.0
    # output file name and location
    datasetOut = "/scratch/izar/chaurasi/data/md_out.hdf5"


args = Args()

preprocessing_db.main(args)