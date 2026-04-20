#### Download the OBO file system (http://purl.obolibrary.org/obo/go/go-basic.obo):

```bash
pip install obonet networkx
python go_basic_download.py.py
```
--------------
#### Generate the Multilabels Target for the PDB IDs:
For checking, you can look into the individual GO terms DAG (eg. https://www.ebi.ac.uk/QuickGO/term/GO:0016787) and see if it and its parents are 0/1.

```bash
python create_GO_labels.py
```

-------------
#### Generate the IA weights for the Fmax criteria
For checking, you can use the train_go_labels.csv and check some GO labels and their parents and calculate IA manually (eg. GO:0004252)

```bash
python ia_weights.py
```
-------------
#### CAFA 5 Evaluation Metric: https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/405237
