# DuGPro
A Deep Learning-Based Method for Protein Function Prediction by Fusing GO Knowledge and Protein Features

## Usage
Here we provide instructions for two use cases: (1) Retraining our model on our or your data. (2) Testing data on trained models.

If you encounter any bugs or issues, feel free to contact us.

## Dependencies
* The code was developed and tested using python 3.10.
* Clone the repository: https://github.com/CSUBioGroup/DuGPro.git
* Create virtual environment with Conda or python3-venv module.
* Install PyTorch, DGL and other requirements: ``` conda create --name new_env --file requirements.txt ```

## Prepare Data
### Using our dataset
* Download the [dataset.tar.gz](https://drive.google.com/uc?export=download&id=1t7bwxzmY1zF0IE0CnJp3igw7ISmZ_456)
* Extract ```tar xvzf dataset.tar.gz```

### Generating the data
If you like to generate a new dataset follow these steps:
* Download [Gene Ontology](https://geneontology.org/docs/download-ontology/).
You'll need go.obo file and save them into dataset folder.
* Download [Uniprot-KB](https://ftp.uniprot.org/pub/databases/uniprot/previous_releases)
You'll need three uniprot data by time stamps.
* Run ```python split_data.py``` to split data into train, test, valid sets.
* Run ```python generate_graph_data.py``` to get graph data.
* Download [esm2_t36_3B_UR50D](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt)
You'll need `esm2_t36_3B_UR50D.pt` file and save them into esm2 folder.
* Run ```python extract_esm2.py``` to generate esm2 features.

### Data Construction
All data required save in ```./dataset```
* `[cc|mf|bp]_direct_parents_pairs.pkl` and `[cc|mf|bp]_ancestors_pairs.pkl`  #graph data
* `[train|valid|test]_esm2.pkl`  #esm features
* select_min_count_1_[mf|bp|cc]_labels.csv  #all go terms
* `[train|valid|test]_data_separate.pkl` #proteins information including proteinID, sequence, selected_terms, all_terms...
* `[train|valid|test]_seq.fasta`  #protein sequence

Train/valid/test data of each ontology save in ```./dataset/[mf|bp|cc]```
* `[train|valid|test]_data_separate_[mf|bp|cc]_gos.pkl`  #go names of each subset
* `[train|valid|test]_data_separate_[mf|bp|cc]_labels.pkl`  #onehot labels of each subset
* `[train|valid|test]_data_separate_[mf|bp|cc]_proteins.csv`  #proteinID of each subset
* `[train|valid|test]_data_separate_[mf|bp|cc]_sequences.pkl`  #sequences of each subset

## Training DuGPro
If you have prepared the data, you can train our model on your data as follows (Ensure that your configure is right):
```
cd ./src
python train.py --ont cc --lr 5e-5

arguments:
    --ont: the ontology(mf/bp/cc)
    --lr: learning rate(defult: {'mf':7e-5, 'cc':5e-5,'bp':5e-5})
```

## Predicting
* You could use retrained model or our trained model to predict functions.
```
cd ./src
python predict.py --ont cc
arguments:
    --ont: the ontology(mf/bp/cc)
```
If you use our trained model, please first download [saved_model](https://drive.google.com/uc?export=download&id=1wmHozZd7iDDgoOaHwkOwR8RAz1) and save them into saved_model folder.

## Evaluation
* We provide Fmax, AUPR, IC_AUPR and DP_AUPR metrics to evaluate the performance. 
```
cd ./utils
python metrics.py
```

## Contact
Please feel free to contact us for any further questions.
* Qiurong Yang yangqr@csu.edu.cn
* Min Li limin@mail.csu.edu.cn


