# RecSysChallenge2022

### Author: Edoardo D'Amico
### Mail: damicoedoardo95@gmail.com, edoardo.damico@insight-centre.org

This repository has been used to participate at the RecSys Challenge 2022, sponsored by Dressipi.

# Repository setup
To repository has been setup using poetry, to prepare the environment run 
```bash
poetry install
``` 
# Data preparation
To prepare the data used to run the code run 
```python
python src/datasets/dataset.py
``` 
all the preprocessing needed for the different models and the dataset splits will be created.

# Models
The final submission is an hybrid of several different models, we report in the following the **MRR** score obtained on the local split for each of them:

- EASE: 0.1488
- EASE-TW: 0.1668
---
- CEASE: 0.0843
- CEASE-TW: 0.1030
---
- Hybrid-Ease: 0.1834
--- 
- Rp3Beta: 0.1519
- Rp3Beta-TW: 0.1720
---
- ItemKNN: 0.1400
- ItemKNN-TW: 0.1597
---
Personal developed model with new attention mechanism specifically thought for session embedding
- **Context_attention: 0.1845**

# Model training
Every model can be trained running the linked notebook under the folder **src/notebooks**, the recommendations used to train xgboost will be saved along, 
the only model that has a script for the training is the context attention, and can be trained running:
**src/models/cassandra/train_cassandra.py**

# Hybrid Model
## Hybrid dataset
Once every single model has been trained, and the recommendation saved, the dataset for the xgboost hybrid can be created running: **src/xgboost_dataset** selecting the model reported above.

## Train Xgboost
To train the Xgboost hybrid run the file src/xgboost_reranked.py
the final model will be saved.

## Create final submission
The final submission can be created running the notebook
**src/notebooks/xgboost_predictions.ipynb** selecting the name of the final xgboost model trained previously.

