# Machine Learning-Guided Selection of Cyclization Sites in Head-to-Tail Cyclic Peptides
code for training and applying ML model to predict cyclization sites in head-to-tail cyclic peptides 

## Files

`Training.ipynb` : iPython notebook for training

`Prediction.ipynb`: iPython notebook for predicting unknown sequences

`model_307.pkl`:  best trained model for predicting 

`scaler307.pkl`:  unified normalization method 

`Cyclic peptides disconnection sequences generated.ipynb`:  iPython notebook for generation of sequences by disconnecting each cyclization site of the cyclic peptides


## Install the required packages
```bash
conda create -n cyclopeptide python==3.7.1
conda activate cyclopeptide
pip install numpy==1.21.5
pip install pandas==1.3.5
pip install scikit-learn==1.0.2
pip install tensorflow==2.8.2
pip install xgboost==1.5.1
pip install ipykernel
python -m ipykernel install --name cyclopeptide
```

# Usage

To run the ML model, navigate to the Anaconda interface and execute the provided Jupyter Notebook (`Training.ipynb`). Follow the step-by-step instructions within the notebook to perform the analysis.(Switch the kernel to `cyclopeptide` in the installation guide)
