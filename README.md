FAIR-UMN-HGCAL (FAIR-UMN)
==============================

Repository for analyzing simulated data from HGCAL prototype uploaded on Zenodo. Run the following commands to setup a conda environment.

```
conda env create -f fair_cpu.yml

```

If you are running the code on the a Mac OS with arm64 architecture, use fair_macos_m1.yml file instead. One can also manually install the enviroment from scratch using the following commands.

```
conda env create -n fair_cpu --python=3.6
conda activate fair_cpu
pip install numpy scikit-learn scipy awkward ipykernel jupyter h5py
pip install matplotlib plotly
python -m ipykernel install -n fair_cpu
```

The tree structure of the directory is shown below. Onde can open and run notebooks from the notebooks folder. The dataloaders for pytorch are stored in src/data/data_utils.py and the pytorch related custom functions can be found under src/utils/torch_utils.py, where one can define custom functions for training loss and optimizer.

```bash
├── CITATION.cff
├── LICENSE
├── README.md
├── data
├── docs
│   ├── Makefile
│   ├── commands.rst
│   ├── conf.py
│   ├── getting-started.rst
│   ├── index.rst
│   └── make.bat
├── fair_cpu.yml
├── fair_macos_m1.yml
├── metadata
│   └── hgcal_electron_dataset.json
├── models
├── notebooks
│   ├── DNN.ipynb
│   ├── GetBinnedResolution.ipynb
│   ├── iframe_figures
│   │   ├── figure_38.html
│   │   ├── figure_39.html
│   │   └── figure_40.html
│   └── sparkles.ipynb
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── create_test_file.py
│   │   └── make_dataset.py
│   ├── dataset_utils.py
│   ├── features
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models
│   │   ├── DNN.py
│   │   ├── __init__.py
│   │   ├── predict_model.py
│   │   └── train_model.py
│   ├── utils
│   │   └── torch_utils.py
│   └── visualization
│       ├── __init__.py
│       └── visualize.py
└── training
```
<p><small>Project based on the <a target="_blank" href="https://github.com/FAIR4HEP/cookiecutter4fair">cookiecutter4fair project template</a>.</p>
