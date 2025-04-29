# Semiparametric semi-supervised learning for general targets under distribution shift and decaying overlap

This repository contains the code for the paper "Semiparametric semi-supervised learning for general targets under distribution shift and decaying overlap" by Testa, Xu, Lei, and Roeder (2025+).

The Python scripts in this repository implement the simulation study and the real-data analysis described in the paper. The files are organized as follows:
- *multivariate_mean_simulation.py*: This script implements the simulation study for the multivariate mean target.
- *linear_coefficient_simulation.py*: This script implements the simulation study for the linear coefficient target.
- *plot_sim.py*: This script generates Figures 1 and 2, showing results for the simulation study in the main manuscript.
- *plot_sim_supp.py*: This script generates all the Figures in the supplementary material, showing additional results for the simulation study.
- *real_data_metabric.py*: This script implements the real-data analysis on METABRIC. Data can be found on [CBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric).
- *real_data_ibeacon.py*: This script implements the real-data analysis on BLE-RSSI. Data can be found on [Kaggle](https://www.kaggle.com/datasets/mehdimka/ble-rssi-dataset/data).
