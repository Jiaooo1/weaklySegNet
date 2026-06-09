# weaklySegNet
Requirement
conda env create --name weaklySeg --file environment.yml

Datasets
MoNuSeg: https://monuseg.grand-challenge.org/Data/
CPM17: https://www.cancer.gov/ccg/research/genome-sequencing/tcga

Data preparation
Run prepare_data.py for Voronoi label and the k-means clustering label.
Run labelbycolor.py and post_processing.py for color-aware pseudo labels.

weaklySegNet pipeline
Run train_coarse.py
Run train_fine.py
Run test_fine.py

Contact details
If you have any questions, please contact Jiaooo111@126.com.

