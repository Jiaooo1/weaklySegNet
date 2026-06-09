# WeaklySegNet

## 📋 Requirements

Create the conda environment with all dependencies:

```bash
conda env create --name weaklySeg --environment.yml
```

Activate the environment:

```bash
conda activate weaklySeg
```

## 📊 Datasets

The following datasets are supported:

- **MoNuSeg**: [Download from Grand Challenge](https://monuseg.grand-challenge.org/Data/)
- **CPM17**: [Download from TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)

## 🛠️ Data Preparation

Before training, prepare the data with the following steps:

### 1. Generate Voronoi and K-means labels

```bash
python prepare_data.py
```

This script generates Voronoi label and k-means clustering label for the dataset.

### 2. Generate color-aware pseudo labels

```bash
python labelbycolor.py
python post_processing.py
```

These scripts generate and refine color-aware pseudo labels.

## 🚀 WeaklySegNet Pipeline

Run the following scripts in order:

### Step 1: Coarse training

```bash
python train_coarse.py
```

### Step 2: Fine training

```bash
python train_fine.py
```

### Step 3: Testing

```bash
python test_fine.py
```

## 📧 Contact

If you have any questions, please feel free to contact:

**Email**: [Jiaooo111@126.com](mailto:Jiaooo111@126.com)
