# Lateral Connections

## Setup without GPU (slow)
Create conda environment

```bash
conda create --name lateral-connections python=3.10 pytorch torchvision -c pytorch
```

Activate Environment

```bash
conda activate lateral-connections
```

Install requirements

```bash
pip install -r requirements.txt
```

## Setup with GPU (fast)
Create conda environment

```bash
conda create --name lateral-connections python=3.10 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
```

Activate Environment

```bash
conda activate lateral-connections
```

Install requirements

```bash
pip install -r requirements.txt
pip install ffcv
```
