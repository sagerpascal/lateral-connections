# Lateral Connections

## Setup with/without GPU (slow)
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
pip install torch~=2.0.0 torchvision~=0.15.1
pip install -r requirements.txt
```

## Setup with GPU (fast)
Create conda environment

```bash
conda create --name lateral-connections python=3.10 cupy pkg-config libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.6 numba -c conda-forge -c pytorch                    
```

Activate Environment

```bash
conda activate lateral-connections
```

Install requirements

```bash
conda update ffmpeg
pip install ffcv
pip install -r requirements.txt
```
