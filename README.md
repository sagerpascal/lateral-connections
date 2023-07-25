# Lateral Connections

## Setup with/without GPU (slow)
Create conda environment

```bash
conda create --name lateral-connections python=3.10
```

Activate Environment

```bash
conda activate lateral-connections
```

Install requirements

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Setup with GPU (fast)
Create conda environment

```bash
conda create --name lateral-connections python=3.10 cupy pkg-config libjpeg-turbo opencv numba -c conda-forge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118           
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

## Create documentation

Locally:
```bash
cd docs
make clean html
```

On GitHub: Push to main branch
