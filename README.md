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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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
