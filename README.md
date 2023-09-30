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

## Run experiments



```bash
python main_lateral_connections.py <config> --wandb --plot --store <store_path>
python main_visualization.py <config> --load <store_path>
```

For config, use one of the following:
- `lateral_connection_baseline.yaml` (4 straight lines without alternative cells)
- `lateral_connection_alternative_cells.yaml` (straight lines)
- `lateral_connection_alternative_cells_8bit.yaml` (straight line digits)
- `lateral_connection_alternative_cells_mnist.yaml` (mnist digits)


## Create plots published in thesis

```bash
python print_thesis.py
```


## Create documentation

Locally:
```bash
cd docs
make clean html
```

On GitHub: Push to main branch
