# Artifact Evaluation

## Setup

```bash
# Starting from the official container
docker run -ti --gpus all --shm-size 5g --name pregated -v ${DATA_PATH}:/data nvcr.io/nvidia/pytorch:22.09-py3 bash
git clone --recursive https://github.com/kaleid-liner/FasterTransformer.git

# build on A100
mkdir -p FasterTransformer/build
cd FasterTransformer/build
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
make -j

# Python dependencies
pip install -r ../examples/pytorch/t5/requirement.txt
```

Replace `${DATA_PATH}` with path on host.

## Prepare models

```bash
mkdir /data/ft
cd /workspace/FasterTransformer/
./scripts/convert.sh
```

## Evaluation

```bash
cd /workspace/FasterTransformer/
# logs will be output here
mkdir logs/
python scripts/eval_all.py
```

Check `block_lats.csv`, `throughputs.csv` and `peak_mems.csv` for block latencies, throughputs and peak memory usage respectively.

You can modify the following lines in `scripts/eval_all.py`:

```python
models = [
    "switch-base-8",
    "switch-base-16",
    "switch-base-32",
    "switch-base-64",
    "switch-base-128",
    "switch-base-256",
    "switch-large-128",
]
batch_sizes = [
    1,
    # 2,
    # 4,
    # 8,
    # 16,
]
methods = [
    "GPU-only",
    "Pre-gated",
    "DeepSpeed",
    "SE-MoE",
]
metrics = [
    "block_lat",
    "throughput",
    "peak_mem",
    # "max_active_expert",
    # "cache_hit_rate",
]
forced_num_experts = [
    0,
    # 1,
    # 2,
    # 4,
    # 8,
    # 16,
]
cache_ratios = [
    0,
    # 0.01,
    # 0.03,
    # 0.05,
    # 0.1,
    # 0.2,
    # 0.4,
    # 0.8,
]
disk_offloads = [
    0,
    # 1,
]
```
- By default, the data profiling will be conducted for Figures 10, 11, and 12 in the paper.
- To conduct profiling for Figure 13, uncomment the lines within `forced_num_experts`.
- For profiling related to Figure 15, uncomment the lines within `cache_ratios`.
- To profile for Figure 16, uncomment the lines in `disk_offloads`. Note that setting `disk_offload=1` will activate SSD offloading.

## Raw data

The raw data can be accessed at https://github.com/kaleid-liner/FasterTransformer/tree/main/performance_data.
