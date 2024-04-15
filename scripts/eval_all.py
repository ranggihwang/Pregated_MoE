import configparser
import subprocess
import pandas as pd
import re
from cpuinfo import get_cpu_info
import psutil
import torch
import argparse


def parse_output(output: str):
    block_lat = 0
    for line in reversed(output.splitlines()):
        m = re.search(r"BLK AVG: ([\d\.]+) ms", line)
        if m:
            block_lat = float(m[1])
            break

    throughput = .0
    for line in reversed(output.splitlines()):
        m = re.search(r", (\d+) tokens/sec\.", line)
        if m:
            throughput = int(m[1])
            break
    
    peak_mem_encoder = peak_mem_decoder = 0
    for line in output.splitlines():
        m = re.search(r"MEM usage: (\d+) (\d+)", line)
        if m:
            peak_mem_encoder = int(m[1])
            peak_mem_decoder = int(m[2])
            break

    max_active_experts = 0
    for line in output.splitlines():
        m = re.search(r"Max active experts: (\d+)", line)
        if m:
            max_active_experts = int(m[1])
            break

    cache_hit_rate = 0
    for line in output.splitlines():
        m = re.search(r"Average cache hit rate: ([\d\.]+)", line)
        if m:
            cache_hit_rate = float(m[1])
    
    return block_lat, throughput, peak_mem_encoder, peak_mem_decoder, max_active_experts, cache_hit_rate


def profile_config(cpp_config, model, method, batch_size, forced_num_expert=0, cache_ratio=0, disk_offload=0):
    iterations = 4
    exp_name = f"{model}_{method}_{batch_size}_{forced_num_expert}_{cache_ratio}_{disk_offload}"
    print(f"Running {exp_name}")
    if method == "GPU-only":
        encoder_fetcher_mode = "0"
        decoder_fetcher_mode = "0"
    elif method == "Pre-gated":
        encoder_fetcher_mode = "1"
        decoder_fetcher_mode = "2"
    elif method == "DeepSpeed":
        encoder_fetcher_mode = "1"
        decoder_fetcher_mode = "1"
    elif method == "SE-MoE":
        encoder_fetcher_mode = "1"
        decoder_fetcher_mode = "2"
        iterations = 1

    if "base" in model:
        size_per_expert = 18874368
        num_layer = 6
    elif "large" in model:
        size_per_expert = 33554432
        num_layer = 12

    total_experts = int(re.search(r"\d+", model)[0])

    arena_size = 21474836480
    use_cache = 0
    if cache_ratio != 0:
        use_cache = 1
        arena_size = max(round(num_layer * total_experts * cache_ratio), 1)
        arena_size = arena_size * size_per_expert

    cpp_config["default"] = {
        "arena_size": f"{arena_size}",
        "encoder_fetcher_mode": encoder_fetcher_mode,
        "decoder_fetcher_mode": decoder_fetcher_mode,
        "profiling": "1",
        "detailed_timing": "0",
        "offload_path": f"/data/ft/{model}/",
        "disk_offload": f"{disk_offload}",
        "load_from_cpp": "1",
        "use_cache": f"{use_cache}",
        "quant_mode": "0",
        "vocab_size": "32128",
        "fetch_all": f"{int(method == 'SE-MoE')}",
        "forced_num_experts": f"{forced_num_expert}",
        "cache_policy": "LFU",
    }

    with open("/workspace/FasterTransformer/cpp_config.ini", "w") as fp:
        cpp_config.write(fp)

    command = (
        f"python /workspace/FasterTransformer/examples/pytorch/t5/perf_benchmark.py "
        f"--batch_size {batch_size} "
        f"--beam_width 4 "
        f"--seq_len 256 "
        f"--data_type fp32 "
        f"--test_time 3 "
        f"--sampling_topk 1 "
        f"--model_type Megatron-DeepSpeed "
        f"--ckpt_path /data/ft/{model}/ "
        f"--model t5-base "
        f"--duration 0 "
        f"--iterations {iterations} "
    )

    print(command)

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd="/workspace/FasterTransformer/build"
    )

    with open(f"/workspace/FasterTransformer/logs/{exp_name}.log", "w") as fp:
        fp.write(result.stdout)

    block_lat, throughput, peak_mem_encoder, peak_mem_decoder, max_active_experts, cache_hit_rate = parse_output(result.stdout)

    if method == "Pre-gated":
        used_buffer = 2 * max_active_experts
    elif method == "DeepSpeed":
        used_buffer = max_active_experts
    elif method == "GPU-only":
        used_buffer = num_layer * total_experts
    elif method == "SE-MoE":
        used_buffer = 2 * total_experts

    peak_mem = peak_mem_decoder - arena_size - size_per_expert * (2 * total_experts - used_buffer)
    print(
        f"BLK AVG: {block_lat} ms, "
        f"throughput: {throughput} tokens/sec, "
        f"peak_mem_encoder: {peak_mem_encoder}, "
        f"peak_mem_decoder: {peak_mem_decoder}, "
        f"max_active_experts: {max_active_experts}, "
        f"peak_mem: {peak_mem}, "
        f"cache_hit_rate: {cache_hit_rate}"
    )

    return {
        "block_lat": block_lat,
        "throughput": throughput,
        "peak_mem": peak_mem,
        "max_active_expert": max_active_experts,
        "cache_hit_rate": cache_hit_rate,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re_run", action="store_true")
    parser.add_argument("--name", type=str, default="")
    parser.set_defaults(rerun=False, use_cache=False)
    return parser.parse_args()


def main():
    args = parse_args()

    models = [
        "switch-base-8",
        "switch-base-64",
        "switch-base-128",
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

    cpp_config = configparser.ConfigParser()
    cpp_config.read("/workspace/FasterTransformer/cpp_config.ini")

    hardware_infos = {
        "CPU": [get_cpu_info()["brand_raw"]] * len(batch_sizes),
        "RAM (GB)": [int(psutil.virtual_memory().total / 1024 / 1024 / 1024)] * len(batch_sizes),
        "GPU": [torch.cuda.get_device_name()] * len(batch_sizes),
    }

    results = {}
    for metric in metrics:
        results[metric] = {
            "bs": batch_sizes,
            "active experts": forced_num_experts,
            "cache ratio": cache_ratios,
        }
        results[metric].update(hardware_infos)

        for key, value in results[metric].items():
            if len(value) == 1:
                results[metric][key] = value * max(len(forced_num_experts), len(batch_sizes), len(cache_ratios))

    def gen_csv_name(metric):
        return f"{args.name}{'_' if args.name else ''}{metric}s.csv"

    if not args.re_run:
        for method in methods:
            for model in models:
                for disk_offload in disk_offloads:
                    records = []

                    for batch_size in batch_sizes:
                        for forced_num_expert in forced_num_experts:
                            for cache_ratio in cache_ratios:
                                records.append(
                                    profile_config(
                                        cpp_config,
                                        model,
                                        method,
                                        batch_size,
                                        forced_num_expert,
                                        cache_ratio,
                                        disk_offload,
                                    )
                                )

                    for metric, result in results.items():
                        result[f"{model}/{method}/{'SSD' if disk_offload else 'CPU'}"] = [
                            record[metric] for record in records
                        ]

                    # Generate CSV after each model and method runned
                    for metric, result in results.items():
                        df = pd.DataFrame.from_dict(result)
                        df.to_csv(f"{gen_csv_name(metric)}", index=False)

    else:
        dfs = {metric: pd.read_csv(f"/workspace/FasterTransformer/performance_data/{gen_csv_name(metric)}") for metric in metrics}
        models = [
            "switch-base-8",
            # "switch-base-64",
            # "switch-base-128",
            # "switch-large-128",
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
            # "Pre-gated",
            # "DeepSpeed",
            # "SE-MoE",
        ]
        rerun_configs = [
            (model, method, batch_size)
            for model in models
            for method in methods
            for batch_size in batch_sizes
        ]
        for model, method, batch_size in rerun_configs:
            row_idx = batch_sizes.index(batch_size)
            col_idx = "{}/{}".format(model, method)
            record = profile_config(cpp_config, model, method, batch_size)
            for metric, df in dfs.items():
                df.loc[row_idx, col_idx] = record[metric]
                df.to_csv(f"{gen_csv_name(metric)}", index=False)


if __name__ == "__main__":
    main()
