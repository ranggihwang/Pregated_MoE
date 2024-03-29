import re
import pandas as pd
from cpuinfo import get_cpu_info
import psutil
import torch


def infer_peak_mem(model, method, pre_gated_peak_mem, pre_gated_max_active_experts):
    if "base" in model:
        size_per_expert = 18874368
        num_layer = 6
    elif "large" in model:
        size_per_expert = 33554432
        num_layer = 12

    total_experts = int(re.search(r"\d+", model)[0])

    pre_gated_used_buffer = 2 * pre_gated_max_active_experts
    if method == "Pre-gated":
        used_buffer = 2 * pre_gated_max_active_experts
    elif method == "DeepSpeed":
        used_buffer = pre_gated_max_active_experts
    elif method == "GPU-only":
        used_buffer = num_layer * total_experts
    elif method == "SE-MoE":
        used_buffer = 2 * total_experts

    peak_mem = pre_gated_peak_mem - size_per_expert * (pre_gated_used_buffer - used_buffer)
    return {
        "peak_mem": peak_mem,
        "max_active_expert": pre_gated_max_active_experts,
    }


def main():
    metrics = [
        "peak_mem",
        "max_active_expert",
    ]
    dfs = {metric: pd.read_csv(f"/workspace/FasterTransformer/performance_data/{metric}s.csv") for metric in metrics}
    models = [
        "switch-base-8",
        "switch-base-64",
        "switch-base-256",
        "switch-large-128",
    ]
    batch_sizes = [
        1,
        # 2,
        # 4,
        8,
        16,
    ]
    methods = [
        "GPU-only",
        "Pre-gated",
        "DeepSpeed",
        "SE-MoE",
    ]

    hardware_infos = {
        "CPU": [get_cpu_info()["brand_raw"]] * len(batch_sizes),
        "RAM (GB)": [int(psutil.virtual_memory().total / 1024 / 1024 / 1024)] * len(batch_sizes),
        "GPU": [torch.cuda.get_device_name()] * len(batch_sizes),
    }

    results = {}
    for metric in metrics:
        results[metric] = {"bs": batch_sizes}
        results[metric].update(hardware_infos)

    for model in models:
        for method in methods:
            records = []

            for batch_size in batch_sizes:
                row_idx = batch_sizes.index(batch_size)
                col_idx = "{}/{}".format(model, "Pre-gated")
                records.append(
                    infer_peak_mem(
                        model,
                        method,
                        dfs["peak_mem"].loc[row_idx, col_idx],
                        dfs["max_active_expert"].loc[row_idx, col_idx]
                    )
                )

            for metric, result in results.items():
                result["{}/{}".format(model, method)] = [record[metric] for record in records]

            # Generate CSV after each model and method runned
            for metric, result in results.items():
                df = pd.DataFrame.from_dict(result)
                df.to_csv(f"{metric}s.csv", index=False)


if __name__ == "__main__":
    main()