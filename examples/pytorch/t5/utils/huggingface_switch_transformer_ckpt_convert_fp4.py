# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import configparser
import multiprocessing
from datetime import datetime
import logging
from pathlib import Path

import sys
import os
import re

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../../3rdparty/transformers/src/")

from transformers import SwitchTransformersForConditionalGeneration, SwitchTransformersEncoderModel, AutoConfig, SwitchTransformersPreTrainedModel

import numpy as np
import torch  # pytype: disable=import-error

def mock_random_weight(*args, **kwargs):
    pass

torch.nn.init.kaiming_normal_ = mock_random_weight
torch.nn.init.kaiming_uniform_ = mock_random_weight

LOGGER = logging.getLogger(__name__)

rename_mapping = {"relative_attention_num_buckets": "relative_attention_num_buckets_or_max_pos_seq_len"}
new_configs = {
    "structure": {"t5_with_bias": "false", "use_gated_activation": "false", "position_embedding_type": "relative"}}


def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def fuse_decoder_qkv(model, factor, saved_dir, np_weight_data_type):
    model_dict = {}
    for name, param in model.named_parameters():
        if name.find("decoder") != -1 and name.find("SelfAttention") != -1:
            model_dict[name] = param

    for i in range(model.decoder.config.num_layers):
        shape = model_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"].T.shape
        qkv = torch.cat([model_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"].T,
                         model_dict[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"].T,
                         model_dict[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"].T], dim=-1)

        qkv = qkv.reshape([shape[0], 3, shape[1]])
        qkv = qkv.cpu().detach().numpy().astype(np_weight_data_type)

        split_vals = np.split(qkv, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"decoder.block.{i}.layer.0.SelfAttention.qkv.weight.{j}.bin"
            split_vals[j].tofile(saved_path.as_posix())


# TODO: replace with real quantization
def quantize(param):
    # WI, WO, WI_SCALE, WO_SCALE
    out_feature, in_feature = param.shape
    weight_size = (in_feature * out_feature + out_feature * in_feature) // 2 + (out_feature + in_feature) * 2
    return np.empty((1, weight_size), dtype=np.int8)


def fuse_expert(model, factor, saved_dir, np_weight_data_type):
    model_dict = {}
    for name, param in model.named_parameters():
        if name.find("mlp") != -1:
            model_dict[name] = param

    for name, param in model.named_parameters():
        if name.find("mlp") != -1 and name.find("wi") != -1:
            all_weight = quantize(param)

            split_vals = np.split(all_weight, factor, axis=-1)

            m = re.match(r"(\w+)\.block\.(\d+)\.layer\.\d+\.mlp(?:\.experts\.expert_(\d+))?\.wi", name)

            if m[3]:
                new_name = f"{m[1]}::layer{m[2]}expert{m[3]}.bin"
            else:
                new_name = f"{m[1]}::layer{m[2]}.bin"

            for j in range(factor):
                saved_path = saved_dir / new_name
                split_vals[j].tofile(saved_path.as_posix())


def split_and_convert_process(key, val, factor, saved_dir):
    # if val.ndim == 2:
    #     val = val.transpose(1, 0)
    saved_key = key
    LOGGER.debug(f"key: {key}, val.shape: {val.shape}")

    if key.find("shared.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())

        saved_path = saved_dir / f"{saved_key}_T.bin"
        val.T.tofile(saved_path.as_posix())
    elif key.find("lm_head.weight") != -1:
        # lm_head weights, only need to convert the weights of rank 0
        val = val.transpose(1, 0)  # For lm_head, we use TN gemm to compute, so we don't need to transpose
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())

    elif key.find("layer_norm.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())

    elif (
            key.find("SelfAttention.o.weight") != -1
            or key.find("EncDecAttention.o.weight") != -1
    ):
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif ((
            key.find("encoder") != -1 and (
                key.find("SelfAttention.q.weight") != -1
                or key.find("SelfAttention.k.weight") != -1
                or key.find("SelfAttention.v.weight") != -1
            )
        )
            or key.find("EncDecAttention.q.weight") != -1
            or key.find("EncDecAttention.k.weight") != -1
            or key.find("EncDecAttention.v.weight") != -1
            or key.find("mlp.router.classifier.weight") != -1
    ):
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif (
            key.find("DenseReluDense.wi_0.weight") != -1
            or key.find("DenseReluDense.wi_1.weight") != -1
    ):
        # For gated activation.
        if key.find("DenseReluDense.wi_0.weight") != -1:
            saved_key = key.replace("wi_0", "wi")
        elif key.find("DenseReluDense.wi_1.weight") != -1:
            saved_key = key.replace("wi_1", "wi2")
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("relative_attention_bias") != -1:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif (
            key.find("decoder") != -1 and
            (
                    key.find("SelfAttention.q.weight") != -1
                    or key.find("SelfAttention.k.weight") != -1
                    or key.find("SelfAttention.v.weight") != -1
            )
    ):
        pass
    elif (
            key.find("DenseReluDense.wo.weight") != -1
            or key.find("DenseReluDense.wi.weight") != -1
            or key.find("mlp.wi.weight") != -1
            or key.find("mlp.wo.weight") != -1
            or re.search(r"mlp.experts.expert_\d+.wo.weight", key) is not None
            or re.search(r"mlp.experts.expert_\d+.wi.weight", key) is not None
    ):
        pass
    elif key.find("encoder.embed_tokens.weight") != -1 or \
            key.find("decoder.embed_tokens.weight") != -1:
        LOGGER.warning(f"Not save {key}, using shared.weight directly.")
    else:
        LOGGER.warning(f"cannot find key '{key}' with shape {val.shape}")


def convert_checkpoint(args):
    saved_dir = Path(args.saved_dir) / f"{args.inference_tensor_para_size:d}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    if args.encoder_only:
        model = SwitchTransformersEncoderModel.from_pretrained(args.in_file)
    else:
        if args.random_weight:
            cfg = AutoConfig.from_pretrained(args.in_file)
            SwitchTransformersForConditionalGeneration._init_weights = mock_random_weight
            SwitchTransformersPreTrainedModel._init_weights = mock_random_weight
            model = SwitchTransformersForConditionalGeneration(cfg)
        else:
            model = SwitchTransformersForConditionalGeneration.from_pretrained(args.in_file)

    config = configparser.ConfigParser()

    if model.encoder.config.feed_forward_proj.find("gated") != -1:
        new_configs["structure"]["use_gated_activation"] = "1"

    config["encoder"] = {}
    for key, val in model.encoder.config.to_dict().items():
        config["encoder"][key] = f"{val}"
    config["encoder"]["weight_data_type"] = args.weight_data_type
    config["decoder"] = {}
    if not args.encoder_only:
        for key, val in model.decoder.config.to_dict().items():
            config["decoder"][key] = f"{val}"
        config["decoder"]["weight_data_type"] = args.weight_data_type

    for key, val in rename_mapping.items():
        config['encoder'][val] = config['encoder'].pop(key)
        if not args.encoder_only:
            config['decoder'][val] = config['decoder'].pop(key)
    for key, val in new_configs.items():
        config[key] = {}
        for val_key, val_val in val.items():
            config[key][val_key] = val_val

    config["structure"]["t5_with_moe"] = "1"
    config["structure"]["moe_layers_in_encoder"] = str(list(range(1, int(config["encoder"]["num_layers"]), 2)))
    config["structure"]["moe_layers_in_decoder"] = str(list(range(1, int(config["decoder"]["num_layers"]), 2)))

    with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)
    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    i_gpu_num = args.inference_tensor_para_size

    pool = multiprocessing.Pool(args.processes)
    pool.starmap_async(split_and_convert_process,
                       [(name, np.empty_like(param, dtype=np_weight_data_type), i_gpu_num, saved_dir)
                        for name, param in model.state_dict().items()])

    pool.close()
    pool.join()

    if not args.encoder_only:
        fuse_decoder_qkv(model, i_gpu_num, saved_dir, np_weight_data_type)

    fuse_expert(model, i_gpu_num, saved_dir, np_weight_data_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of input checkpoint file", required=True)
    parser.add_argument("-inference_tensor_para_size", "-i_g", type=int, help="How many gpus for inference",
                        required=True)
    parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 4)",
                        default=4)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--encoder_only", "-e", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Provide verbose messages")
    parser.add_argument("--random_weight", action="store_true")
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=log_format)
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))
