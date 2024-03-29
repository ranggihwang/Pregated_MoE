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

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import os
import sys


class FTT5DecodingWeight(object):

    def empty_weights(self):
        print("[INFO] Load weights for decoding from CPP.")

        np_weight_dtype = self.weight_data_type
        torch_weight_dtype = {np.float32: torch.float32, np.float16: torch.float16}[np_weight_dtype]

        self.w = []
        for _ in range(39):
            self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
    
    def random_weights_for_inference_test(self):

        # suppose inter_size = d_ff
        
        hidden_dim = self.config.num_heads * self.config.d_kv;
        print("random_weights_for_inference_test for FT T5 Decoding are used!!!!!!!!!!!!!!")
        self.w = []
        # gen = np.random.rand
        def gen(*args):
            return np.zeros((args))
        # CHECK_INPUT(self_layernorm_gamma, _st);                     // layer_num, d_model
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_model).astype(np.float32)))
        # CHECK_INPUT(self_kernel_q, _st);                            // layer_num, d_model, 3 * hidden_dim
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_model, 3 * hidden_dim).astype(np.float32)))
        # CHECK_INPUT(self_output_kernel, _st);                       // layer_num, hidden_dim, d_model
        self.w.append(torch.tensor(gen(self.config.num_layers, hidden_dim, self.config.d_model).astype(np.float32)))
        # CHECK_INPUT(cross_layernorm_gamma, _st);                    // layer_num, d_model
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_model).astype(np.float32)))
        # CHECK_INPUT(cross_kernel_q, _st);                           // layer_num, d_model, hidden_dim
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_model, hidden_dim).astype(np.float32)))
        # CHECK_INPUT(cross_kernel_k, _st);                           // layer_num, mem_d_model, hidden_dim
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_model, hidden_dim).astype(np.float32)))
        # CHECK_INPUT(cross_kernel_v, _st);                           // layer_num, mem_d_model, hidden_dim
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_model, hidden_dim).astype(np.float32)))
        # CHECK_INPUT(cross_output_kernel, _st);                      // layer_num, hidden_dim, d_model
        self.w.append(torch.tensor(gen(self.config.num_layers, hidden_dim, self.config.d_model).astype(np.float32)))
        # CHECK_INPUT(ffn_layernorm_gamma, _st);                      // layer_num, d_model
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_model).astype(np.float32)))
        # CHECK_INPUT(inter_kernel, _st);                             // layer_num, d_model, inter_size

        moe_layer_num = len(self.config.moe_layer_index);
        no_moe_layer_num = self.config.num_layers - moe_layer_num
        size = (no_moe_layer_num + moe_layer_num * self.config.num_experts) * self.config.d_model * self.config.d_ff;

        self.w.append(torch.tensor(gen(size).astype(np.float32)))

        # CHECK_INPUT(inter_kernel2, _st);                            // layer_num, d_model, inter_size
        size = (no_moe_layer_num) * self.config.d_model * self.config.d_ff;
        self.w.append(torch.tensor(gen(size).astype(np.float32)))
        # CHECK_INPUT(output_kernel, _st);                            // layer_num, inter_size, d_model
        size = (no_moe_layer_num + moe_layer_num * self.config.num_experts) * self.config.d_ff * self.config.d_model;
        self.w.append(torch.tensor(gen(size).astype(np.float32)))
        # CHECK_INPUT(decoding_gamma, _st);                           // d_model
        self.w.append(torch.tensor(gen(self.config.d_model).astype(np.float32)))
        # CHECK_INPUT(embedding_table, _st);                          // vocab_size, d_model
        self.w.append(torch.tensor(gen(self.config.vocab_size, self.config.d_model).astype(np.float32)))
        # CHECK_INPUT(lm_head, _st);                                  // d_model, vocab_size
        self.w.append(torch.tensor(gen(self.config.d_model, self.config.vocab_size).astype(np.float32)))
        # CHECK_INPUT(absolute_or_relative_position_embedding, _st);  // head_num, num_bucket or max_seq_len, d_model
        self.w.append(torch.tensor(gen(self.config.num_heads, self.config.relative_attention_num_buckets, self.config.d_model).astype(np.float32)))
        # if (t5_with_bias) {
        #     CHECK_INPUT(self_layernorm_beta, _st);   // layer_num, d_model
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_model).astype(np.float32)))
        #     CHECK_INPUT(self_bias_qkv, _st);         // layer_num,3 * hidden_dim
        self.w.append(torch.tensor(gen(self.config.num_layers, 3 * hidden_dim).astype(np.float32)))
        #     CHECK_INPUT(self_output_bias, _st);      // layer_num, d_model
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_model).astype(np.float32)))
        #     CHECK_INPUT(cross_layernorm_beta, _st);  // layer_num, d_model
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_model).astype(np.float32)))
        #     CHECK_INPUT(cross_bias_q, _st);          // layer_num, hidden_dim
        self.w.append(torch.tensor(gen(self.config.num_layers, hidden_dim).astype(np.float32)))
        #     CHECK_INPUT(cross_bias_k, _st);          // layer_num, hidden_dim
        self.w.append(torch.tensor(gen(self.config.num_layers, hidden_dim).astype(np.float32)))
        #     CHECK_INPUT(cross_bias_v, _st);          // layer_num, hidden_dim
        self.w.append(torch.tensor(gen(self.config.num_layers, hidden_dim).astype(np.float32)))
        #     CHECK_INPUT(cross_output_bias, _st);     // layer_num, d_model
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_model).astype(np.float32)))
        #     CHECK_INPUT(ffn_layernorm_beta, _st);    // layer_num, d_model
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_model).astype(np.float32)))
        #     CHECK_INPUT(inter_bias, _st);            // layer_num, inter_size
        size = (no_moe_layer_num + moe_layer_num * self.config.num_experts) * self.config.d_ff;
        self.w.append(torch.tensor(gen(size).astype(np.float32)))
        #     CHECK_INPUT(inter_bias2, _st);           // layer_num, inter_size
        self.w.append(torch.tensor(gen(self.config.num_layers, self.config.d_ff).astype(np.float32)))
        #     CHECK_INPUT(output_bias, _st);           // layer_num, d_model
        size = (no_moe_layer_num + moe_layer_num * self.config.num_experts) * self.config.d_model;
        self.w.append(torch.tensor(gen(size).astype(np.float32)))
        #     CHECK_INPUT(decoding_beta, _st);         // d_model
        self.w.append(torch.tensor(gen(self.config.d_model).astype(np.float32)))
        #     CHECK_INPUT(embedding_bias, _st);        // vocab_size
        self.w.append(torch.tensor(gen(self.config.vocab_size).astype(np.float32)))
        # }
        # if (expert_num != 0) {
        #     CHECK_INPUT(moe_gate, _st);  // hidden_dim, expert_num
        self.w.append(torch.tensor(gen(moe_layer_num, self.config.d_model, self.config.num_experts).astype(np.float32)))
        # }
        for _ in range(8):
            self.w.append(torch.zeros(1))

        for i in range(len(self.w)):


            # self.w[i] = self.w[i].contiguous().cuda()

            self.w[i] = self.w[i].contiguous()
            if i not in [9, 11]:
                self.w[i] = self.w[i].cuda()
            else:
                self.w[i] = self.w[i].pin_memory()

    def __init__(
            self,
            config,
            tensor_para_size,
            pipeline_para_size,
            *,
            t5_with_bias=False,
            use_gated_activation=False,
            t5_with_moe=False,
            position_embedding_type=0,
            weight_data_type
    ):
        self.config = config
        self.num_layer = config.num_layers
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.t5_with_bias = t5_with_bias
        self.use_gated_activation = use_gated_activation
        self.t5_with_moe = t5_with_moe
        self.position_embedding_type = position_embedding_type
        self.real_weights_num = 31  # assume all weights are allocated and converted to specific data type
        self.weight_data_type = weight_data_type
        self.adapter_inter_size = config.adapter_inter_size if hasattr(config, "adapter_inter_size") else 0
        self.w = []
        self.use_mpi = dist.is_mpi_available()

        if self.use_mpi:
            try:
                dist.init_process_group(backend='mpi')
            except:
                print("[INFO] WARNING: Exception occurred in dist.init_process_group(backend = 'mpi'). Maybe the process group has been initialized somewhere else.")
        else:
            print("[INFO] MPI is not available in this PyTorch build.")
            assert tensor_para_size == 1, "[FATAL] MPI is required for tensor_para_size > 1."
            assert pipeline_para_size == 1, "[FATAL] MPI is required for pipeline_para_size > 1."

        self.rank = dist.get_rank() if self.use_mpi else 0
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size() if self.use_mpi else 1
        assert world_size == tensor_para_size * \
            pipeline_para_size, "[ERROR] world_size != tensor_para_size * pipeline_para_size"
        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

    def load_from_model(self, model):
        start_layer = self.pipeline_para_rank * self.num_layer // self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer // self.pipeline_para_size

        np_weight_dtype = self.weight_data_type
        torch_weight_dtype = {np.float32: torch.float32, np.float16: torch.float16}[np_weight_dtype]

        weight_dict = {}
        qkv_tmp = []
        for name, param in model.state_dict().items():
            if param.dim() == 2:
                param_t = param.transpose(1, 0)
            elif param.dim() == 1:
                param_t = param
            else:
                assert False, f"The dimension of param {name} should be 2"
            if name.find("decoder.block") != -1:
                if name.find(".SelfAttention.q.weight") != -1 or name.find(".SelfAttention.k.weight") != -1 or name.find(".SelfAttention.v.weight") != -1:
                    qkv_tmp.append(param_t)
                    if len(qkv_tmp) == 3:
                        qkv = torch.cat(qkv_tmp, dim=-1)
                        weight_dict[name.replace("v.weight", "qkv.weight")] = qkv
                        qkv_tmp = []
                else:
                    weight_dict[name] = param_t
            elif name.find("decoder") != -1:
                weight_dict[name] = param_t
            else:
                weight_dict[name] = param_t

        # load by torch model directly
        t = torch.stack([weight_dict["decoder.block.{}.layer.0.layer_norm.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.0.SelfAttention.qkv.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.reshape([t.shape[0], t.shape[1], 3, t.shape[2] // 3])
        t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.0.SelfAttention.o.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.1.layer_norm.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.1.EncDecAttention.q.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.1.EncDecAttention.k.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.1.EncDecAttention.v.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.1.EncDecAttention.o.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = torch.stack([weight_dict["decoder.block.{}.layer.2.layer_norm.weight".format(i)]
                        for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        self.w.append(t)
        if self.use_gated_activation:
            t = torch.stack([weight_dict["decoder.block.{}.layer.2.DenseReluDense.wi_0.weight".format(i)]
                             for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
            t = torch.stack([weight_dict["decoder.block.{}.layer.2.DenseReluDense.wi_1.weight".format(i)]
                             for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
        else:
            t = torch.stack([weight_dict["decoder.block.{}.layer.2.DenseReluDense.wi.weight".format(i)]
                            for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            t = t.split(t.shape[-1] // self.tensor_para_size, dim=-1)[self.tensor_para_rank].contiguous()
            self.w.append(t)
            ## empty wi2 weight
            self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())
        t = torch.stack([weight_dict["decoder.block.{}.layer.2.DenseReluDense.wo.weight".format(i)] for i in range(start_layer, end_layer)], 0).contiguous().cuda()
        t = t.split(t.shape[1] // self.tensor_para_size, dim=1)[self.tensor_para_rank].contiguous()
        self.w.append(t)
        t = weight_dict["decoder.final_layer_norm.weight"].contiguous().cuda()
        self.w.append(t)
        t = weight_dict["shared.weight"].transpose(1, 0).contiguous().cuda()
        self.w.append(t)
        t = weight_dict["lm_head.weight"].transpose(1, 0).contiguous().cuda() # Transpose back to [vocab, hidden]
        self.w.append(t)
        t = weight_dict["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].contiguous().cuda()
        t = t.split(t.shape[0] // self.tensor_para_size, dim=0)[self.tensor_para_rank].contiguous()
        self.w.append(t)

        #TODO: pass None Type to Torch Op
        for i in range(23):
            self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

    def load_from_bin(self, ckpt_path, model_type):
        start_layer = self.pipeline_para_rank * self.num_layer //  self.pipeline_para_size
        end_layer = (self.pipeline_para_rank + 1) * self.num_layer //  self.pipeline_para_size

        np_weight_dtype = self.weight_data_type
        torch_weight_dtype = {np.float32: torch.float32, np.float16: torch.float16}[np_weight_dtype]
        
        # load by binary files
        if model_type == "Megatron":
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.0.layer_norm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.0.SelfAttention.qkv.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.0.SelfAttention.o.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.layer_norm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.q.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.k.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.v.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.o.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.layer_norm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.DenseReluDense.wi.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            if self.use_gated_activation:
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.DenseReluDense.wi2.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
            else:
                self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.DenseReluDense.wo.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.final_layer_norm.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.weight_T.bin", dtype=np_weight_dtype).reshape([self.config.d_model, self.config.vocab_size])).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/lm_head.weight.bin", dtype=np_weight_dtype).reshape(
                [self.config.d_model, self.config.vocab_size])).contiguous().cuda()
            self.w.append(t)
            t = None
            if (self.position_embedding_type == 0):
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)).contiguous().cuda()
            else:
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.ape.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            
            # add 14 additional bias if it is t5 megatron structure
            if self.t5_with_bias:
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.0.layer_norm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.0.SelfAttention.qkv.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.0.SelfAttention.o.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.layer_norm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.q.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.k.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.v.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.1.EncDecAttention.o.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.layer_norm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.DenseReluDense.wi.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                if self.use_gated_activation:
                    t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.DenseReluDense.wi2.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                    self.w.append(t)
                else:
                    self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.DenseReluDense.wo.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.final_layer_norm.bias.bin", dtype=np_weight_dtype)).contiguous().cuda()
                self.w.append(t)
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.bias.bin", dtype=np_weight_dtype)).contiguous().cuda()
                self.w.append(t)
            else:
                #TODO: pass None Type to Torch Op
                for i in range(14):
                    self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            # add empty moe gate weight
            self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

            if self.adapter_inter_size > 0:
                ckpt_path_block = f"{ckpt_path}/decoder.block"
                for adapter in ["after_attention_adapter", "after_ffn_adapter"]:
                    for in_out in ["wi", "wo"]:
                        t = torch.stack([torch.from_numpy(np.fromfile(
                            f"{ckpt_path_block}.{i}.{adapter}.DenseSiluDense.{in_out}.weight.{self.tensor_para_rank}.bin",
                            dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                        self.w.append(t)
                    for weight_bias in ["weight", "bias"]:
                        t = torch.stack([torch.from_numpy(np.fromfile(
                            f"{ckpt_path_block}.{i}.{adapter}.layer_norm.{weight_bias}.bin",
                            dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                        self.w.append(t)
            else:
                for i in range(8):
                    self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())

        else:
            # Megatron-DeepSpeed, no tensor parallelism currently
            #TODO: add tensor parallelism in the conversion script
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.input_layernorm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.attention.query_key_value.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.attention.dense.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.post_attention_layernorm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.query.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.key.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.value.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.dense.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.post_inter_attention_layernorm.weight.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
            self.w.append(t)
            
            # =========== process normal and moe dense layer =================
            t_list = []
            for i in range(start_layer, end_layer):
                if (os.path.isfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_h_to_4h.weight.{self.tensor_para_rank}.bin")):
                    t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_h_to_4h.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
                else:
                    t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.dense_h_to_4h.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
            self.w.append(torch.cat(t_list, 0).contiguous().cuda())
            # ================================================================

            # We don't have use_gated_activation in Megatron-DeepSpeed currently, so here weight placeholder is always empty
            # If we have it in the future, the binary file name should be modified according to the actual name.
            if self.use_gated_activation:
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.DenseReluDense.wi2.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
            else:
                self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

            # =========== process normal and moe dense layer =================
            t_list = []
            for i in range(start_layer, end_layer):
                if (os.path.isfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_4h_to_h.weight.{self.tensor_para_rank}.bin")):
                    t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_4h_to_h.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
                else:
                    t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.dense_4h_to_h.weight.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
            self.w.append(torch.cat(t_list, 0).contiguous().cuda())
            # ================================================================

            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.final_layernorm.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/word_embeddings.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            # lm_head weight
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/word_embeddings.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)
            # assume absolute position
            t = torch.from_numpy(np.fromfile(f"{ckpt_path}/position_embeddings.weight.bin", dtype=np_weight_dtype)).contiguous().cuda()
            self.w.append(t)

            if self.t5_with_bias:
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.input_layernorm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.attention.query_key_value.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.attention.dense.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.post_attention_layernorm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.query.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.key.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.value.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.inter_attention.dense.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.post_inter_attention_layernorm.bias.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                self.w.append(t)
                
                # =========== process normal and moe dense layer =================
                t_list = []
                for i in range(start_layer, end_layer):
                    if (os.path.isfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_h_to_4h.bias.{self.tensor_para_rank}.bin")):
                        t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_h_to_4h.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
                    else:
                        t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.dense_h_to_4h.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)))
                self.w.append(torch.cat(t_list, 0).contiguous().cuda())
                # ================================================================

                # We don't have use_gated_activation in Megatron-DeepSpeed currently, so here weight placeholder is always empty
                # If we have it in the future, the binary file name should be modified according to the actual name.
                if self.use_gated_activation:
                    t = torch.stack([torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.block.{i}.layer.2.DenseReluDense.wi2.bias.{self.tensor_para_rank}.bin", dtype=np_weight_dtype)) for i in range(start_layer, end_layer)], 0).contiguous().cuda()
                    self.w.append(t)
                else:
                    self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

                # =========== process normal and moe dense layer =================
                t_list = []
                for i in range(start_layer, end_layer):
                    if (os.path.isfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_4h_to_h.bias.bin")):
                        t_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.dense_4h_to_h.bias.bin", dtype=np_weight_dtype)))
                    else:
                        t_list.append(torch.zeros_like(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.deepspeed_moe.experts.deepspeed_experts.dense_4h_to_h.bias.bin", dtype=np_weight_dtype))))
                self.w.append(torch.cat(t_list, 0).contiguous().cuda())
                # ================================================================

                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.final_layernorm.bias.bin", dtype=np_weight_dtype)).contiguous().cuda()
                self.w.append(t)
                t = torch.from_numpy(np.fromfile(f"{ckpt_path}/shared.bias.bin", dtype=np_weight_dtype)).contiguous().cuda()
                self.w.append(t)
            else:
                for i in range(14):
                    self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())
            
            if self.t5_with_moe:
                gate_list = []
                for i in range(start_layer, end_layer):
                    if (os.path.isfile(f"{ckpt_path}/decoder.layers.{i}.mlp.deepspeed_moe.gate.wg.weight.bin")):
                        gate_list.append(torch.from_numpy(np.fromfile(f"{ckpt_path}/decoder.layers.{i}.mlp.deepspeed_moe.gate.wg.weight.bin", dtype=np_weight_dtype)))
                self.w.append(torch.stack(gate_list, 0).contiguous().cuda())
            else:
                self.w.append(torch.empty((1,1), dtype=torch_weight_dtype).contiguous().cuda())

            # adapters are not supported in Megatron-DeepSpeed currently, so here weight placeholder is always empty
            for i in range(8):
                self.w.append(torch.empty((1, 1), dtype=torch_weight_dtype).contiguous().cuda())

    def to_cuda(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].cuda()

    def to_float(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].float()

    def to_half(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].half()

    def to_single(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].float()

    def to_bfloat16(self):
        for i in range(self.real_weights_num):
            self.w[i] = self.w[i].bfloat16()


class FTT5Decoding(nn.Module):
    def __init__(self, decoding_weight_list, lib_path, head_num, head_size, inter_size,
                 mem_d_model, d_model, num_layer, start_id, end_id, vocab_size, q_scaling=1.0, num_bucket=32,
                 num_expert=0, moe_layer_index=[],
                 max_distance=128, tensor_para_size=1, pipeline_para_size=1, t5_with_bias=False,
                 position_embedding_type=0, moe_k=0,
                 activation_type="relu", tie_word_embeddings=True, adapter_inter_size=0, adapter_norm_position="pre"):
        super().__init__()

        self.use_mpi = dist.is_mpi_available()

        if self.use_mpi:
            try:
                dist.init_process_group(backend='mpi')
            except:
                print("[INFO] WARNING: Exception occurred in dist.init_process_group(backend = 'mpi'). Maybe the process group has been initialized somewhere else.")
        else:
            print("[INFO] MPI is not available in this PyTorch build.")
            assert tensor_para_size == 1, "[FATAL] MPI is required for tensor_para_size > 1."
            assert pipeline_para_size == 1, "[FATAL] MPI is required for pipeline_para_size > 1."

        torch.classes.load_library(lib_path)
        try:
            self.decoding = torch.classes.FasterTransformer.T5Decoding(head_num, head_size, inter_size, mem_d_model,
                                                                       d_model, num_layer,
                                                                       vocab_size, num_bucket, num_expert, max_distance,
                                                                       q_scaling, start_id, end_id,
                                                                       tensor_para_size, pipeline_para_size,
                                                                       t5_with_bias,
                                                                       position_embedding_type, moe_k, activation_type,
                                                                       tie_word_embeddings, adapter_inter_size,
                                                                       adapter_norm_position,
                                                                       moe_layer_index, *decoding_weight_list)
        except:
            self.decoding = torch.classes.FasterTransformerT5Decoding(head_num, head_size, inter_size, mem_d_model,
                                                                      d_model, num_layer,
                                                                      vocab_size, num_bucket, num_expert, max_distance,
                                                                      q_scaling, start_id, end_id,
                                                                      tensor_para_size, pipeline_para_size,
                                                                      t5_with_bias,
                                                                      position_embedding_type, moe_k, activation_type,
                                                                      tie_word_embeddings, adapter_inter_size,
                                                                      adapter_norm_position,
                                                                      moe_layer_index, *decoding_weight_list)

    def forward(self, beam_width, max_seq_len, top_k, top_p,
                beam_search_diversity_rate, temperature,
                len_penalty, repetition_penalty, presence_penalty, min_length, random_seed,
                mem_hidden_states, mem_seq_len,
                is_return_output_log_probs, is_return_cum_log_probs, is_return_cross_attentions=False,
                bad_words_list=None, stop_words_list=None):
        # TODO (bhsueh) Not found an method to put a None Type into op forward function
        # So, the top_k and top_p must be some values now.
        results = self.decoding.forward(beam_width, max_seq_len,
                                        top_k, top_p, beam_search_diversity_rate,
                                        temperature, len_penalty, repetition_penalty, presence_penalty, min_length,
                                        random_seed, mem_hidden_states, mem_seq_len,
                                        is_return_output_log_probs, is_return_cum_log_probs, is_return_cross_attentions,
                                        bad_words_list, stop_words_list)
        return results


class FTT5(nn.Module):
    def __init__(self, encoder, decoding):
        super().__init__()
        self.encoder = encoder
        self.decoding = decoding

    def forward(self, input_token, inputs_embeds, beam_size, max_seq_len,
                top_k, top_p, beam_search_diversity_rate = 0.0,
                temperature=1.0, len_penalty=0.0, repetition_penalty=None, presence_penalty=None, min_length=0, random_seed=0,
                is_return_output_log_probs=False, is_return_cum_log_probs=False, is_return_cross_attentions=False,
                bad_words_list=None, stop_words_list=None):
        input_ids = input_token.input_ids.to("cuda").type(torch.int32)
        mem_seq_len = 0
        if hasattr(input_token, "attention_mask"):
            mem_seq_len = torch.sum(input_token.attention_mask, dim=1).type(torch.int32).to("cuda")
        else:
            mem_seq_len = input_token.seq_len.type(torch.int32).to("cuda")

        print("===== ft encoder forward start")
        sys.stdout.flush()

        ft_encoder_outputs = self.encoder.forward(input_ids, mem_seq_len, inputs_embeds)

        print("===== ft encoder forward finished")
        print("===== ft decoder forward start")
        sys.stdout.flush()
        results = self.decoding.forward(beam_size,  # optional, can be None
                                        max_seq_len,
                                        top_k,  # optional, can be None
                                        top_p,  # optional, can be None
                                        beam_search_diversity_rate,  # optional, can be None
                                        temperature,  # optional, can be None
                                        len_penalty,  # optional, can be None
                                        repetition_penalty,  # optional, can be None
                                        presence_penalty,  # optional, can be None
                                        min_length,  # optional, can be None
                                        random_seed,  # optional, can be None
                                        ft_encoder_outputs,
                                        mem_seq_len,
                                        is_return_output_log_probs,  # optional, can be None
                                        is_return_cum_log_probs,  # optional, can be None
                                        is_return_cross_attentions,  # optional, can be None
                                        bad_words_list, # optional, can be None
                                        stop_words_list, # optional, can be None
                                        )
        print("===== ft decoder forward finished")

        ft_decoding_outputs = results.pop(0).reshape([-1, beam_size, max_seq_len])
        ft_decoding_seq_lens = results.pop(0).reshape([-1, beam_size])
        if is_return_output_log_probs:
            ft_output_log_probs = results.pop(0)
        if is_return_cum_log_probs:
            ft_cum_log_probs = results.pop(0)
        if is_return_cross_attentions:
            ft_cross_attentions = results.pop(0)
            return ft_decoding_outputs.cpu().numpy(), ft_decoding_seq_lens.cpu().numpy(), ft_cross_attentions.cpu().numpy()

        return ft_decoding_outputs.cpu().numpy(), ft_decoding_seq_lens.cpu().numpy()
