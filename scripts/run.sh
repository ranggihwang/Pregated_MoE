export CUDA_VISIBLE_DEVICES=1

python /workspace/FasterTransformer/examples/pytorch/t5/perf_benchmark.py \
        --batch_size 1 \
        --beam_width 4 \
        --seq_len 256 \
        --data_type fp16 \
        --test_time 3 \
        --sampling_topk 1 \
        --model_type Megatron-DeepSpeed \
        --ckpt_path /data/ft/switch-base-8-fp4/ \
        --model t5-base \
        --duration 0 \
        --iterations 4
