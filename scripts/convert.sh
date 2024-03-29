for model in switch-base-8 switch-base-64 switch-base-128 switch-large-128
do
    echo $model
    rm -rf /data/ft/$model
    python /workspace/FasterTransformer/examples/pytorch/t5/utils/huggingface_switch_transformer_ckpt_convert.py -saved_dir /data/ft/$model -in_file google/$model -inference_tensor_para_size 1
    mv /data/ft/$model/1-gpu/* /data/ft/$model/
    rm -d /data/ft/$model/1-gpu
done
