for model in switch-xxl-128
do
    dir=/data/ft/$model-fp4
    echo $model $dir
    rm -rf $dir
    python /workspace/FasterTransformer/examples/pytorch/t5/utils/huggingface_switch_transformer_ckpt_convert_fp4.py -saved_dir $dir -in_file google/$model -inference_tensor_para_size 1 -weight_data_type fp16 --random_weight
    mv $dir/1-gpu/* $dir/
    rm -d $dir/1-gpu
done
