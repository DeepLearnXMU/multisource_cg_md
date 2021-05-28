num_last_checkpoints=5

model_dir=xxx

output_path=xxx
mkdir ${output_path}

CUDA_VISIBLE_DEVICES="" python avg_checkpoints.py \
    --num_last_checkpoints ${num_last_checkpoints} \
    --output_path ${output_path}/model.ckpt \
    --prefix ${model_dir}

