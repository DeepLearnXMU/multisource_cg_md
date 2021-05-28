base_dir=xxx
param_set=base
dropout=0.1
train_steps=200000
save_steps=5000
keep_checkpoint_max=50
model_name=xxx
model_dir=${base_dir}/model_dir/${model_name}
data_dir=xxx
src_vocab_filename=xxx
tgt_vocab_filename=xxx
batch_size=2048

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -u -m transformer.main \
    --param_set ${param_set} \
    --data_dir ${data_dir} \
    --model_dir ${model_dir} \
    --train_steps ${train_steps} \
    --num_gpus 4\
    --src_vocab_filename ${src_vocab_filename} \
    --tgt_vocab_filename ${tgt_vocab_filename} \
    --batch_size ${batch_size} \
    --dropout ${dropout} \
    --keep_checkpoint_max ${keep_checkpoint_max} \
    --save_checkpoints_steps ${save_steps} \
    --max_length 50\
    > ../log.output 2>&1 & \
