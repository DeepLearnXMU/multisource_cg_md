Requiements:
python 3.6
tensorflow 1.15

Input file:
{train/test}.{src_language}:   one source sentence per line
{train/test}.{src_language}.src2:   one source sentence in the other language per line
{train/test}.{src_language}.adj:   one word alignment matrix(flatten) per line
{train}.{tgt_language}:   one target sentence per line
{vocabfile}:   one word per line


1) generate the training data:
python -m transformer.data_generate \
    --raw_dir RAW_DIR\
    --data_dir DATA_DIR \
    --input_files_pattern INPUT_FILES_PATTERN \ #{train/test}.{language}
    --src_vocab_filename SRC_VOCAB_FILENAME \
    --tgt_vocab_filename TGT_VOCAB_FILENAME \
    --problem NAME\
    --fro FRO \
    --to TO \

2) train:
python -m transformer.main \
    --data_dir DATA_DIR \
    --model_dir MODEL_DIR \
    --train_steps TRAIN_STEPS \
    --param_set base \
    --num_gpus NUM_GPUS \
    --src_vocab_filename SRC_VOCAB_FILENAME \
    --tgt_vocab_filename TGT_VOCAB_FILENAME \
    --keep_checkpoint_max KEEP_CHECKPOINT_MAX \
    --save_checkpoints_steps SAVE_CHECKPOINTS_STEPS

3) average checkpoints:
bash avg_checkpoints.sh

4) test:
python -m transformer.translate \
    --data_dir DATA_DIR \
    --model_dir MODEL_DIR \
    --param_set base \
    --checkpoint_path CHECKPOINT_PATH \
    --file DECODER_SRC_FILE \
    --file_out DECODER_TGT_FILE \
    --src_vocab_filename SRC_VOCAB_FILENAME \
    --tgt_vocab_filename TGT_VOCAB_FILENAME \
    --beam_size 4 \
    --batch_size 32
sed -r 's/(@@ )|(@@ ?$)//g' DECODER_TGT_FILE > DECODER_TGT_FILE_WOBPE
perl multi-bleu-detok.perl  GROUNDTRUTH_FILE < DECODER_TGT_FILE_WOBPE

