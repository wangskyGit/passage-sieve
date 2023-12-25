set -e
EXP_NAME=co_training_tq_simANS_test
TB_DIR=tensorboard_log/$EXP_NAME    # tensorboard log path
OUT_DIR=output/$EXP_NAME

DE_CKPT_PATH=../TQ/triviaqa_fintinue.pkl
CE_CKPT_PATH=../TQ/checkpoint-reranker34000
Origin_Data_Dir=$OUT_DIR/temp/train_ce.json
Origin_Data_Dir_Dev=../TQ/dev_ce_0.json

Iteration_step=2000
Iteration_reranker_step=500
MAX_STEPS=10000

# for global_step in `seq 0 2000 $MAX_STEPS`; do echo $global_step; done;
for global_step in `seq 0 $Iteration_step $MAX_STEPS`;
#for global_step in 0
do
    if ((global_step > 0)); then
    python -u -m torch.distributed.launch --nproc_per_node=4 wiki/negative_train.py --lr 1e-7 --num_hard_negatives 5 --epoch 1 --mode pos --num_negatives_eval 30 --renew True\
     --model_path $DE_CKPT_PATH --path_to_dataset $OUT_DIR/temp/train_ce.json --log_dir ../tb_log/TQar2_n30_$global_step > ../log/TQar2_$global_step.txt
    fi
    python -u -m torch.distributed.launch --nproc_per_node=8  wiki/co_training_wiki_train.py \
    --model_type=nghuyong/ernie-2.0-base-en \
    --model_name_or_path=$DE_CKPT_PATH \
    --max_seq_length=256 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
    --number_neg=15 --learning_rate=5e-6 \
    --reranker_model_type=nghuyong/ernie-2.0-large-en \
    --reranker_model_path=$CE_CKPT_PATH \
    --reranker_learning_rate=1e-6 \
    --output_dir=$OUT_DIR \
    --log_dir=$TB_DIR \
    --origin_data_dir=$Origin_Data_Dir \
    --warmup_steps=1000 --logging_steps=100 --save_steps=2000 --max_steps=$MAX_STEPS \
    --gradient_checkpointing --normal_loss \
    --iteration_step=$Iteration_step \
    --iteration_reranker_step=$Iteration_reranker_step \
    --temperature_normal=1 --ann_dir=$OUT_DIR/temp --adv_lambda 0 --global_step=$global_step

    g_global_step=`expr $global_step + $Iteration_step`
    python -u -m torch.distributed.launch --nproc_per_node=4 wiki/co_training_wiki_generate.py \
    --model_type=nghuyong/ernie-2.0-base-en \
    --model_name_or_path=$DE_CKPT_PATH \
    --max_seq_length=256 --per_gpu_train_batch_size=8 \
    --output_dir=output/$EXP_NAME \
    --log_dir=tensorboard/logs/$EXP_NAME \
    --origin_data_dir=$Origin_Data_Dir \
    --origin_data_dir_dev=$Origin_Data_Dir_Dev \
    --train_qa_path=../TQ/trivia-train.qa.csv \
    --test_qa_path=../TQ/trivia-test.qa.csv \
    --dev_qa_path=../TQ/trivia-dev.qa.csv \
    --passage_path=../TQ/psgs_w100.tsv \
    --max_steps=$MAX_STEPS \
    --gradient_checkpointing \
    --ann_dir=output/$EXP_NAME/temp --global_step=$g_global_step \
    --encode_passages

    python wiki/co_training_wiki_generate.py \
    --model_type=nghuyong/ernie-2.0-base-en \
    --model_name_or_path=$DE_CKPT_PATH \
    --max_seq_length=256 --per_gpu_train_batch_size=8 \
    --output_dir=output/$EXP_NAME \
    --log_dir=tensorboard/logs/$EXP_NAME \
    --origin_data_dir=$Origin_Data_Dir \
    --origin_data_dir_dev=$Origin_Data_Dir_Dev \
    --train_qa_path=../TQ/trivia-train.qa.csv \
    --test_qa_path=../TQ/trivia-test.qa.csv \
    --dev_qa_path=../TQ/trivia-dev.qa.csv \
    --passage_path=../TQ/psgs_w100.tsv \
    --max_steps=$MAX_STEPS \
    --gradient_checkpointing \
    --ann_dir=output/$EXP_NAME/temp --global_step=$g_global_step 
done
