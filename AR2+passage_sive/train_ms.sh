set -e
EXP_NAME=co_training_MS_MARCO_Pas_SimANS
Iteration_step=5000
Iteration_reranker_step=500
MAX_STEPS=30000
# for global_step in `seq 0 2000 $MAX_STEPS`; do echo $global_step; done;
for global_step in `seq 0 $Iteration_step $MAX_STEPS`;
do
    if ((global_step > 0)); then
    python -u -m torch.distributed.launch --nproc_per_node=4 co_training/negative_train_ms.py --lr 1e-7 --num_hard_negatives 3 --epoch 1 --mode pos --num_negatives_eval 30 --renew True\
     --model_path ../ms/checkpoint-20000 --path_to_dataset output/$EXP_NAME/temp/train_ce.tsv --path_to_corpus ../ms --log_dir ../tb_log/MSar2_n30_$global_step > ../log/MSar2_$global_step.txt
    fi
    
    python -u -m torch.distributed.launch --nproc_per_node=4  co_training/co_training_marco_train.py \
    --model_type=Luyu/co-condenser-marco \
    --model_name_or_path=../ms/checkpoint-20000 \
    --max_seq_length=128 --per_gpu_train_batch_size=32 --gradient_accumulation_steps=2 \
    --number_neg=15 --learning_rate=5e-6 \
    --teacher_model_type=nghuyong/ernie-2.0-large-en \
    --teacher_model_path=../ms/checkpoint-reranker20000 \
    --teacher_learning_rate=5e-7 \
    --output_dir=output/$EXP_NAME \
    --log_dir=tensorboard/logs/$EXP_NAME \
    --origin_data_dir=output/$EXP_NAME/temp/train_ce.tsv \
    --train_qa_path=../ms/train.query.txt \
    --dev_qa_path=../ms/dev.query.txt \
    --passage_path=../ms \
    --logging_steps=10 --save_steps=5000 --max_steps=$MAX_STEPS \
    --gradient_checkpointing --distill_loss \
    --iteration_step=$Iteration_step \
    --iteration_reranker_step=$Iteration_reranker_step \
    --temperature_distill=1 --ann_dir=output/$EXP_NAME/temp --adv_lambda 1 --global_step=$global_step
    
    g_global_step=`expr $global_step + $Iteration_step`
    python -u -m torch.distributed.launch --nproc_per_node=4 co_training/co_training_marco_generate.py \
    --model_type=Luyu/co-condenser-marco \
    --max_seq_length=128 \
    --output_dir=output/$EXP_NAME \
    --log_dir=tensorboard/logs/$EXP_NAME \
    --train_qa_path=../ms/train.query.txt \
    --dev_qa_path=../ms/dev.query.txt \
    --passage_path=../ms \
    --max_steps=$MAX_STEPS \
    --gradient_checkpointing --adv_step=0 \
    --iteration_step=$Iteration_step \
    --iteration_reranker_step=$Iteration_reranker_step \
    --ann_dir=output/$EXP_NAME/temp --global_step=$g_global_step
 
done