
# TASK=(bbq_sm bbh)
GPU=$1
LLM=flan_t5 # flan_t5, llama, gpt2_neox
SA=checkpoints/checkpoint-928/ # path/to/study_assistant/agent
BBH_FB="path/to/bbh/feedback/file"
BBQ_FB="path/to/bbq/feedback/file"
SAVEROOT=results

SOURCE=(none correct mistake feedback feedback_only)

for s in ${SOURCE[@]}
do
    echo "bbq $s | GPU: $GPU | ${LLM} | ${BBQ_FB}" 
    CUDA_VISIBLE_DEVICES=$GPU python src/run.py \
        --mode exam \
        --task_name bbq_sm \
        --source $s \
        --llm_model ${LLM} \
        --save_root ${SAVEROOT}_bbq \
        --fbfile ${BBQ_FB} \
        --topk 3 --thre 0.9 --batch_size 1 --save
done

echo "bbq agent | GPU: $GPU | ${LLM} + ${SA}" 
CUDA_VISIBLE_DEVICES=$GPU python src/run.py \
    --mode exam \
    --task_name bbq_sm \
    --source agent \
    --llm_model ${LLM} \
    --sa_model ${SA} \
    --save_root  ${SAVEROOT}_bbq \
    --save

for s in ${SOURCE[@]}
do
    echo "bbh | $s | GPU: $GPU | ${LLM}| ${BBH_FB}" 
    CUDA_VISIBLE_DEVICES=$GPU python src/run.py \
        --mode exam \
        --task_name bbh \
        --source $s \
        --llm_model ${LLM} \
        --save_root ${SAVEROOT}_bbh \
        --fbfile ${BBH_FB} \
        --topk 1 --thre 0.95 --batch_size 1
done

echo "bbh agent | GPU: $GPU | ${LLM} + ${SA}" 
CUDA_VISIBLE_DEVICES=$GPU python src/run.py \
    --mode exam \
    --task_name bbh \
    --source agent \
    --llm_model ${LLM} \
    --sa_model ${SA} \
    --save_root  ${SAVEROOT}_bbg \
    --save