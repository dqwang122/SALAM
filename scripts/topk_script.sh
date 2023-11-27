GPU=$1
LLM=flan_t5
SAVEROOT=results
BBQ_FB="path/to/bbh/feedback/file"

TOPK=(1 3 5 10)
SOURCE=(correct mistake feedback)

for s in ${SOURCE[@]}
do
    for k in ${TOPK[@]}
    do
        echo "Source: $s, Topk: $k Threshold: 0"
        CUDA_VISIBLE_DEVICES=$GPU python src/run.py \
            --mode exam \
            --task_name bbq_sm \
            --source $s \
            --llm_model ${LLM} \
            --save_root ${SAVEROOT}_bbq \
            --fbfile ${BBQ_FB} \
            --topk $k --thre 0 --batch_size 1
    done
done
