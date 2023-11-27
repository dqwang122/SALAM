
GPU=$1
LLM=flan_t5
SAVEROOT=results
BBQ_FB="path/to/bbh/feedback/file"

SOURCE=(correct mistake feedback)
THRESHOLD=(0.5 0.6 0.7 0.8 0.9)


for s in ${SOURCE[@]}
do
    for thre in ${THRESHOLD[@]}
    do
        echo "Source: $s, Threshold: $thre Topk: 10"
        CUDA_VISIBLE_DEVICES=$GPU python src/run.py \
            --mode exam \
            --task_name bbq_sm \
            --source $s \
            --llm_model ${LLM} \
            --save_root ${SAVEROOT}_bbq \
            --fbfile ${BBQ_FB} \
            --topk 10 --thre $thre --batch_size 1
    done
done