SOURCE=(correct mistake feedback)
TOPK=(1 3 5 10)
DEVICE=1

for s in ${SOURCE[@]}
do
    for k in ${TOPK[@]}
    do
        echo "Source: $s, TOPK: $k Threhold: 0"
        CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py --source $s --thre 0.0 --topk $k
    done
done