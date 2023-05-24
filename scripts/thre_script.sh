
SOURCE=(correct mistake feedback)
THRESHOLD=(0.5 0.6 0.7 0.8 0.9)
DEVICE=cuda:0

for s in ${SOURCE[@]}
do
    for thre in ${THRESHOLD[@]}
    do
        echo "Source: $s, Threshold: $thre Topk: 10"
        python3 train.py --source $s --thre $thre --topk 10 --device $DEVICE
    done
done