GPUID=$1
OUTDIR=outputs/split_CIFAR100_incremental_task
REPEAT=5
mkdir -p $OUTDIR
python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet                                              --lr 0.001 --offline_training         | tee ${OUTDIR}/Offline_adam.log
python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet                                              --lr 0.001                            | tee ${OUTDIR}/Adam.log
python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name EWC        --lr 0.001 --reg_coef 100      | tee ${OUTDIR}/EWC.log
