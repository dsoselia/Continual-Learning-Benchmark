GPUID=$1
OUTDIR=outputs/extended_xray_task
REPEAT=1
mkdir -p $OUTDIR
# python -u iBatchLearn.py --dataset XRAY_seq --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 6 --other_split_size 6 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name EWC        --lr 0.001 --reg_coef 100 --allow_shuffle 0   --workers 8      | tee ${OUTDIR}/EWC.log
# #python -u iBatchLearn.py --dataset XRAY_seq --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 2 --other_split_size 2 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet                                              --lr 0.001 --offline_training    --allow_shuffle 0   --workers 8      | tee ${OUTDIR}/Offline_adam.log
# python -u iBatchLearn.py --dataset XRAY_seq --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 6 --other_split_size 6 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet                                              --lr 0.001    --allow_shuffle 0   --workers 8                     | tee ${OUTDIR}/Adam.log



# python -u iBatchLearn.py --dataset XRAY_seq --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 6 --other_split_size 6 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type regularization --agent_name SI  --lr 0.001 --reg_coef 2    --allow_shuffle 0   --workers 8            | tee ${OUTDIR}/SI.log
# python -u iBatchLearn.py --dataset XRAY_seq --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 6 --other_split_size 6 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name EWC_online --lr 0.001 --reg_coef 3000  --allow_shuffle 0   --workers 8    | tee ${OUTDIR}/EWC_online.log
# python -u iBatchLearn.py --dataset XRAY_seq --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 6 --other_split_size 6 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type regularization --agent_name MAS --lr 0.001 --reg_coef 10    --allow_shuffle 0   --workers 8           |tee  ${OUTDIR}/MAS.log
# python -u iBatchLearn.py --dataset XRAY_seq --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 6 --other_split_size 6 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name EWC        --lr 0.001 --reg_coef 100 --allow_shuffle 0   --workers 8      | tee ${OUTDIR}/EWC.log


