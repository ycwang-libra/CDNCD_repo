set -e
set -x

for corrupt_mode in 'gaussian_blur' 'jpeg_compression' 'impulse_noise'
do
    for corrupt_severity in 0 1 2 3 4 5
    do  
        for seed in 0 1 2 3 4
        do
            for mode in 'train' 'test'
            do
                python main.py \
                    --total_name CDNCD \
                    --current_name Prototype_code \
                    --log_level info \
                    --mode $mode \
                    --batch_size 64 \
                    --num_workers 4 \
                    --Server_select server_name \
                    --gpu 0 \
                    --epochs 200 \
                    --seed $seed \
                    --dataset_series CIFAR10 \
                    --dataset_subclass CIFAR10CMix \
                    --lab_corrupt_severity 0 \
                    --unlab_corrupt_severity $corrupt_severity \
                    --corrupt_mode $corrupt_mode \
                    --source_domain real \
                    --target_domain real \
                    --resizecrop_size 32 \
                    --num_labeled_classes 5 \
                    --num_unlabeled_classes 5 \
                    --weight_decay 5e-5 \
                    --lr 0.01 \
                    --warmup_teacher_temp 0.07 \
                    --teacher_temp 0.04 \
                    --warmup_teacher_temp_epochs 30 \
                    --memax_weight 1 \
                    --use_wandb True \
                    --use_wandb_offline True \
                    --style_remove_function orth \
                    --style_remove_loss_w 0.01 \
                    --test_matrics True
            done
        done
    done
done