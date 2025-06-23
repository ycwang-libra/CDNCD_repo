set -e
set -x

for corrupt_mode in 'gaussian_blur' 'jpeg_compression' 'impulse_noise'
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
                --dataset_series OfficeHome \
                --dataset_subclass all \
                --lab_corrupt_severity 0 \
                --unlab_corrupt_severity 3 \
                --corrupt_mode $corrupt_mode \
                --source_domain Real_World \
                --target_domain Real_World \
                --resizecrop_size 128 \
                --num_labeled_classes 20 \
                --num_unlabeled_classes 20 \
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
                --test_matrics True \
                --tsne_embedding True \
                --delete_model True
        done
    done
done