set -e
set -x

for corrupt_mode in 'gaussian_blur' 'jpeg_compression' 'impulse_noise'
do
    for corrupt_severity in 0 1 2 3 4 5
    do
        for seed in 0 1 2 3 4
        do
            for stage in 'self' 'sup' 'auto' 'test'
            do
                python main.py \
                    --total_name NCD_SOTA_RS \
                    --current_name RS_no_style_remove \
                    --stage $stage \
                    --log_level info \
                    --batch_size 64 \
                    --num_workers 4 \
                    --Server_select server_name \
                    --gpu 0 \
                    --lr 0.1 \
                    --changestlye_rate 0 \
                    --epochs 200 \
                    --seed $seed \
                    --dataset_series CIFAR10 \
                    --dataset_subclass CIFAR10CMix \
                    --source_domain real \
                    --target_domain real \
                    --lab_corrupt_severity 0 \
                    --unlab_corrupt_severity $corrupt_severity \
                    --corrupt_mode $corrupt_mode \
                    --resizecrop_size 32 \
                    --num_labeled_classes 5 \
                    --num_unlabeled_classes 5 \
                    --proj_dim 512 \
                    --use_wandb True \
                    --use_wandb_offline True \
                    --test_matrics True \
                    --use_tsne_visual True
            done
        done
    done
done