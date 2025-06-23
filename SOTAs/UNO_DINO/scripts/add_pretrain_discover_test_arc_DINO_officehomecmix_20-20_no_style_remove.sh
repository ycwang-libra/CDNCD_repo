set -e
set -x

for corrupt_mode in 'gaussian_blur' 'jpeg_compression' 'impulse_noise'
do
    for seed in 0 1 2 3 4
    do
        for stage in 'pretrain' 'discover' 'test'
        do
            python main.py \
                --total_name NCD_SOTA_UNO \
                --current_name UNO_DINO_no_style_remove \
                --log_level info \
                --stage $stage \
                --arch DINO \
                --use_pretrained_arch True \
                --batch_size 64 \
                --num_workers 4 \
                --Server_select server_name \
                --gpu 0 \
                --epochs 200 \
                --seed $seed \
                --base_lr 0.01 \
                --min_lr 0.0001 \
                --dataset_series OfficeHome \
                --dataset_subclass all \
                --lab_corrupt_severity 0 \
                --unlab_corrupt_severity 5 \
                --corrupt_mode $corrupt_mode \
                --source_domain Real_World \
                --target_domain Real_World \
                --resizecrop_size 128 \
                --num_labeled_classes 20 \
                --num_unlabeled_classes 20 \
                --num_heads 4 \
                --multicrop \
                --overcluster_factor 10 \
                --use_wandb True \
                --use_wandb_offline True \
                --test_matrics True \
                --del_trained_model True
        done
    done
done