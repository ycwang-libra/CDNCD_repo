set -e
set -x

for corrupt_mode in 'gaussian_blur' 'jpeg_compression' 'impulse_noise'
do
    for function in 'orth' 'coco' 'cossimi'
    do  
        for seed in 0 1 2 3 4
        do
            for stage in 'pretrain' 'discover' 'test'
            do
                python main.py \
                    --total_name NCD_SOTA_UNO \
                    --current_name UNO_DINO_style_remove \
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
                    --dataset_series CIFAR10 \
                    --dataset_subclass CIFAR10CMix \
                    --lab_corrupt_severity 0 \
                    --unlab_corrupt_severity 3 \
                    --corrupt_mode $corrupt_mode \
                    --resizecrop_size 32 \
                    --num_labeled_classes 5 \
                    --num_unlabeled_classes 5 \
                    --num_heads 4 \
                    --multicrop \
                    --overcluster_factor 10 \
                    --use_wandb True \
                    --use_wandb_offline True \
                    --style_remove_function $function \
                    --style_remove_loss_w 0.01 \
                    --test_matrics True \
                    --del_trained_model True \
                    --save_tsne_feature True
            done
        done
    done
done