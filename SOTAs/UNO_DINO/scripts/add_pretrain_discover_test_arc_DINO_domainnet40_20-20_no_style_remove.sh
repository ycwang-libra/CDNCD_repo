set -e
set -x

for t_domain in 'real' 'clipart' 'painting' # all domains: 'real' 'clipart' 'painting' 'sketch'
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
                --dataset_series DomainNet40 \
                --dataset_subclass DomainNet40 \
                --source_domain sketch \
                --target_domain $t_domain \
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