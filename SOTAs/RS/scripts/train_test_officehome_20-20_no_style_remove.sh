set -e
set -x

for domain in 'Art' 'Clipart' 'Product' # all domains: 'Real_World' 'Art' 'Clipart' 'Product'
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
                --dataset_series OfficeHome \
                --dataset_subclass all \
                --source_domain Real_World \
                --target_domain $domain \
                --resizecrop_size 128 \
                --num_labeled_classes 20 \
                --num_unlabeled_classes 20 \
                --use_wandb True \
                --use_wandb_offline True \
                --test_matrics True \
                --del_trained_model True
        done
    done
done