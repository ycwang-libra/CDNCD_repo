set -e
set -x

for target_domain in 'Art' 'Clipart' 'Product' # all domains: 'Real_World' 'Art' 'Clipart' 'Product' from Real_World to remaining domains
do
    for style_remove_function in 'orth' 'cossimi' 'coco' # all style remove loss: 'orth' 'cossimi' 'coco'
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
                    --source_domain Real_World \
                    --target_domain $domain \
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
                    --style_remove_function $style_remove_function \
                    --style_remove_loss_w 0.01 \
                    --test_matrics True \
                    --tsne_embedding True \
                    --delete_model True
            done
        done
    done
done