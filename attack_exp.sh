CUDA_VISIBLE_DEVICES=0 python3 attack_exp.py --dp_sgd 0 --noise_scale 0 --grad_norm 0 --mmd_loss_lambda 3 --mixup 0 --model_name alexnet --shadow_model_number 2 --test_model_number 2 --target_data_size 10000 --membership_attack_number 5000  --dataset cifar10 --target_learning_rate 0.01 --target_l2_ratio 1e-5 --early_stopping 0 --target_epochs 160 --schedule 80 120 --target_batch_size 100 --validation_mi 1 --pretrained 0 --alpha 1