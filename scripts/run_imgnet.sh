torchrun --nproc_per_node=4 --master_port=29500 main_imgnet_ddp.py --batch_size 64 --time_step 6 --epochs 250 --learning_rate 0.05 --seed 42 --weight_decay 1e-5
