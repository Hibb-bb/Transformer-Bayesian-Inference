CUDA_VISIBLE_DEVICES=1 python3 main_fast.py --num-example 100 --steps 50000
CUDA_VISIBLE_DEVICES=1 python3 main_fast.py --num-example 50 --steps 50000

CUDA_VISIBLE_DEVICES=1 python3 main_fast.py --steps 50000 --heads 2 
CUDA_VISIBLE_DEVICES=1 python3 main_fast.py --steps 50000 --heads 1
