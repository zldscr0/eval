model="/home/wym/bzx/verl/hf_model/NoEntropySplit"

CUDA_VISIBLE_DEVICES=4 python run.py --datasets aime2025 --hf-type chat --hf-path "$model" --max-out-len 8000  --dump-eval-details --accelerator vllm

CUDA_VISIBLE_DEVICES=4 python run.py --datasets aime2024_0shot_nocot_gen_2b9dc2 --hf-type chat --hf-path "$model" --max-out-len 8000 --dump-eval-details --accelerator vllm

CUDA_VISIBLE_DEVICES=4 python run.py --datasets amc23 --hf-type chat --hf-path "$model" --max-out-len 8000  --dump-eval-details --accelerator vllm

CUDA_VISIBLE_DEVICES=4 python run.py --datasets math_500_gen --hf-type chat --hf-path "$model" --max-out-len 8000 --accelerator vllm

