CUDA_VISIBLE_DEVICES=1 python run.py --datasets amc23 --hf-type chat --hf-path /data1/wym_data/llm/DeepSeek-R1-Distill-Qwen-1.5B --max-out-len 8000  --dump-eval-details --accelerator vllm

CUDA_VISIBLE_DEVICES=1 python run.py --datasets amc23 --hf-type chat --hf-path /home/wym/bzx/verl/hf_model/global_step_197_hf --max-out-len 8000  --dump-eval-details --accelerator vllm

#CUDA_VISIBLE_DEVICES=1 python run.py --datasets amc23 --hf-type chat --hf-path /home/wym/bzx/verl/hf_model/en0_non0_global_step_95_hf --max-out-len 8000  --dump-eval-details --accelerator vllm