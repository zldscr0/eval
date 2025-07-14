python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /data1/wym_data/checkpoints/noentropyloss_ENP/global_step_95/actor \
    --target_dir hf_model/noentropyloss_ENP_step95
    