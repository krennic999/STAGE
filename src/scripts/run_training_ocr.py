import os
import subprocess

os.environ["DEBUG_MODE"] = "true"
os.environ["LOG_PATH"] = "./outputs/debug.txt"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

run_name = "your-run-name"
qwen_path = "/your-ckpt-root/ckpts/deepseek-ai/Janus-Pro-7B"
hf_dataset = "/your-code-root/STAGE/data/train_ocr_tag.json"
output_dir = f"/your-save-root/{run_name}"

cmd = [
    "torchrun",
    "--nproc_per_node=8",
    "--nnodes=1",
    "--node_rank=0",
    "--master_addr=127.0.0.1",
    "--master_port=12346",
    "open_r1/grpo.py",
    "--use_vllm", "False",
    "--deepspeed", "../configs/zero3.json",
    "--output_dir", output_dir,
    "--model_name_or_path", qwen_path,
    "--semantic_cot", "False",
    "--dataset_name", hf_dataset,
    "--max_prompt_length", "512",
    "--max_completion_length", "1024",
    "--temperature", "1.0",
    "--num_generations", "1",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "1",
    "--logging_steps", "1",
    "--bf16",
    "--torch_dtype", "bfloat16",
    # "--report_to", none,
    "--gradient_checkpointing", "false",
    "--attn_implementation", "flash_attention_2",
    "--max_steps", "1600",
    "--run_name", run_name,
    "--save_steps", "100",
    "--new_generations_image", "8",
    "--image_token_num_per_image", "576",
    "--cfg_weight", "5",
    "--reasoning_prompt_path", "../../../data/prompt/reasoning_prompt.txt",
    "--reward_funcs", "ocr",
    "--beta", "0.01",
    "--tf32", "true",
    "--learning_rate", "1e-6",
    "--reward_ckpt_path_file", "/your-code-root/STAGE/src/grpo/configs/reward_paths.json",
    
    "--reward_smooth", "True",
    "--kl_reweight", "True",
    "--update_ref", "False",
    "--progress_learning", "False",
    "--add_noise", "False",
    "--entropy_reward", "False",
]

os.chdir("/your-code-root/STAGE/src/grpo/src")

os.environ['WANDB_PROJECT']="your-proj-name"
os.environ["PYTHONPATH"] = f"{os.getcwd()}/..:" + os.environ.get("PYTHONPATH", "")

subprocess.run(cmd)
