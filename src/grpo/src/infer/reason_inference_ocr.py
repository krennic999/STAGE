import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import re
from typing import List, Dict
import time
import argparse
from collections import OrderedDict
from transformers import AutoTokenizer
from transformers import GenerationConfig, TextStreamer
from typing import List
import hpsv2
import random
import copy
import pdb
import json
import torch
from torchvision import transforms
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

def process_text_tokens(vl_chat_processor,cot_prompt,prompt_list):
    prompt_list_return=[];prompt_text_list_return=[]
    for prompt in prompt_list:
        prompt_text = copy.deepcopy(prompt)
        conversation = [
            {
                "role": "User",
                "content": cot_prompt.format(prompt),
            },
            {"role": "Assistant", "content": ""},
        ]

        system_prompt = 'You are a helpful assistant that receives an image prompt and generate a visualization of the prompt.'
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=system_prompt,
        )
        prompt_ = sft_format
        prompt_list_return.append(prompt_)
        prompt_text_list_return.append(prompt_text)
    return prompt_list_return,prompt_text_list_return

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: List[str],
    prompt_text: List[str],
    semantic_cot: bool,
    temperature: float = 1,
    num_generation: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    conversation: List[Dict[str, str]] = None,
):  

    print('num_generations:',num_generation)
    prompt_inputs = vl_chat_processor.tokenizer(
            text=prompt,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=True
    )
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    prompt_ids = prompt_ids.to('cuda')
    prompt_mask = prompt_mask.to('cuda')
    input_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids)

    if semantic_cot:
        # TODO: if num_generations is too large, we need to split it into multiple batches
        if num_generation > 20:
            total_generations = []
            for i in range(prompt_ids.shape[0] // num_generation):
                current_input_embeds = input_embeds[i*num_generation: (i+1)*num_generation]
                current_attn_mask = prompt_mask[i*num_generation: (i+1)*num_generation]
                prompt_completion_ids = mmgpt.language_model.generate(
                    inputs_embeds=current_input_embeds,
                    attention_mask=current_attn_mask,
                    pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
                    bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
                    eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=True,
                    use_cache=True,
                )
                total_generations.append(prompt_completion_ids)
            prompt_completion_ids = torch.cat(total_generations, dim=0)
        else: # if num_generations == 1, we directly generate all for the batch data
            prompt_completion_ids = mmgpt.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=prompt_mask,
                pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
                bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
                eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True,
                use_cache=True,
            )

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_ids
        completion_ids = prompt_completion_ids

        image_gen_prompt_list = []
        
        prompt = vl_chat_processor.tokenizer.decode(prompt_ids[0].cpu().tolist(), skip_special_tokens=True)
        for i in range(completion_ids.shape[0]):
            answer = vl_chat_processor.tokenizer.decode(completion_ids[i].cpu().tolist(), skip_special_tokens=True)
            image_gen_prompt = f"{prompt_text[i]}. {answer}"

            conversation = [
                {
                    "role": "User",
                    "content": image_gen_prompt,
                },
                {"role": "Assistant", "content": ""},
            ]
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )

            print(f"Prompt {i}: {sft_format}\Semantic-CoT {i}: {answer}")
            image_gen_prompt_list.append(sft_format)

        prompt_inputs = vl_chat_processor.tokenizer(
            text=image_gen_prompt_list,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=True,
        ) # {'input_ids', 'attention_mask'}

        prompt_ids, attention_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    else:
        processed_prompt_text=[]
        for prompt_item in prompt_text:
            conversation = [
                {
                    "role": "User",
                    "content": prompt_item,
                },
                {"role": "Assistant", "content": ""},
            ]
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )
            processed_prompt_text.append(sft_format)
        print(processed_prompt_text)
            
        prompt_inputs = vl_chat_processor.tokenizer(
                text=processed_prompt_text,
                return_tensors="pt",
                padding=True,
                padding_side="right",
                add_special_tokens=True
        )

        prompt_ids, attention_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    
    prompt_ids = prompt_ids.to('cuda')
    attention_mask = attention_mask.to('cuda')
    image_start_token_id = vl_chat_processor.tokenizer.encode(vl_chat_processor.image_start_tag)[1]
    prompt_ids = torch.cat([prompt_ids, prompt_ids.new_full((prompt_ids.size(0), 1), image_start_token_id)], dim=1)
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=1)

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids)
    pad_input_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids.new_full((1, 1), vl_chat_processor.pad_id))
    total_generated_tokens_img = []

    for j in range(inputs_embeds.shape[0] // num_generation):
        cond_inputs_embeds = inputs_embeds[j*num_generation: (j+1)*num_generation]
        cond_attention_mask = attention_mask[j*num_generation: (j+1)*num_generation]
        uncond_inputs_embeds = cond_inputs_embeds.clone()
        uncond_inputs_embeds[:, 1:-1] = pad_input_embeds
        
        inputs_embeds_img = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
        inputs_embeds_img[1::2] = uncond_inputs_embeds
        attention_mask_img = torch.repeat_interleave(cond_attention_mask, 2, dim=0)
        attention_mask_img[1::2] = torch.ones_like(attention_mask_img[1::2])
        # import pdb; pdb.set_trace()

        split_size = 2 * num_generation
        for jj in range(0, inputs_embeds_img.shape[0], split_size):
            print(f"Generating image {jj}")
            start = jj
            end = min(jj + split_size, inputs_embeds_img.shape[0])
            generated_tokens = torch.zeros(((end-start)//2, image_token_num_per_image), dtype=torch.int64).cuda()
            cur_inputs_embeds_img = inputs_embeds_img[start: end]
            cur_attention_mask_img = attention_mask_img[start: end]

            for k in range(image_token_num_per_image):
                outputs = mmgpt.language_model.model(
                    inputs_embeds=cur_inputs_embeds_img, 
                    use_cache=True, 
                    past_key_values=outputs.past_key_values if k != 0 else None, 
                    attention_mask=cur_attention_mask_img
                )
                
                hidden_states = outputs.last_hidden_state
                logits = mmgpt.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                
                logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, k] = next_token.squeeze(dim=-1)

                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
                cur_inputs_embeds_img = img_embeds.unsqueeze(dim=1)
                cur_attention_mask_img = torch.cat([cur_attention_mask_img, cur_attention_mask_img.new_ones((cur_attention_mask_img.shape[0], 1), dtype=torch.int)], dim=1)


            print(generated_tokens.shape)
            total_generated_tokens_img.append(generated_tokens)

    total_generated_tokens_img = torch.cat(total_generated_tokens_img, dim=0)

    torch.cuda.empty_cache()
    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[num_generation, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((num_generation, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    
    return visual_img

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def transform_image(image, size=512):
    transform = transforms.Compose([
        transforms.Resize(size, max_size=None),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    return transform(image)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
    
def deduplicate_annotations(meta_json):
    unique_annotations = {}
    for annotation in meta_json['annotations']:
        unique_annotations[annotation['image_id']] = annotation
    meta_json['annotations'] = list(unique_annotations.values())

def get_prompts_for_rank_from_txt(
    txt_path: str,
    rank: int,
    world_size: int,
    batch_size: int,
    save_root: str | None = None,
    exts=(".jpg", ".png", ".jpeg"),
):
    with open(txt_path, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]

    all_items = [{"caption": p, "idx": i} for i, p in enumerate(prompts)]

    if save_root and os.path.isdir(save_root):
        existed = set()
        for name in os.listdir(save_root):
            low = name.lower()
            if not low.endswith(exts):
                continue
            m = re.search(r"(\d+)(?=\.[^.]+$)", name)
            if m:
                existed.add(int(m.group(1)))
        all_items = [it for it in all_items if it["idx"] not in existed]

    total = len(all_items)
    if rank == 0:
        print(f"[Info] Total prompt samples (after filtering): {total}")

    samples_per_rank = total // world_size
    start = rank * samples_per_rank
    end = total if rank == world_size - 1 else start + samples_per_rank
    rank_items = all_items[start:end]

    batched = []
    for i in range(0, len(rank_items), batch_size):
        batch = rank_items[i:i + batch_size]
        batched.append({
            "caption": [x["caption"] for x in batch],
            "idx":     [x["idx"] for x in batch],
        })
    return batched

    
    
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict

def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    seed=20
    batch_size=6
    
    print('setup inference...')
    setup(rank, world_size)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    print('begin inference..')
    
    model_path = args.model_path
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    with open(args.reasoning_prompt_path, 'r') as f:
        cot_prompt = f.read().strip()

    print("VQ model loaded")

    txt_path=args.ocr_txt_path
    batched_data=get_prompts_for_rank_from_txt(txt_path,rank,world_size,batch_size,args.save_root)

    # 处理数据集
    for item in batched_data:
        t2=time.time()
        # print(item)
        text_prompts = item['caption']
        name = item['idx']
        print('rank %d: '%rank, text_prompts)
        prompt,prompt_text=process_text_tokens(vl_chat_processor,cot_prompt,text_prompts)

        samples =  generate(
                            vl_gpt,
                            vl_chat_processor,
                            prompt,
                            prompt_text,
                            semantic_cot=args.semantic_cot,
                            num_generation=len(text_prompts)
                        )

        
        for idx in range(samples.shape[0]):
            fname=name[idx]
            img_pred = samples[idx]
            img_pred = Image.fromarray(img_pred)

            try:
                os.makedirs(args.save_root, exist_ok=True)
                img_pred.save(os.path.join(args.save_root, f"{fname:05d}.jpg"))
            except Exception as e:
                log_path = os.path.join("log.txt")
                error_msg = f"Failed to save {fname}, error: {str(e)}\n"
                print(error_msg)
                with open(log_path, "a") as log_file:
                    log_file.write(error_msg)

            print(f"Image {fname} saved. to {args.save_root}")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr_txt_path", type=str, default="/your-code-root/STAGE/data/test_ocr.txt")
    parser.add_argument("--model_path", type=str, default="/your-janus-ckpt")
    parser.add_argument("--reasoning_prompt_path", type=str, default="/your-code-root/STAGE/data/prompt/reasoning_prompt.txt")
    parser.add_argument("--semantic_cot", type=bool, default=False)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_root", type=str, default='/your-save-root')
    args = parser.parse_args()
    main(args)