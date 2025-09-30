import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import torch.nn.functional as F
import re
import numpy as np
import os
import PIL.Image
import math
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision
import json
import argparse
import copy
import random
from typing import List, Dict
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/your-janus-ckpt")
parser.add_argument("--data_path", type=str, default="test_data.txt")
parser.add_argument("--reasoning_prompt_path", type=str, default="/your-code-root/STAGE/data/prompt/reasoning_prompt.txt")
parser.add_argument("--save_dir", type=str, default="./oup_images_cot")
parser.add_argument("--semantic_cot", type=bool, default=False)
parser.add_argument("--num_generation", type=int, default=4)
parser.add_argument("--temperature", type=float, default=1.0)

args = parser.parse_args()

# specify the path to the model
model_path = args.model_path
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

prompt_list = []
with open(args.data_path, 'r') as f:
    for line in f:
        prompt_list.append(line.strip())

with open(args.reasoning_prompt_path, 'r') as f:
    cot_prompt = f.read().strip()


def get_caption_height(text, font, img_width, draw):
    """Calculate the height needed for given text at specified width"""
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        # Use textlength instead of textsize to get text width
        text_width = draw.textlength(test_line, font=font)
        
        if text_width < img_width - 20:  # 20 pixels margin
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # Calculate required height based on number of lines
    try:
        font_size = font.size
    except:
        font_size = font.getsize('X')
        font_size = max(font_size)
    line_height = font_size + 4 # 4 pixel spacing between lines
    return len(lines) * line_height + 20  # 10 pixels margin at top and bottom

def create_grid_with_captions(visual_img, answer_list, save_dir, prompt_text, num_generation, temperature):
    os.makedirs(save_dir, exist_ok=True)
    raw_dir = os.path.join(save_dir, "raw")
    cap_dir = os.path.join(save_dir, "captioned")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(cap_dir, exist_ok=True)

    safe_prefix = re.sub(r"[^\w\.-]+", "_", prompt_text).strip("_")

    count = min(num_generation, len(visual_img))
    if count == 0:
        print("[WARN] No images to save."); return

    sample_img = Image.fromarray(visual_img[0])
    img_width, _ = sample_img.size

    font_size = 16
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    try:
        font_size = font.size
    except Exception:
        fs = font.getsize('X')
        font_size = max(fs) if isinstance(fs, (tuple, list)) else int(fs)

    temp_img = Image.new('RGB', (img_width, 200), color='white')
    temp_draw = ImageDraw.Draw(temp_img)

    def get_caption_height(text, font, max_w, draw_obj):
        words = text.split()
        lines, cur = [], ""
        for w in words:
            t = (cur + " " + w) if cur else w
            if draw_obj.textlength(t, font=font) <= max_w - 20:  # 20 边距
                cur = t
            else:
                lines.append(cur); cur = w
        if cur: lines.append(cur)
        line_h = font_size + 4
        return max(line_h, len(lines) * line_h)

    max_caption_height = 0
    for i in range(count):
        cap = answer_list[i] if i < len(answer_list) else ""
        max_caption_height = max(max_caption_height, get_caption_height(cap, font, img_width, temp_draw))
    max_caption_height = max(max_caption_height, 30)
    print(f"Maximum caption height: {max_caption_height} px")

    for idx in range(count):
        img = Image.fromarray(visual_img[idx])
        if img.mode != "RGB":
            img = img.convert("RGB")
        raw_path = os.path.join(raw_dir, f"{safe_prefix}_{idx:03d}_temp{temperature:02f}.png")
        img.save(raw_path)
        print(raw_path)

        caption = answer_list[idx] if idx < len(answer_list) else ""
        img_w, img_h = img.size
        captioned_img = Image.new('RGB', (img_w, img_h + max_caption_height), color='white')
        captioned_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(captioned_img)

        words = caption.split()
        lines, cur = [], ""
        for w in words:
            t = (cur + " " + w) if cur else w
            if draw.textlength(t, font=font) <= img_w - 20:
                cur = t
            else:
                lines.append(cur); cur = w
        if cur: lines.append(cur)

        line_h = font_size + 4
        y_text = img_h + 10
        for line in lines:
            tw = draw.textlength(line, font=font)
            x = (img_w - tw) // 2
            draw.text((x, y_text), line, fill="black", font=font)
            y_text += line_h

        cap_path = os.path.join(cap_dir, f"{safe_prefix}_{idx:03d}_temp{temperature:02f}.png")
        captioned_img.save(cap_path)
        print(cap_path)

    print("[INFO] Saved originals to:", raw_dir)
    print("[INFO] Saved captioned to:", cap_dir)
    

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    prompt_text: str,
    semantic_cot: bool,
    temperature: float = 1.0,
    num_generation: int = 9,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    conversation: List[Dict[str, str]] = None,
):  


    prompt_inputs = vl_chat_processor.tokenizer(
            text=[prompt],
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=True
    )
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    prompt_ids = prompt_ids.repeat_interleave(num_generation, dim=0).to('cuda')
    prompt_mask = prompt_mask.repeat_interleave(num_generation, dim=0).to('cuda')
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
            image_gen_prompt = f"{prompt_text}. {answer}"

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
        conversation = [
            {
                "role": "User",
                "content": prompt_text,
            },
            {"role": "Assistant", "content": ""},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt_inputs = vl_chat_processor.tokenizer(
                text=[sft_format],
                return_tensors="pt",
                padding=True,
                padding_side="right",
                add_special_tokens=True
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        prompt_ids = prompt_ids.repeat_interleave(num_generation, dim=0).to('cuda')
        prompt_mask = prompt_mask.repeat_interleave(num_generation, dim=0).to('cuda')
        input_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids)
        prompt_ids, attention_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.repeat_interleave(num_generation, dim=0).to('cuda')
        attention_mask = attention_mask.repeat_interleave(num_generation, dim=0).to('cuda')
    
    prompt_ids = prompt_ids.to('cuda')
    attention_mask = attention_mask.to('cuda')
    image_start_token_id = vl_chat_processor.tokenizer.encode(vl_chat_processor.image_start_tag)[1]
    prompt_ids = torch.cat([prompt_ids, prompt_ids.new_full((prompt_ids.size(0), 1), image_start_token_id)], dim=1)
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=1)

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids)
    pad_input_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids.new_full((1, 1), vl_chat_processor.pad_id))
    total_generated_tokens_img = []

    total_entropy = 0.0
    total_tokens  = 0
    # Currently only one image generation (since the diversity is low)
    for j in range(inputs_embeds.shape[0] // num_generation):
        cond_inputs_embeds = inputs_embeds[j*num_generation: (j+1)*num_generation]
        cond_attention_mask = attention_mask[j*num_generation: (j+1)*num_generation]
        uncond_inputs_embeds = cond_inputs_embeds.clone()
        uncond_inputs_embeds[:, 1:-1] = pad_input_embeds
        
        inputs_embeds_img = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
        inputs_embeds_img[1::2] = uncond_inputs_embeds
        attention_mask_img = torch.repeat_interleave(cond_attention_mask, 2, dim=0)
        attention_mask_img[1::2] = torch.ones_like(attention_mask_img[1::2])

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
                logits_scaled = logits / temperature
                log_probs = F.log_softmax(logits_scaled, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)

                total_entropy += float(entropy.sum().item())
                total_tokens  += int(entropy.numel())

                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, k] = next_token.squeeze(dim=-1)

                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
                cur_inputs_embeds_img = img_embeds.unsqueeze(dim=1)
                cur_attention_mask_img = torch.cat([cur_attention_mask_img, cur_attention_mask_img.new_ones((cur_attention_mask_img.shape[0], 1), dtype=torch.int)], dim=1)


            print(generated_tokens.shape)
            total_generated_tokens_img.append(generated_tokens)

    total_generated_tokens_img = torch.cat(total_generated_tokens_img, dim=0)
    avg_entropy_nats = total_entropy / total_tokens if total_tokens > 0 else float('nan')
    avg_entropy_bits = avg_entropy_nats / math.log(2.0)
    print(f"[Entropy] tokens={total_tokens}  avg={avg_entropy_nats:.4f} nats  ({avg_entropy_bits:.4f} bits)")

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[num_generation, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((num_generation, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    
    if semantic_cot:
        create_grid_with_captions(visual_img, image_gen_prompt_list, args.save_dir, prompt_text, num_generation, temperature)
    else:
        create_grid_with_captions(visual_img, [prompt_text]*num_generation, args.save_dir, prompt_text, num_generation, temperature)


random.shuffle(prompt_list)
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
    prompt = sft_format

    generate(
        vl_gpt,
        vl_chat_processor,
        prompt,
        prompt_text,
        num_generation=args.num_generation,
        semantic_cot=args.semantic_cot,
        conversation=conversation,
        temperature=args.temperature
    )