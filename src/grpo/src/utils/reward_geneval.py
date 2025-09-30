import requests
import time
from requests.adapters import HTTPAdapter, Retry
from io import BytesIO
import torch
import numpy as np
import pickle
from collections import defaultdict
from PIL import Image, ImageOps
import json
import os
from mmdet.apis import inference_detector, init_detector
from concurrent.futures import ThreadPoolExecutor
import open_clip
import mmdet
# from mmcv.transforms import Compose
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmengine.dataset import default_collate
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it


def batch_inference_detector(model, image_pils, device='cuda:0'):
    cfg = model.cfg
    pipeline = cfg.data.test.pipeline

    for i, transform in enumerate(pipeline):
        if transform['type'] == 'LoadImageFromFile':
            pipeline[i] = dict(type='LoadImageFromWebcam')
    pipeline = Compose(pipeline)

    data_list = []
    for image_pil in image_pils:
        data = dict(img=np.array(image_pil))
        data = pipeline(data)
        data_list.append(data)

    data = collate(data_list, samples_per_gpu=len(image_pils))
    data = scatter(data, [device])[0]

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, transform):
        self._image = image.convert("RGB")
        self.transform=transform
        bgcolor = "#999"
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        image = image.crop(box[:4])
        return (self.transform(image), 0)


class Geneval_score:
    def __init__(self,args):
        # self.clip_path = args.geneval_clip_path
        # self.mmdet_path = args.geneval_mmdet_path
        self.THRESHOLD = 0.3
        self.COUNTING_THRESHOLD = 0.9
        self.MAX_OBJECTS = 16
        self.NMS_THRESHOLD = 1.0
        self.POSITION_THRESHOLD = 0.1
        self.only_strict=True
        self.COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
        self.COLOR_CLASSIFIERS = {}

        default_config = "/your-mmdet-code/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        default_ckpt = "/your-mmdet-ckpt/mmdetection"
        default_clip = "your-clip-ckpt/timm/vit_large_patch14_clip_224.openai/open_clip_pytorch_model.bin"
        default_obj  = "/your-stage-root/STAGE/src/grpo/src/utils/reward-server/reward_server/object_names.txt"

        self.MY_CONFIG_PATH = getattr(args, "geneval_config_path", default_config)
        self.MY_CKPT_PATH   = getattr(args, "geneval_ckpt_path", default_ckpt)
        self.MY_CLIP_PATH   = getattr(args, "geneval_clip_path", default_clip)
        self.OBJ_NAMES_PATH = getattr(args, "geneval_obj_names_path", default_obj)

    @property
    def __name__(self):
        return 'Geneval'
    
    def load_to_device(self, load_device):
        CONFIG_PATH = os.path.join(
                os.path.dirname(mmdet.__file__),
                self.MY_CONFIG_PATH
            )
        OBJECT_DETECTOR = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
        CKPT_PATH = os.path.join(self.MY_CKPT_PATH, f"{OBJECT_DETECTOR}.pth")
        object_detector = init_detector(CONFIG_PATH, CKPT_PATH, device=load_device)

        clip_arch = "ViT-L-14"
        clip_model, _, transform = open_clip.create_model_and_transforms(clip_arch, pretrained=self.MY_CLIP_PATH, device=load_device)
        tokenizer = open_clip.get_tokenizer(clip_arch)

        with open(self.OBJ_NAMES_PATH) as cls_file:
            classnames = [line.strip() for line in cls_file]
            
        self.object_detector=object_detector
        self.classnames=classnames
        self.clip_model=clip_model
        self.transform=transform
        self.tokenizer=tokenizer
            
    def relative_position(self, obj_a, obj_b):
        """Give position of A relative to B, factoring in object dimensions"""
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        #
        revised_offset = np.maximum(np.abs(offset) - self.POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        #
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5: relations.add("left of")
        if dx > 0.5: relations.add("right of")
        if dy < -0.5: relations.add("above")
        if dy > 0.5: relations.add("below")
        return relations

    def color_classification(self, image, bboxes, classname, device, transform):
        if classname not in self.COLOR_CLASSIFIERS:
            self.COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
                self.clip_model, self.tokenizer, self.COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object"
                ],
                device
            )
        clf = self.COLOR_CLASSIFIERS[classname]
        dataloader = torch.utils.data.DataLoader(
            ImageCrops(image, bboxes, transform),
            batch_size=32, num_workers=4
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(self.clip_model, clf, dataloader, device)
            return [self.COLORS[index.item()] for index in pred.argmax(1)]

    def evaluate(self, image, objects, metadata, device):
        """
        Evaluate given image using detected objects on the global metadata specifications.
        Assumptions:
        * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
        * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
        * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
            therefore, objects are expected to appear in sorted order
        """
        correct = True
        reason = []
        matched_groups = []
        # Check for expected objects
        for req in metadata.get('include', []):
            classname = req['class']
            matched = True
            found_objects = objects.get(classname, [])[:req['count']]
            if len(found_objects) < req['count']:
                correct = matched = False
                reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
            else:
                if 'color' in req:
                    # Color check
                    colors = self.color_classification(image, found_objects, classname, device, transform=self.transform)
                    if colors.count(req['color']) < req['count']:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found " +
                            f"{colors.count(req['color'])} {req['color']}; and " +
                            ", ".join(f"{colors.count(c)} {c}" for c in self.COLORS if c in colors)
                        )
                if 'position' in req and matched and req['position']!=None:
                    # Relative position check
                    expected_rel, target_group = req['position']
                    target_group=int(target_group)
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self.relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found " +
                                        f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        # Check for non-expected objects
        for req in metadata.get('exclude', []):
            classname = req['class']
            if len(objects.get(classname, [])) >= req['count']:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
        return correct, "\n".join(reason)


    def evaluate_reward(self,image, objects, metadata, device):
        """
        Evaluate given image using detected objects on the global metadata specifications.
        Assumptions:
        * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
        * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
        * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
            therefore, objects are expected to appear in sorted order
        """
        correct = True
        reason = []
        rewards = []
        matched_groups = []
        # Check for expected objects
        for req in metadata.get('include', []):
            classname = req['class']
            matched = True
            found_objects = objects.get(classname, [])
            rewards.append(1-abs(int(req['count']) - len(found_objects))/int(req['count']))
            if len(found_objects) != req['count']:
                correct = matched = False
                reason.append(f"expected {classname}=={req['count']}, found {len(found_objects)}")
                if 'color' in req or 'position' in req:
                    rewards.append(0.0)
            else:
                if 'color' in req:
                    # Color check
                    colors = self.color_classification(image, found_objects, classname,device, transform=self.transform)
                    rewards.append(1-abs(int(req['count']) - colors.count(req['color']))/int(req['count']))
                    if colors.count(req['color']) != req['count']:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found " +
                            f"{colors.count(req['color'])} {req['color']}; and " +
                            ", ".join(f"{colors.count(c)} {c}" for c in self.COLORS if c in colors)
                        )
                if 'position' in req and matched and req['position']!=None:
                    # Relative position check
                    expected_rel, target_group = req['position']
                    target_group=int(target_group)
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                        rewards.append(0.0)
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self.relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found " +
                                        f"{' and '.join(true_rels)} target"
                                    )
                                    rewards.append(0.0)
                                    break
                            if not matched:
                                break
                        rewards.append(1.0)
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        reward = sum(rewards) / len(rewards) if rewards else 0
        return correct, reward, "\n".join(reason)


    def _evaluate_single(self, result, image_pil, metadata, device):
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        image = ImageOps.exif_transpose(image_pil)
        detected = {}
        confidence_threshold = self.THRESHOLD if metadata['tag'] != "counting" else self.COUNTING_THRESHOLD

        for index, classname in enumerate(self.classnames):
            if bbox[index].shape[0] == 0:
                continue
            ordering = np.argsort(bbox[index][:, 4])[::-1]
            ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
            ordering = ordering[:self.MAX_OBJECTS].tolist()
            detected[classname] = []
            while ordering:
                max_obj = ordering.pop(0)
                detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
                ordering = [
                    obj for obj in ordering
                    if self.NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < self.NMS_THRESHOLD
                ]
            if not detected[classname]:
                del detected[classname]

        is_strict_correct, score, reason = self.evaluate_reward(image, detected, metadata, device)
        is_correct = False if self.only_strict else self.evaluate(image, detected, metadata, device)[0]

        return {
            'tag': metadata['tag'],
            'prompt': metadata['prompt'],
            'correct': is_correct,
            'strict_correct': is_strict_correct,
            'score': score,
            'reason': reason,
            'metadata': json.dumps(metadata),
            'details': json.dumps({
                key: [box.tolist() for box, _ in value]
                for key, value in detected.items()
            })
        }


    def evaluate_image(self,image_pils, metadatas, only_strict, device):
        start=time.time()
        with torch.inference_mode():
            results = inference_detector(self.object_detector, [np.array(image_pil) for image_pil in image_pils])
            # results = batch_inference_detector(self.object_detector, image_pils, device)
        print('1 cost %.2f s'%(time.time()-start))
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._evaluate_single, result, image_pil, metadata, device)
                for result, image_pil, metadata in zip(results, image_pils, metadatas)
            ]
            ret = [f.result() for f in futures]
        print('cost %.2f s'%(time.time()-start))
        return ret

    def reformulate_metadata(self, metadatas):
        for item in metadatas:
            if "include" in item:
                for obj in item["include"]:
                    if "count" in obj and isinstance(obj["count"], str):
                        try:
                            obj["count"] = int(obj["count"])
                        except ValueError:
                            pass

            if "exclude" in item:
                for obj in item["exclude"]:
                    if "count" in obj and isinstance(obj["count"], str):
                        try:
                            obj["count"] = int(obj["count"])
                        except ValueError:
                            pass 

                    if "position" in obj and isinstance(obj["position"], list) and len(obj["position"]) == 2:
                        if isinstance(obj["position"][1], str):
                            try:
                                obj["position"][1] = int(obj["position"][1])
                            except ValueError:
                                pass
        return metadatas
        

    def __call__(self,images, prompts, metadatas):
        metadatas=self.reformulate_metadata(metadatas)
        device = list(self.clip_model.parameters())[0].device

        required_keys = ['single_object', 'two_object', 'counting', 'colors', 'position', 'color_attr']
        scores = []
        strict_rewards = []
        grouped_strict_rewards = defaultdict(list)
        rewards = []
        grouped_rewards = defaultdict(list)
        results = self.evaluate_image(images, metadatas, only_strict=self.only_strict, device=device)
        for result in results:
            strict_rewards.append(1.0 if result["strict_correct"] else 0.0)
            scores.append(result["score"])
            rewards.append(1.0 if result["correct"] else 0.0)
            tag = result["tag"]
            for key in required_keys:
                if key != tag:
                    grouped_strict_rewards[key].append(-10.0)
                    grouped_rewards[key].append(-10.0)
                else:
                    grouped_strict_rewards[tag].append(1.0 if result["strict_correct"] else 0.0)
                    grouped_rewards[tag].append(1.0 if result["correct"] else 0.0)
        print(scores)
        return scores, rewards, strict_rewards, dict(grouped_rewards), dict(grouped_strict_rewards)
    
    
if __name__ == "__main__":
    data_path='/geneval-data'
    image_path='/img-path'
    with open(f'{data_path}/metadata.jsonl','r') as f:
        meta_data=[json.loads(line) for line in f]
        prompts=[item['prompt'] for item in meta_data]
        
    images=[]
    for item in os.listdir(image_path):
        images.append(Image.open(f'{image_path}/{item}'))
        
    Geneval=Geneval_score(None)
    Geneval.load_to_device('cuda')
    
    start=time.time()
    Geneval(images,prompts*len(images),meta_data*4)
    print('cost %.2f s'%(time.time()-start))