import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from askchart.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from askchart.conversation import conv_templates, SeparatorStyle
from askchart.model.builder import load_pretrained_model
from askchart.utils import disable_torch_init
from askchart.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import ipdb
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="en", det = False) 

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        image_file_path = os.path.join(self.image_folder, image_file)
        if self.model_config.online_ocr:
            ocr_text = get_ocr_promots(image_file_path)[1]
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs + 'Reference OCR tokens in the image from left to right, top to bottom:' +  ocr_text 
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs + 'Reference OCR tokens in the image from left to right, top to bottom:' +  ocr_text
        else:
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        # print(f"qs: {qs}")
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(f"conv: {conv}, prompt: {prompt}")
        # ipdb.set_trace()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def try_eval_to_number(s):
    s = s.rstrip('- ').strip()
    
    int_pattern = r'^\d+$'
    float_pattern = r'^\d+\.\d+$'
    
    if re.match(int_pattern, s):
        return int(s)
    elif re.match(float_pattern, s):
        return float(s)
    else:
        return s

def get_ocr_promots(img_path):
    result = ocr.ocr(img_path, cls=True)
    if result[0] == None:
        with_order_str = ""
        no_order_str = ""
    else:
        numbers  = []
        all_str = []
        no_order = []
        with_order = []
        no_order.append(img_path[:-4])
        with_order.append(img_path[:-4])
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                value = try_eval_to_number(line[1][0])
                if isinstance(value, (int, float)):
                    numbers.append(str(value))
                else:
                    all_str.append(str(value))
                no_order.append(str(value))
        for number in numbers:
            with_order.append(number)
        for each_str in all_str:
            with_order.append(each_str)
        with_order_str = ",".join(with_order[1:])
        no_order_str = ",".join(no_order[1:])
    return with_order_str, no_order_str


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Model name: {model_name}, Model path: {model_path}")
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # print(f"Model: {model}, Tokenizer: {tokenizer}, Processor: {processor}", f"Context length: {context_len}")
    # ipdb.set_trace()

    if args.return_gating_logit is not None:
        from askchart.utils import get_gating_logit_by_hook
        print(model)
        fea_hooks = get_gating_logit_by_hook(model)
        all_gating_logits = {}
    image_processor = processor['image']
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")] # load questions dataset
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # print(f"Number of questions: {len(questions)}, chunk_idx: {args.chunk_idx}, num_chunks: {args.num_chunks}")
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config) # load dataset

    cnt = -1
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)): # iterate over dataset
        cnt += 1
        # if cnt == 5:
        #     break
        # print(f"line: {line}")
        # ipdb.set_trace()
        idx = line["id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True) 

        conv = conv_templates[args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] # stopping criteria

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True if args.return_gating_logit is None else False,
                stopping_criteria=stopping_criteria
            )
        if args.return_gating_logit is not None:
            # import ipdb
            # ipdb.set_trace()
            all_gating_logits[cnt] = dict(gating_logit=[i.fea for i in fea_hooks],
                                          images=image_tensor if image_tensor is None else image_tensor.detach().cpu(),
                                          input_ids=input_ids.detach().cpu(),
                                          output_ids=output_ids.detach().cpu())
            print(input_ids.shape, output_ids.shape, fea_hooks[0].fea.shape, image_tensor.shape if image_tensor is not None else [])
            # assert fea_hooks[0].fea.shape[0] + 1 == output_ids.shape[1] + 575
            print('The number of hooks is:', len(fea_hooks))

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()


        print(f"output: {outputs}")
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

    if args.return_gating_logit is not None:
        torch.save(all_gating_logits, f'{args.return_gating_logit}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image_tower", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024) # change to 128 for chartqa
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--return_gating_logit", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
