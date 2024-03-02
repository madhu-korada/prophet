# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Runner that handles the prompting process
# ------------------------------------------------------------------------------ #

import os, sys
# sys.path.append(os.getcwd())

import pickle
import json, time
import math
import random
import argparse
from datetime import datetime
from copy import deepcopy
import yaml
from pathlib import Path
import openai

from evaluation.okvqa_evaluate import OKEvaluater
from transformers import AutoTokenizer, AutoModelForCausalLM 
import transformers
import torch

from .utils.fancy_pbar import progress, info_column
from .utils.data_utils import Qid2Data
from configs.task_cfgs import Cfgs



class Runner:
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater
        openai.api_key = __C.OPENAI_KEY

        # self.model_id = "gg-hf/gemma-2b-it"
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cuda",
            torch_dtype=self.dtype,
            # use_auth_token=True,
        )
        
    def gpt3_infer(self, prompt_text, _retry=0):
        # exponential backoff
        if _retry > 0:
            print('retrying...')
            st = 2 ** _retry
            time.sleep(st)
        
        if self.__C.DEBUG:
            # print(prompt_text)
            time.sleep(0.05)
            return 0, 0

        try:
            # print('calling gpt3...')
            response = openai.Completion.create(
                engine=self.__C.MODEL,
                prompt=prompt_text,
                temperature=self.__C.TEMPERATURE,
                max_tokens=self.__C.MAX_TOKENS,
                logprobs=1,
                stop=["\n", "<|endoftext|>"],
                # timeout=20,
            )
            # print('gpt3 called.')
        except Exception as e:
            print(type(e), e)
            if str(e) == 'You exceeded your current quota, please check your plan and billing details.':
                exit(1)
            return self.gpt3_infer(prompt_text, _retry + 1)

        response_txt = response.choices[0].text.strip()
        plist = []
        for ii in range(len(response['choices'][0]['logprobs']['tokens'])):
            if response['choices'][0]['logprobs']['tokens'][ii] in ["\n", "<|endoftext|>"]:
                break
            plist.append(response['choices'][0]['logprobs']['token_logprobs'][ii])
        prob = math.exp(sum(plist))
        
        return response_txt, prob
    
    
    def gemma_infer(self, prompt_text):
        chat = [
            { "role": "user", "content": {prompt_text} },
        ] 
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=10)
        # print(self.tokenizer.decode(outputs[0]))
        response = self.tokenizer.decode(outputs[0])
        print("response: ", response)
        response_txt = response.split("Answer: ")[-1].strip()
        print("response_txt: ", response_txt)
        prob = 1.0 # TODO: calculate the probability (how?)
        return response_txt, prob, response
    
    def mistral_infer(self, prompt_text):
        # print(prompt_text)
        # global tokenizer, model
        chat = [
            {"role": "user", "content": prompt_text}
        ] 
        prompt = self.tokenizer.apply_chat_template(chat,  return_tensors="pt")
        inputs = prompt.to(self.model.device) 
        outputs = self.model.generate(input_ids=inputs, max_new_tokens=1000, do_sample=True) 
        response = self.tokenizer.batch_decode(outputs)
        response = response[0]
        response_txt = response.split("Answer: [/INST] ")[-1].strip()
        response_txt = response_txt.split("</s>")[0]
        response_txt = response_txt.split("\n")[0]
        response_txt = response_txt.replace(".", "")
        print("response_txt: ", response_txt)
        print('\n\n')
        prob = 1 # TODO: calculate the probability (how?)
        return response_txt, prob, response

    def sample_make(self, ques, capt, cands, ans=None):
        line_prefix = self.__C.LINE_PREFIX
        cands = cands[:self.__C.K_CANDIDATES]
        prompt_text = line_prefix + f'Context: {capt}\n'
        prompt_text += line_prefix + f'Question: {ques}\n'
        cands_with_conf = [f'{cand["answer"]}({cand["confidence"]:.2f})' for cand in cands]
        cands = ', '.join(cands_with_conf)
        prompt_text += line_prefix + f'Candidates: {cands}\n'
        prompt_text += line_prefix + 'Answer:'
        if ans is not None:
            prompt_text += f' {ans}'
        return prompt_text

    def get_context(self, example_qids):
        # making context text for one testing input
        prompt_text = self.__C.PROMPT_HEAD
        examples = []
        for key in example_qids:
            ques = self.trainset.get_question(key)
            caption = self.trainset.get_caption(key)
            cands = self.trainset.get_topk_candidates(key)
            gt_ans = self.trainset.get_most_answer(key)
            examples.append((ques, caption, cands, gt_ans))
            prompt_text += self.sample_make(ques, caption, cands, ans=gt_ans)
            prompt_text += '\n\n'
        return prompt_text
    
    def run(self, write_only_prompt=False):
        ## where logs will be saved
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')
        ## where results will be saved
        Path(self.__C.RESULT_DIR).mkdir(parents=True, exist_ok=True)
        
        self.cache = {}
        self.cache_without_prompt = {}
        self.cache_file_path = os.path.join(
            self.__C.RESULT_DIR,
            'cache.json'
        )
        if self.__C.RESUME:
            self.cache = json.load(open(self.cache_file_path, 'r'))
            self.cache_without_prompt = json.load(open(self.cache_file_path.replace('.json', '_without_prompt.json'), 'r'))
        
        print('Note that the accuracies printed before final evaluation (the last printed one) are rough, just for checking if the process is normal!!!\n')
        self.trainset = Qid2Data(
            self.__C, 
            self.__C.TRAIN_SPLITS,
            True
        )
        self.valset = Qid2Data(
            self.__C, 
            self.__C.EVAL_SPLITS,
            self.__C.EVAL_NOW,
            json.load(open(self.__C.EXAMPLES_PATH, 'r'))
        )

        # if 'aok' in self.__C.TASK:
        #     from evaluation.aokvqa_evaluate import AOKEvaluater as Evaluater
        # else:
        #     from evaluation.okvqa_evaluate import OKEvaluater as Evaluater
        # evaluater = Evaluater(
        #     self.valset.annotation_path,
        #     self.valset.question_path
        # )

        infer_times = self.__C.T_INFER
        N_inctx = self.__C.N_EXAMPLES
        
        print()

        for qid in progress.track(self.valset.qid_to_data, description="Working...  "):
            if qid in self.cache:
                continue
            ques = self.valset.get_question(qid)
            caption = self.valset.get_caption(qid)
            cands = self.valset.get_topk_candidates(qid, self.__C.K_CANDIDATES)

            prompt_query = self.sample_make(ques, caption, cands)
            example_qids = self.valset.get_similar_qids(qid, k=infer_times * N_inctx)
            random.shuffle(example_qids)

            prompt_info_list = []
            prompt_info_full_response_list = []
            ans_pool = {}
            # multi-times infer
            for t in range(infer_times):
                # print(f'Infer {t}...')
                prompt_in_ctx = self.get_context(example_qids[(N_inctx * t):(N_inctx * t + N_inctx)])
                prompt_text = prompt_in_ctx + prompt_query
                # print("prompt_text: ", prompt_text)
                # gen_text, gen_prob = self.gpt3_infer(prompt_text)
                if write_only_prompt:
                    prompt_info = {
                        'prompt': prompt_text,
                        'answer': '',
                        'confidence': 0
                    }
                    prompt_info_list.append(prompt_info)
                else:
                    gen_text, gen_prob, full_response = self.gemma_infer(prompt_text)
                    # gen_text, gen_prob, full_response = self.mistral_infer(prompt_text)
                    print("gen_text: ", gen_text)
                    print("gen_prob: ", gen_prob)
                    ans = self.evaluater.prep_ans(gen_text)
                    print("ans: ", ans)
                    if ans != '':
                        ans_pool[ans] = ans_pool.get(ans, 0.) + gen_prob

                    prompt_info = {
                        'prompt': prompt_text,
                        'answer': gen_text,
                        'confidence': gen_prob
                    }
                    prompt_info_full_response = {
                        # 'prompt': prompt_text,
                        'answer': gen_text,
                        'confidence': gen_prob,
                        'full_response': full_response
                    }
                    prompt_info_list.append(prompt_info)
                    prompt_info_full_response_list.append(prompt_info_full_response)
                # time.sleep(self.__C.SLEEP_PER_INFER)
                # exit(0)
            print("\n\n")
            # vote
            if len(ans_pool) == 0:
                answer = self.valset.get_topk_candidates(qid, 1)[0]['answer']
            else:
                answer = sorted(ans_pool.items(), key=lambda x: x[1], reverse=True)[0][0]
            
            self.evaluater.add(qid, answer)
            self.cache[qid] = {
                'question_id': qid,
                'answer': answer,
                'prompt_info': prompt_info_list
            }
            self.cache_without_prompt[qid] = {
                'question_id': qid,
                'answer': answer,
                'prompt_info': prompt_info_full_response_list
            }
            json.dump(self.cache, open(self.cache_file_path, 'w'))
            json.dump(self.cache_without_prompt, open(self.cache_file_path.replace('.json', '_without_prompt.json'), 'w'))
            
            ll = len(self.cache)
            if self.__C.EVAL_NOW and not self.__C.DEBUG:
                if ll > 21 and ll % 10 == 0:
                    rt_accuracy = self.valset.rt_evaluate(self.cache.values())
                    info_column.info = f'Acc: {rt_accuracy}'

        self.evaluater.save(self.__C.RESULT_PATH)
        if self.__C.EVAL_NOW:
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                self.evaluater.evaluate(logfile)
        
def prompt_login_args(parser):
    parser.add_argument('--debug', dest='DEBUG', help='debug mode', action='store_true')
    parser.add_argument('--resume', dest='RESUME', help='resume previous run', action='store_true')
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, default='ok')#, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, default="okvqa_prompt_1")#, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, default='configs/prompt.yml')
    parser.add_argument('--examples_path', dest='EXAMPLES_PATH', help='answer-aware example file path, default: "assets/answer_aware_examples_for_ok.json"', type=str, default="assets/answer_aware_examples_okvqa.json")
    parser.add_argument('--candidates_path', dest='CANDIDATES_PATH', help='candidates file path, default: "assets/candidates_for_ok.json"', type=str, default="assets/candidates_okvqa.json")
    parser.add_argument('--captions_path', dest='CAPTIONS_PATH', help='captions file path, default: "assets/captions_for_ok.json"', type=str, default="assets/captions_okvqa.json")
    parser.add_argument('--openai_key', dest='OPENAI_KEY', help='openai api key', type=str, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heuristics-enhanced Prompting')
    prompt_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)


    evaluater = OKEvaluater(
        __C.EVAL_ANSWER_PATH,
        __C.EVAL_QUESTION_PATH,
    )
    runner = Runner(__C, evaluater)
    runner.run(write_only_prompt=False)
