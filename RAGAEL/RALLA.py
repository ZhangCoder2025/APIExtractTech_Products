# -*- coding: utf-8 -*-
# @File : RALLA.py
# @Author : zhang-Double
# @Time : 2025/03/16 16:30:56

from final_utils import parse_to_list
import os
import numpy as np
from tqdm import tqdm
import json
from final_utils import load_complete_datasets
from final_utils import Preprocess_API_mention
from queue import Queue
import threading
from LLM_utils import send_request_with_retry
import time
from Thread import Thread
from final_utils import Process_API_with_imports
from final_utils import Process_API_with_assigns
from final_utils import fetch_thread_content_from_SO
from final_utils import process_string_by_adding_special_markers
from final_utils import construct_var_source_chain
from final_utils import remove_code_blocks
from twokenize import tokenize
from final_utils import jaro_winkler_similarity
from LLM_utils import generate_API_link_for_RALLA
from final_utils import match_simplename_candidate



def typo_correction(api_mention,API_full_name_list,entity_to_index,similarity_table):
    candidate_list = []
    api_mention_entries = api_mention.split('.')
    jaro_winkler_dict = {}
    for cur_standard_api in API_full_name_list:
        cur_sim = 0
        standard_api_entries = cur_standard_api.split('.')
        if(len(standard_api_entries)>=len(api_mention_entries)):
            compare_entries = standard_api_entries[-(len(api_mention_entries)):]
            for cur_mention,cur_standard in zip(api_mention_entries,compare_entries):
                i, j = entity_to_index.get(cur_mention, -1), entity_to_index.get(cur_standard, -1)
                if i == -1 or j == -1:
                    cur_sim += jaro_winkler_similarity(cur_mention,cur_standard)
                else:
                    cur_sim += similarity_table[i, j]
            jaro_winkler_dict[cur_standard_api] = cur_sim / len(compare_entries)

    # 获取前20大的值（去重处理）
    unique_top_values = sorted(set(jaro_winkler_dict.values()), reverse=True)[:20]
    for cur_api,cur_value in jaro_winkler_dict.items():
        if(cur_value in unique_top_values):
            candidate_list.append(cur_api)
    return candidate_list

def rank_candidates_by_text(cur_mention_token_index,text_scope_tokens,candidate_list):
    score_candidate_dict = {}
    for cur_candidate in candidate_list:
        score_candidate_dict[cur_candidate] = 0
        candidate_entries = cur_candidate.split('.')
        for cur_token_index,cur_token in enumerate(text_scope_tokens):
            # preprocess tokens
            if(cur_token_index!=cur_mention_token_index):
                if(cur_token.startswith('API_')):
                    cur_token = cur_token[4:]
                cur_token = Preprocess_API_mention(cur_token).lstrip('.')
            
            
                cur_token_entries = cur_token.split('.')
                for token_entry in cur_token_entries:
                    if(token_entry in candidate_entries):
                        score_candidate_dict[cur_candidate] += (1/abs(cur_token_index-cur_mention_token_index))
    sorted_score_candidate_dict = dict(sorted(score_candidate_dict.items(), key=lambda item: item[1],reverse=True))
    return sorted_score_candidate_dict



def check_constraint(api_mention_list,constraint_api_mention_list,LLM_predictions_list,API_database_list):
    if(len(api_mention_list)!=len(LLM_predictions_list)):
        return "The number of predicted results is inconsistent with the number of given API mentions. Please re-infer the full API names. The output format must be consistent with before!!!"
    else:
        if("None" in LLM_predictions_list or 'None' in LLM_predictions_list):
            return  "There is 'None' in the inference result. Please re-infer to ensure that each API mention has a corresponding API full name."
        else:
            fake_api_list = []
            for cur_predict_api in LLM_predictions_list:
                if(cur_predict_api not in API_database_list):
                    fake_api_list.append(cur_predict_api)
            if(len(fake_api_list)>0):
                generate_msg = "In the full name of the API obtained by inference,"
                for cur_fake_api in fake_api_list:
                    generate_msg += "'{}', ".format(cur_fake_api)
                generate_msg += "are not real APIs. Please re-infer the corresponding full names of these API mentions based on the candidate list, and output the corresponding API full names of all API mentions, including the API full names that do not need to be re-inferred."
                return generate_msg
            else:
                not_match_list = []
                for mention_index,cur_api_mention in enumerate(api_mention_list):
                    if(not (LLM_predictions_list[mention_index].endswith(constraint_api_mention_list[mention_index]))):
                        not_match_list.append(mention_index)
                if(len(not_match_list)==0):
                    return "PASS"
                else:
                    generate_msg = "According to the analysis of the code snippet of the given Stack Overflow post, "
                    for cur_mention,cur_constraint in zip(api_mention_list,constraint_api_mention_list):
                        generate_msg += "the API full name of '{}' should end with '{}',".format(cur_mention,cur_constraint)
                    generate_msg += "Please refer to these information to re-infer the API full names of these API mentions."
                    return generate_msg

def generate_mention_endswith(api_mention,API_database_list):
    processed_api_mention = Preprocess_API_mention(api_mention).lstrip('.')
    mention_api_entries = processed_api_mention.split('.')
    reversed_mention_api_entries = list(reversed(mention_api_entries))
    match_api_entry = []
    for cur_entry in reversed_mention_api_entries:
        try_api_entry = '.'.join([cur_entry] + list(reversed(match_api_entry)))
        if any(item.endswith(".{}".format(try_api_entry)) for item in API_database_list):
            match_api_entry.append(cur_entry)
        elif try_api_entry in API_database_list:
            match_api_entry.append(cur_entry)
    constrainted_api = '.'.join(list(reversed(match_api_entry)))
    return constrainted_api

def RALLA_generate_prompt(Prompt_record_file="RALLA_complete.npy",Dataset_file="small_API_threads.jsonl",alias_flag=True,typo_flag = True,rank_flag = True):
    prompts_with_ids = []

    API_post_data = load_complete_datasets(Dataset_file)
    
    with open("return_value_dict.json", "r", encoding="utf-8") as json_file:
        return_value_dict = json.load(json_file)
    
    with open('deDup_multi_Source_Aliases.json', "r", encoding="utf-8") as json_file:
        loaded_aliases_dict = json.load(json_file)
    
    with open("5libs_candidate.json") as f:
        API_database_list = json.load(f)
    
    API_term_list = []
    for cur_standard_api in API_database_list:
        API_term_list.extend(cur_standard_api.split('.'))
    API_term_list = list(set(API_term_list))
        
    # 加载 entity_to_index
    with open('entity_to_index.json', 'r') as f:
        entity_to_index = json.load(f)

    # 加载相似度表
    similarity_table = np.load('similarity_table.npy')
    
    prompts_with_ids = []
    

    for element in tqdm(API_post_data,desc="Generating"):
        post_id = int(element['id'])
        
        post_content = element['post']
        sample_label = element['label']
        #post_title = element['title']
        
        API_mention_list = []
        for single_label in sample_label:
            if(single_label[2]=='Mention'):
                API_mention_list.append([post_content[single_label[0]:single_label[1]],(single_label[0],single_label[1])])

        post_thread = Thread(input_text= post_content,post_id=post_id)
        #post_thread = Thread(post_content,post_title,post_id)
        import_modules = post_thread.import_types
        
        variable_sources_dict = construct_var_source_chain(post_thread.import_types,post_thread.assign_body)
        
        
        # text tokens process
        process_text_blocks = process_string_by_adding_special_markers(post_content,[indices[1] for indices in API_mention_list],start_symbol='  API_',end_symbol='  ')

        cleaned_text_blocks = remove_code_blocks(process_text_blocks).replace('Text: ','').strip()
        
        text_scope_tokens = tokenize(cleaned_text_blocks)
        api_mention_token_index_list = []
        for token_index,cur_token in enumerate(text_scope_tokens):
            if(cur_token.startswith('API_')):
                api_mention_token_index_list.append(token_index)
        
        
        raw_api_mention_list = []
        api_mention_candidates = []
        constrainted_api_list = []
        for index, (api_mention,mention_index) in enumerate(API_mention_list):
            
            
            
            raw_api_mention_list.append(api_mention)             
            
            
            processed_api_mention = Preprocess_API_mention(api_mention).lstrip('.')
            #  step 1 Preprocess API mention alias replace
            if(alias_flag):
                processed_api_mention_with_imports = Process_API_with_imports(processed_api_mention,import_modules)
                processed_api_mention_with_assigns = Process_API_with_assigns(processed_api_mention_with_imports,variable_sources_dict,return_value_dict,loaded_aliases_dict,API_term_list)
            else:
                processed_api_mention_with_assigns = processed_api_mention
            if(processed_api_mention_with_assigns in API_database_list):
                prediction_fqn = processed_api_mention_with_assigns
                sorted_candidate_list = [prediction_fqn]
            else:
                # typo get candidate
                if(typo_flag):
                    candidate_list = typo_correction(processed_api_mention_with_assigns,API_database_list,entity_to_index,similarity_table)
                    
                    #candidate_list = typo_correction(api_mention,API_database_list,entity_to_index,similarity_table)
                else:
                    candidate_list = match_simplename_candidate(processed_api_mention_with_assigns,API_database_list)
                if(rank_flag):
                    # rank candidate list
                    cur_mention_token_index = api_mention_token_index_list[index]
                    sorted_score_candidate_dict = rank_candidates_by_text(cur_mention_token_index,text_scope_tokens,candidate_list)
                    sorted_candidate_list = [cur_api for cur_api, cur_value in sorted_score_candidate_dict.items()]
                else:
                    sorted_candidate_list = candidate_list
            api_mention_candidates.append(sorted_candidate_list)
            
            constrainted_api = generate_mention_endswith(api_mention,API_database_list)
            constrainted_api_list.append(constrainted_api)
        to_prompt_sentence = process_string_by_adding_special_markers(post_content,[indices[1] for indices in API_mention_list]).strip()
        
        to_prompt_candidate = ""
        for candidate_index, cur_candidates in enumerate(api_mention_candidates):
            to_prompt_candidate += "{}. {}\n".format(candidate_index+1,cur_candidates)
        
        msg = generate_API_link_for_RALLA(to_prompt_sentence,to_prompt_candidate)        
        prompts_with_ids.append({'id':post_id,'prompt':msg,'mention':raw_api_mention_list,'candidates':api_mention_candidates,'constraint':constrainted_api_list})
    np.save(Prompt_record_file,prompts_with_ids)



def RALLA_pipline(Record_result_file="chatgpt_APIlink_with_Complete_1218.npy",Dataset_file="small_API_threads.jsonl",specific_post_id=None,Prompt_record_file="RALLA_wo_typo_prompt.npy"):   
    if(os.path.exists(Record_result_file)):
        store_results_dict = np.load(Record_result_file,allow_pickle=True).item()
    else:
        store_results_dict = {}
    with open("5libs_candidate.json") as f:
        API_database_list = json.load(f)
    
    prompts_with_ids = np.load(Prompt_record_file,allow_pickle=True)
    # 定义存储结果的字典
    #results = {}
    results_lock = threading.Lock()  # 用于线程安全地访问 results 字典
    # 定义处理单个提示的函数
    def process_prompt(prompt_id, prompt, pbar,api_mention_list,constraint_api_list,prompt_candidates):
        try:
            #response = send_request_with_retry(prompt, max_retries=5, retry_interval=60,model_type="gpt-4o-2024-08-06",Link_type="nixiang")
            response = send_request_with_retry(prompt,max_retries=5,retry_interval=60,model_type="deepseek-v3-241226")
            response_text = response['choices'][0]['message']['content']
            
            annotation_result = parse_to_list(response_text)
            if(annotation_result is None):
                annotation_result = [None for element in api_mention_list]
            
            output_format_requirements = """### Output Requirements:
            (1) Output the full name of each API mention in a list in order of appearance in the post, without any explanatory or introductory content!!!
            (2) The number of elements in the output list must be equal to the number of API mentions with special tags!!!"""
            
            constraint_check_result = check_constraint(api_mention_list,constraint_api_list,annotation_result,API_database_list)
            if(constraint_check_result!="PASS"):
                constraint_check_cnt = 5
                constraint_prompt = list(prompt)
                while(constraint_check_cnt > 0):
                    constraint_check_cnt -= 1
                    # construct message history
                    constraint_prompt.append({'role':'assistant','content':response_text})
                    constraint_prompt.append({'role':'user','content':constraint_check_result+'\n\n'+output_format_requirements})
                    #response = send_request_with_retry(prompt, max_retries=5, retry_interval=60,model_type="gpt-4o-2024-08-06",Link_type="nixiang")
                    response = send_request_with_retry(prompt,max_retries=5,retry_interval=60,model_type="deepseek-v3-241226")
                    response_text = response["choices"][0]["message"]["content"]
                    annotation_result = parse_to_list(response_text)
                    if(annotation_result is None):
                        annotation_result = [None for element in api_mention_list]
                    constraint_check_result = check_constraint(api_mention_list,constraint_api_list,annotation_result,API_database_list)
                    if(constraint_check_result=="PASS"):
                        break
              
            # 使用锁确保线程安全地更新 results 字典
            with results_lock:
                # store_results_dict[prompt_id] = {
                #     "prompt": prompt,
                #     "response": annotation_result
                # }
                store_results_dict[prompt_id] = annotation_result
                
            # 更新进度条
            pbar.update(1)
        except Exception as e:
            print(f"Error processing prompt ID {prompt_id}: {e}")

    # 定义工作线程函数
    def worker(pbar):
        while not request_queue.empty():
            task = request_queue.get()  # 从队列中获取任务
            prompt_id = task["id"]
            prompt = task["prompt"]
            api_mention_list = task["mention"]
            prompt_candidates = task["candidates"]
            constraint_api_list = task['constraint']
            #process_prompt(prompt_id, prompt, pbar)  # 处理提示
            process_prompt(prompt_id, prompt, pbar,api_mention_list,constraint_api_list,prompt_candidates)
            request_queue.task_done()  # 标记任务完成
    # 创建请求队列
    request_queue = Queue()
    for task in prompts_with_ids:
        request_queue.put(task)  # 将所有任务放入队列
    # 计算请求间隔时间
    rpm_limit = 500  # 每分钟最多 500 个请求
    request_interval = 60 / rpm_limit  # 每个请求之间的间隔时间（秒）    
    
    
    # 创建进度条
    total_tasks = len(prompts_with_ids)
    with tqdm(total=total_tasks, desc="Processing", unit="task") as pbar:
        # 创建并启动线程池
        threads = []
        max_threads = 50  # 最大并发线程数（根据需求调整）
        for _ in range(max_threads):
            thread = threading.Thread(target=worker, args=(pbar,))
            thread.start()
            threads.append(thread)
            time.sleep(request_interval)  # 控制线程启动速率

        # 等待队列中的所有任务完成
        request_queue.join()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

    print("All tasks completed.")
    np.save(Record_result_file,store_results_dict) 

 

if __name__=='__main__':

    
    RALLA_generate_prompt(Prompt_record_file="RALLA_complete_prompt_top20_0625.npy",Dataset_file="api_link.jsonl",alias_flag=True,typo_flag = True,rank_flag = True)
    
    RALLA_pipline(Record_result_file="Records/RALLA_complete_prompt_top20_0625.npy",Dataset_file='api_link.jsonl',specific_post_id=None,Prompt_record_file="RALLA_complete_prompt_top20_0625.npy")