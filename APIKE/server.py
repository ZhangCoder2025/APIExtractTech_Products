# server.py

from Preprocess import text_Process
from CARLDA.API_recognizer import recognize_api_within_text
from RAGAEL.API_link import link_api_with_FQN
from CEDCC.Entity_Detect import CEDCC_step_1
from CEDCC.Entity_Detect import CEDCC_step_2
import os
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pymysql
import subprocess
import json
import re
import copy


app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# —— 在这里声明全局变量 —— 
tokenized_sentences = {}
tagged_post_content = []
api_recognized_sentences = {}
context_dependent_detect_results = []

organize_linked = []
organize_ctx_dep = []


def extract_red_spans(html_string):
    pattern = r'<strong><span style="color:red">(.*?)</span></strong>'
    match = re.search(pattern, html_string)
    return match.group(1) if match else None

def process_string_by_adding_special_markers(input_string, index_list):
    filtered_list = []
    for i in range(len(index_list)):
        _, (start_i, end_i) = index_list[i]
        is_nested = False
        for j in range(len(index_list)):
            _, (start_j, end_j) = index_list[j]
            if i != j and start_j <= start_i and end_i <= end_j:
                is_nested = True
                break
        if not is_nested:
            filtered_list.append(index_list[i])
    
    filtered_list.sort(key=lambda x: x[1][0], reverse=True)

    for type_str, (start, end) in filtered_list:
        marked_text = input_string[start:end]
        if type_str == 'API':
            marked_text = f'<strong><span style="color:red">{marked_text}</span></strong>'
        else:
            marked_text = f'<strong><span style="color:blue">{marked_text}</span></strong>'
        input_string = input_string[:start] + marked_text + input_string[end:]
    
    return input_string

def remove_specific_highlight_tags(text):
    # 去除 <strong> 和 </strong>
    text = re.sub(r'</?strong>', '', text)
    # 去除 <span style="color:..."> 和 </span>
    text = re.sub(r'<span style="color:[^"]+">', '', text)
    text = re.sub(r'</span>', '', text)
    return text


@app.route('/')
def index():
    # 返回你的前端页面
    return send_from_directory('.', 'index.html')

@app.route('/text_process', methods=['POST'])
def text_process():
    global tokenized_sentences
    global tagged_post_content
    text = request.json.get('text', '')
    #raw_post_content = text
    # TODO: 在这里填充你的 TEXT_process 逻辑
    tokenized_sentences,tagged_post_content = text_Process.text_process_func(text)
    showed_processed_tokens = [element for _, element in tokenized_sentences.items()]
    # 这里为了示例直接返回行列表
    return jsonify({'sentences': showed_processed_tokens})

@app.route('/api_recognition', methods=['POST'])
def api_recognition():
    global api_recognized_sentences
    api_recognized_sentences = {}
    # TODO: 在这里实现你的 API_recognition 逻辑
    entities = []
    for sent_index, cur_sent in tokenized_sentences.items():
        args_json = json.dumps(cur_sent)
        result = subprocess.run(
        ["F:/ProgramData/Miniconda3/envs/ptorch2/python.exe", "CARLDA/API_recognizer.py", args_json],
        capture_output=True,
        text=True
        )
        output = json.loads(result.stdout.strip())
        if(len(output)>0):
            for api_index in output:
                cur_sent[api_index] = '<strong><span style="color:red">{}</span></strong>'.format(cur_sent[api_index])
            entities.append(' '.join(cur_sent))
            api_recognized_sentences[sent_index] = output
        else:
            api_recognized_sentences[sent_index] = []
    return jsonify({'api_entities': entities})

@app.route('/api_link', methods=['POST'])
def api_link():

    global organize_linked
    organize_linked = []
    
    API_mention_list = []
    link_post_content = ""
    for cur_sent_index, cur_sent in enumerate(tagged_post_content):
        if(cur_sent_index in api_recognized_sentences.keys()):
            cur_api_index_list = api_recognized_sentences[cur_sent_index]
            updated_sent = "Text: "
            for cur_token_index, cur_token in enumerate(tokenized_sentences[cur_sent_index]):
                if(cur_token_index in cur_api_index_list):
                    cur_token = extract_red_spans(cur_token)
                    cur_api_mention_begin_index = len(updated_sent)+len(link_post_content)
                    cur_api_mention_end_index = cur_api_mention_begin_index + len(cur_token)
                    API_mention_list.append([cur_token, (cur_api_mention_begin_index, cur_api_mention_end_index)])
                    updated_sent += (cur_token + ' ')
                else:
                    updated_sent += (cur_token + ' ')
            link_post_content += updated_sent+'\n'
        else:
            link_post_content += cur_sent+'\n'    
    # TODO: 在这里实现你的 API_LINK 逻辑
    link_result = link_api_with_FQN(link_post_content,API_mention_list)
    linked = []
    for api_entity,api_full_name in zip(API_mention_list,link_result):
        linked.append({'entity': '{}'.format(api_entity[0]),   'fqn': '{}'.format(api_full_name)})
    organize_linked = copy.deepcopy(linked)
    return jsonify({'linked': linked})

@app.route('/cedcc', methods=['POST'])
def cedcc():
    global context_dependent_detect_results
    global organize_ctx_dep
    organize_ctx_dep = []
    context_dependent_detect_results = []
    linked = request.json.get('linked', [])
    raw_post_content = request.json.get('text', [])
    ctx_dep = []
    for cur_sent_index, api_rec_results in api_recognized_sentences.items():
        if(len(api_rec_results)>0):
            
            
            for cur_api_index in api_rec_results:
                tokenized_sentences[cur_sent_index][cur_api_index] = extract_red_spans(tokenized_sentences[cur_sent_index][cur_api_index])

            api_tokens_indices_list = []
            sample_sentence = ""
            for cur_index, cur_token in enumerate(tokenized_sentences[cur_sent_index]):
                
                if cur_index == 0:
                    if(cur_index in api_rec_results):
                        api_tokens_indices_list.append(['API',(len(sample_sentence),len(sample_sentence)+len(cur_token))])
                    sample_sentence += cur_token
                else:
                    if(cur_index in api_rec_results):
                        api_tokens_indices_list.append(['API',(len(sample_sentence)+1,len(sample_sentence)+1+len(cur_token))])
                    sample_sentence += " " + cur_token
            
            #sample_sentence = ' '.join(tokenized_sentences[cur_sent_index])
            unclear_tokens_indices_list = CEDCC_step_1(sample_sentence=sample_sentence,context_body=raw_post_content)
            context_dependent_detect_results.append(unclear_tokens_indices_list)
            ctx_dep.append(process_string_by_adding_special_markers(sample_sentence,api_tokens_indices_list+unclear_tokens_indices_list))
    # TODO: 在这里实现你的 CEDCC 逻辑
    # ctx_dep = [
    #     'To fit a parabola to <strong><span style="color:blue">those points</span></strong>, use <strong><span style="color:red">numpy.polyfit()</span></strong>'
    # ]
    organize_ctx_dep = copy.deepcopy(ctx_dep)
    return jsonify({'context_dependent': ctx_dep})

@app.route('/context_complete', methods=['POST'])
def context_complete():
    ctx_dep = request.json.get('context_dependent', [])
    context_body = request.json.get('text', [])
    completions = []
    
    for depend_entities,to_complete_sentence in zip(context_dependent_detect_results,ctx_dep):
        if(len(depend_entities)>0):
            sample_sentence = remove_specific_highlight_tags(to_complete_sentence)
            completed_context = CEDCC_step_2(sample_sentence,depend_entities,context_body)
            for cur_complete_result in completed_context:
                completions.append({'entity':cur_complete_result['NP'],'context':cur_complete_result['Value']})
    # TODO: 在这里实现你的 Context_complete 逻辑
    # completions = [
    #     {'entity': 'those points', 'context': 'data points defined by two numpy arrays: x and y'}
    # ]
    return jsonify({'completions': completions})





@app.route('/knowledge_organize', methods=['POST'])
def knowledge_organize():
    def extract_red_mentions(html_string):
        # 匹配所有 <span style="color:red">...</span> 中的内容
        pattern = r'<span style="color:red">(.*?)</span>'
        api_mentions = re.findall(pattern, html_string)
        return api_mentions

    def extract_blue_mentions(html_string):
        # 匹配所有 <span style="color:red">...</span> 中的内容
        pattern = r'<span style="color:blue">(.*?)</span>'
        api_mentions = re.findall(pattern, html_string)
        return api_mentions

    def replace_red_mentions(html_string, replacement_list):
        pattern = r'<span style="color:red">(.*?)</span>'
        
        def replacer(match):
            # 使用 replacer.counter 来记录当前替换到哪个 index
            replacement = replacement_list[replacer.counter]
            replacer.counter += 1
            return f'<span style="color:red">{replacement}</span>'
        
        replacer.counter = 0  # 初始化计数器
        result = re.sub(pattern, replacer, html_string)
        return result
    """
    在所有五个阶段跑完后，前端再调用这个接口来拿最终 {api, snippet} 列表
    """
    #ctx_dep = request.json.get('context_dependent', [])
    
    #linked = request.json.get('linked', [])
    completions = request.json.get('completions', [])
    
    
    # ctx_dep = request.json.get('context_dependent', [])
    # linked = request.json.get('linked', [])
    # completions = request.json.get('completions', [])
    
    organized = []
    
    api_count = 0
    context_dependent_count = 0

    for raw_knowledge_snippet in organize_ctx_dep:
        api_fqn_list = []
        api_mention_list = extract_red_mentions(raw_knowledge_snippet)
        context_dependent_entity_list = extract_blue_mentions(raw_knowledge_snippet)

        for cur_api_mention in api_mention_list:
            api_fqn_list.append(organize_linked[api_count]['fqn'])
            api_count += 1
        replaced_knowledge_snippets = replace_red_mentions(raw_knowledge_snippet,api_fqn_list)
        show_knowledge_snippet = remove_specific_highlight_tags(replaced_knowledge_snippets)

        if(len(context_dependent_entity_list)>0):
            for cur_dep_entity in context_dependent_entity_list:
                cur_complete_context = completions[context_dependent_count]
                show_knowledge_snippet += ("\n'{}' refers to {}".format(cur_complete_context['entity'],cur_complete_context['context']))
                context_dependent_count += 1

        for cur_api_fqn in api_fqn_list:
            organized.append({'api': cur_api_fqn, 'snippet': show_knowledge_snippet})
    
    return jsonify({'data': organized})
    
    


@app.route('/store_extraction', methods=['POST'])
def store_extraction():
    # 从前端接收的列表，每项形如 {'api': ..., 'snippet': ...}
    records = request.json.get('data', [])
    if not records:
        return jsonify({'error': 'No data'}), 400

    # 连接数据库，修改成你的连接信息
    conn = pymysql.connect(
        host='10.0.0.193',
        port=3306,
        user='root',
        password='0.mysql',
        database='APIKE_Repository',
        charset='utf8mb4'
    )
    try:
        with conn.cursor() as cur:
            sql = "INSERT INTO API_Knowledge (API, Knowledge) VALUES (%s, %s)"
            for rec in records:
                api = rec.get('api', '').strip()
                snippet = rec.get('snippet', '').strip()
                if api and snippet:
                    # 去掉 HTML 标签
                    clean_snippet = remove_specific_highlight_tags(snippet)
                    cur.execute(sql, (api, clean_snippet))
        conn.commit()
    finally:
        conn.close()

    return jsonify({'status': 'ok', 'count': len(records)})



@app.route('/query_api', methods=['POST'])
def query_api():
    api = request.json.get('api', '').strip()
    if not api:
        return jsonify({'results': []})

    # 根据你的 MySQL 连接信息修改 host/user/password/database
    conn = pymysql.connect(
        host='10.0.0.193',
        port=3306,
        user='root',
        password='0.mysql',
        database='APIKE_Repository',
    )
    try:
        with conn.cursor() as cur:
            # 精确匹配，或改成 LIKE %s%% if 模糊查询
            sql = "SELECT API, Knowledge FROM API_Knowledge WHERE API = %s"
            cur.execute(sql, (api,))
            rows = cur.fetchall()
    finally:
        conn.close()

    # 保证最多 5 条
    rows = rows[:5]
    results = [{'API': r[0], 'snippet': r[1]} for r in rows]
    return jsonify({'results': results})

@app.route('/query_snippet', methods=['POST'])
def query_snippet():
    kw = request.json.get('snippet', '').strip()
    if not kw:
        return jsonify({'results': []})

    conn = pymysql.connect(
        host='10.0.0.193',
        port=3306,
        user='root',
        password='0.mysql',
        database='APIKE_Repository',
    )
    try:
        with conn.cursor() as cur:
            # 模糊匹配 Knowledge 字段
            sql = "SELECT API, Knowledge FROM API_Knowledge WHERE Knowledge LIKE %s"
            cur.execute(sql, ('%' + kw + '%',))
            rows = cur.fetchall()
    finally:
        conn.close()

    # 只保留前 5 条
    rows = rows[:5]
    results = [{'API': r[0], 'snippet': r[1]} for r in rows]
    return jsonify({'results': results})


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    # 从前端拿到 feedback 字符串
    fb_text = request.json.get('feedback', '').strip()
    if not fb_text:
        return jsonify({'error': 'Empty feedback'}), 400

    # 准备 feedback.json 文件路径
    fb_file = 'feedback.json'
    # 如果文件存在就读取，否则初始化列表
    if os.path.exists(fb_file):
        with open(fb_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # 生成新的记录
    new_id = data[-1]['id'] + 1 if data else 1
    new_record = {
        'id': new_id,
        'timestamp': datetime.now().isoformat(),
        'feedback': fb_text
    }
    data.append(new_record)

    # 写回文件
    with open(fb_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return jsonify({'status': 'ok', 'id': new_id})



if __name__ == '__main__':
    app.run(debug=True, port=5000)
