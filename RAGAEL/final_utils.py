import json
import re
import pymysql
import html
from bs4 import BeautifulSoup
import numpy as np
import jellyfish


def remove_parentheses_content(s):
    result = []
    skip = False

    for char in s:
        if char == '(':
            skip = True
        elif char == ')':
            skip = False
        elif not skip:
            result.append(char)
    
    return ''.join(result)

def clean_jupyter_markup(text):
    """
    提取 Jupyter Notebook 或 Python 交互式代码中的输入部分，并处理纯 Python 代码片段。
    
    参数:
    text (str): 包含交互式代码或纯代码的文本。
    
    返回:
    str: 提取的代码文本，忽略 'Out' 部分。
    """
    input_lines = []
    in_input_block = False  # 用于跟踪是否处于 In 输入块
    in_python_prompt = False  # 用于跟踪是否处于 >>> 块
    is_interactive = False  # 检测是否为交互式代码片段

    
    # 定义一个变量来标记多行注释的开始和结束
    in_multiline_comment = False
    
    
    for line in text.split('\n'):
        # 处理 'In' 块
        #if re.match(r"In\s*\[\d+\]:", line):  # 检测 'In' 行
        if re.match(r"In\s*\[\d*\]:", line):
            in_input_block = True  # 进入 In 输入块
            in_python_prompt = False  # 确保非 Python 提示符块
            is_interactive = True  # 标记为交互式
            try:
                code_part = line.split(':', 1)[1].strip()  # 提取 ': ' 后面的代码部分
                input_lines.append(code_part)
            except IndexError:
                continue
            
        #elif re.match(r"Out\s*\[\d+\]:", line):  # 遇到 'Out' 行，停止保存输入
        elif re.match(r"Out\s*\[\d*\]:", line):
            in_input_block = False  # 退出 In 输入块
        elif in_input_block:  # 保留 In 输入块的内容
            # 检查是否遇到 Python 提示符 '>>>'
            if re.match(r"^\s*>>>", line):
                in_input_block = False  # 停止处理 In 输入块
                code_part = re.sub(r"^\s*>>>", "", line).strip()  # 去掉 '>>>'
                input_lines.append(code_part)
                in_python_prompt = True  # 进入 Python 提示符块
            else:
                input_lines.append(line.strip())
        
        # 处理 '>>>' 块
        elif re.match(r"^\s*>>>", line):  # 检测 Python 提示符 '>>>'
            code_part = re.sub(r"^\s*>>>", "", line).strip()  # 去掉 '>>>'
            input_lines.append(code_part)
            in_python_prompt = True  # 进入 Python 提示符块
            is_interactive = True  # 标记为交互式
        elif in_python_prompt:  # 保留 Python 提示符块的内容
            if re.match(r"^\s*\.\.\.", line):  # 去掉 '...' 提示符
                code_part = re.sub(r"^\s*\.\.\.", "", line).strip()
                input_lines.append(code_part)
            else:
                in_python_prompt = False  # 遇到非 '...' 行则退出 Python 提示符块
        
        # 如果不是交互式代码片段，直接保留所有行
        elif not is_interactive:
            # # 检查多行注释的开始和结束
            # if re.search(r'^\s*["\']{3}', line):
            #     in_multiline_comment = not in_multiline_comment
            #     continue
            # if in_multiline_comment:
            #     continue
            # 移除行末注释
            line = re.split(r'\s*#', line)[0].rstrip()
            
            if not re.match(r"^\s*\.\.\.", line) and line:
                # input_lines.append(line.strip())
                input_lines.append(line)
    # 移除多余的空行
    input_lines = [line for line in input_lines if line]

    return '\n'.join(input_lines)

def parse_to_list(input_string):
    '''
    # 测试示例
    test_strings = [
        "[pandas.DataFrame, pandas.DataFrame.ix]",
        "1. matplotlib.pyplot.yticks\n2. matplotlib.pyplot.sca\n3. matplotlib.pyplot.yticks\n4. matplotlib.pyplot\n5. matplotlib.axes\n6. matplotlib.axes.Axes.tick_params\n7. matplotlib.pyplot.subplots",
        "```\n[matplotlib.pyplot.yscale, matplotlib.pyplot.minorticks_on]\n```",
        "\\[ \\text{['sklearn.pipeline.Pipeline']} \\]",
        "[\n    \"sklearn.base.BaseEstimator\",\n    \"sklearn.base.BaseEstimator.get_params\",\n    \"sklearn.base.BaseEstimator.set_params\",\n    \"sklearn.model_selection.GridSearchCV\",\n    \"sklearn.pipeline.Pipeline\"\n]",
        "```json\n[\"scipy.sparse.hstack\"]\n```",
        "```json\n[pandas.DataFrame.iloc]\n```",
        "\\[ \\text{{['sklearn.preprocessing']}} \\]",
        '```plaintext\n[ast.literal_eval]\n```',
        '```python\n[matplotlib.axes.Axes.legend]\n```',
        '```plaintext\n[scipy.optimize.curve_fit, scipy.optimize.curve_fit]\n```',
        "[None]",
        "['matplotlib.pyplot.plot', 'matplotlib.lines.Line2D', 'None']",
        "[pylab.xticks, pylab.xticks]"
    ]

    for s in test_strings:
        print(parse_to_list(s))
    '''
    
    input_string = input_string.strip()
    
    # 检查是否被 ```json 包裹
    if input_string.startswith("```json") and input_string.endswith("```"):
        # 去掉外层的 ```json 和 ``` 并获取内容
        input_string = input_string[7:-3].strip()
        try:
            # 直接尝试解析为合法的 JSON 数据
            elements = json.loads(input_string)
        except json.JSONDecodeError:
            # 如果解析失败，修复潜在的非标准 JSON 格式
            if input_string.startswith("[") and input_string.endswith("]"):
                fixed_json = input_string.replace("[", '["').replace("]", '"]').replace(", ", '", "')
                elements = json.loads(fixed_json)
            else:
                raise ValueError(f"Cannot parse JSON-like string: {input_string}")
        # 确保每个元素只保留单层引号包围
        elements = [item.strip().strip("'").strip('"') for item in elements]
        return elements
    
    # 检查是否被 ```plaintext 包裹
    if input_string.startswith("```plaintext") and input_string.endswith("```"):
        # 去掉外层的 ```plaintext 和 ``` 并获取内容
        input_string = input_string[len("```plaintext"):-3].strip()
        # 检查是否是方括号包裹的格式
        if input_string.startswith("[") and input_string.endswith("]"):
            # 去除外层括号并按逗号分隔
            elements = [item.strip().strip("'").strip('"') for item in input_string[1:-1].split(",")]
            return elements
    
    
    # 检查是否被 ```python 包裹
    if input_string.startswith("```python") and input_string.endswith("```"):
        # 去掉外层的 ```python 和 ``` 并获取内容
        input_string = input_string[len("```python"):-3].strip()
        # 检查是否是方括号包裹的格式
        if input_string.startswith("[") and input_string.endswith("]"):
            # 去除外层括号并按逗号分隔
            elements = [item.strip().strip("'").strip('"') for item in input_string[1:-1].split(",")]
            return elements
    
    
    # 检查是否被 ``` 包裹
    if input_string.startswith("```") and input_string.endswith("```"):
        # 去掉外层的 ``` 并继续解析
        input_string = input_string[3:-3].strip()
    
    
    # 检查是否是 LaTeX 风格的 \\[ \\text{} \\]
    if input_string.startswith("\\[") and input_string.endswith("\\]"):
        # 去掉外层的 \\[ 和 \\]
        input_string = input_string[2:-2].strip()
        # 检查是否包含 \\text{} 并提取内容
        if input_string.startswith("\\text{{") and input_string.endswith("}}"):
            input_string = input_string[7:-2].strip()  # 提取 \\text{{}} 内的内容
        elif input_string.startswith("\\text{") and input_string.endswith("}"):
            input_string = input_string[6:-1].strip()  # 提取 \\text{} 内的内容
    
    
    
    
    if input_string.startswith("[") and input_string.endswith("]"):
        input_string = input_string[1:-1]
        # Split by commas and strip each element
        elements = [item.strip().strip("'").strip('"') for item in input_string.split(",")]
        return elements

    elif "\n" in input_string:  # 判断是否是编号加点的换行分隔格式
        lines = input_string.split("\n")
        elements = []
        for line in lines:
            # 尝试提取编号后内容（支持以数字加点开头的行）
            if line.strip().startswith(tuple(f"{i}." for i in range(1, 10))):
                elements.append(line.split(".", 1)[1].strip())
            else:
                # 若不符合编号格式，直接添加（适配更广情况）
                elements.append(line.strip())
        return elements
    else:
        return None
def load_jsonl(file_path):
    data = []
    with open(file_path,'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data 
def extract_content_between_two_symbols(text,symbol_start,symbol_end):
    # 构建正则表达式
    pattern = re.escape(symbol_start) + r'\s*(.*?)\s*' + re.escape(symbol_end)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None
   
def load_complete_datasets(data_file):
    empir_data = load_jsonl(data_file)
    final_data_list=[]
    for cur_sample in empir_data:
        sample_text = cur_sample['text']
        sample_label = cur_sample['label']
        post_id = extract_content_between_two_symbols(sample_text,"ID:","Post:").strip()
        label_offset_index = sample_text.find('Post:')+len('Post:')
        post_content = sample_text[label_offset_index:sample_text.find('API:')]
        #post_content = extract_content_between_two_symbols(sample_text,"Post:","API:")
        
        process_labels = []
        for cur_label in sample_label:
            if(cur_label[2]!='post_id'):
                process_labels.append([cur_label[0]-label_offset_index,cur_label[1]-label_offset_index,cur_label[2]])
        API_list = []
        for cur_label in sample_label:
            if(cur_label[2]=='API'):
                API_list.append(sample_text[cur_label[0]:cur_label[1]])
        final_data_list.append({'id':post_id,'post':post_content,'API':API_list,'label':process_labels,'Raw_post':sample_text})
    return final_data_list    

def Preprocess_API_mention(api_mention):
    for i, char in enumerate(api_mention):
        if(char == '('):
            api_mention = api_mention[:i]
            break
    cur_mention_entries = api_mention.split('.')
    mention_simple_name = cur_mention_entries[-1]

    for i, char in enumerate(mention_simple_name):
        if(char == '(' or char == '[' or char == "'"):
            mention_simple_name = mention_simple_name[:i]
            break
    api_mention = '.'.join(cur_mention_entries[:-1]+[mention_simple_name])
    
    return api_mention


def Process_API_with_imports(api_mention,import_modules):
    mention_simple_name = api_mention.split('.')[-1]
    if(mention_simple_name in import_modules):
        cleaned_processed_api_mention = import_modules[mention_simple_name][0]
        return cleaned_processed_api_mention
    prefix_entry = api_mention.split(".")[:-1]
    
    if(len(prefix_entry)>0):
        if(prefix_entry[0] in import_modules.keys()):
            prefix_entry[0] = import_modules[prefix_entry[0]][0]
    processed_api_mention = '.'.join(prefix_entry+[mention_simple_name])
    return processed_api_mention

def get_variable_type(variable,variable_sources,return_value_dict,loaded_aliases_dict,API_term_list,visited=None):
    # 初始化递归路径记录
    if visited is None:
        visited = set()

    # 检查是否出现循环引用
    if variable in visited:
        return variable  # 返回变量本身以防止无限递归

    # 将当前变量加入路径
    visited.add(variable)
    
    
    # 如果变量的直接来源已知
    if variable in variable_sources:
        source_info = variable_sources[variable]
        
        # 如果是导入的模块
        if "source" in source_info and "index" not in source_info:
            return source_info["source"]
        
        # 如果是函数调用的返回值
        if "source" in source_info and "index" in source_info:
            func = source_info["source"]
            index = source_info["index"]

            # 递归获取函数的返回值类型
            func_type = get_variable_type(func,variable_sources,return_value_dict,loaded_aliases_dict,visited)

            # 如果函数类型是元组或列表（多返回值），根据 index 提取
            if isinstance(func_type, (tuple, list)):
                if 0 <= index < len(func_type):  # 防止越界
                    return func_type[index]
                return "Unknown"  # 无效的 index
            
            
            # 如果函数类型已知且返回多个值，提取对应位置
            if func_type in return_value_dict:
                func_return = return_value_dict[func_type]
                if isinstance(func_return, (tuple, list)):
                    return func_return[index]  # 返回指定位置的类型
                return func_return  # 如果返回单值，则直接返回该类型

            # else:
            #     print("wait")
            
            return func_type  # 返回已解析的类型，即使不在字典中

        # 如果只有 source，但没有 index
        if "source" in source_info:
            return get_variable_type(source_info["source"],variable_sources,return_value_dict,loaded_aliases_dict,visited)

    # 如果变量不在 variable_sources 中，尝试直接解析路径
    if "." in variable:
        # 分离调用者和成员
        obj, method = variable.rsplit(".", 1)
        obj_type = get_variable_type(obj,variable_sources,return_value_dict,loaded_aliases_dict,visited)

        # 如果调用者无法解析，返回变量本身
        if obj_type == variable:
            return variable

        # 如果调用者是一个多返回值函数
        if obj_type in return_value_dict:
            func_return = return_value_dict[obj_type]
            if isinstance(func_return, (tuple, list)):
                # 动态构造返回值的完整路径，假设所有多返回值的调用者都记录了 index
                for idx, value in enumerate(func_return):
                    if f"{obj}.{method}" == value:
                        return func_return[idx]
        # else:
        #     print("wait")
        # 动态构造完整路径
        full_method = f"{obj_type}.{method}"
        return return_value_dict.get(full_method, full_method)

    # if(variable in loaded_aliases_dict.keys()):
    #     replaced_aliaes = next(iter(loaded_aliases_dict[variable].items()))[0]
    #     return replaced_aliaes

    
    if(variable in loaded_aliases_dict.keys() and variable not in API_term_list and len(variable) > 1):
        max_aliases_frequency = next(iter(loaded_aliases_dict[variable].items()))[1]
        if(max_aliases_frequency >= 10):
            replaced_aliaes = next(iter(loaded_aliases_dict[variable].items()))[0]
            return replaced_aliaes
    
    
    # 如果变量仍然无法解析，直接返回变量本身
    return variable

def construct_var_source_chain(import_modules,assign_body):
    variable_sources = {}
    for cur_key,value_list in import_modules.items():
        for cur_value in value_list:
            variable_sources[cur_key] = {"source": cur_value}
    
    for cur_key,value_list in assign_body.items():
        for cur_value in value_list:
            cur_value_source = cur_value[0]
            cur_value_index = cur_value[1]
            variable_sources[cur_key] = {"source": cur_value_source, "index": cur_value_index}
    
    return variable_sources

def remove_code_blocks(text):
        cleaned_lines = []
        skip = False
        for line in text.splitlines():
            # 检查是否进入 Code 块
            if line.strip().startswith("Code:"):
                skip = True
                continue
            # 检查是否进入 Text 块
            if line.strip().startswith("Text:"):
                skip = False
            
            # 仅保留不在 Code 块中的行
            if not skip:
                cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)


def Process_API_with_assigns(api_mention,variable_sources_dict,return_value_dict,loaded_aliases_dict,API_term_list):
    api_mention_entries = api_mention.split('.')
    var_entry = api_mention_entries[0]
    

    var_type = get_variable_type(var_entry,variable_sources_dict,return_value_dict,loaded_aliases_dict,API_term_list)
    api_mention_entries[0] = var_type
    processed_api_mention = '.'.join(api_mention_entries)
    
    processed_api_mention_entries = processed_api_mention.split('.')
    var_alias = processed_api_mention_entries[0]
    
    
    if(var_alias in loaded_aliases_dict.keys() and var_alias not in API_term_list and len(var_alias) > 1):
        max_aliases_frequency = next(iter(loaded_aliases_dict[var_alias].items()))[1]
        if(max_aliases_frequency >= 10):
            globa_var_alias = next(iter(loaded_aliases_dict[var_alias].items()))[0]
            processed_api_mention_entries[0] = globa_var_alias 
            processed_api_mention = '.'.join(processed_api_mention_entries)
    
    
    
    # if(var_alias in loaded_aliases_dict.keys()):
    #     globa_var_alias = next(iter(loaded_aliases_dict[var_alias].items()))[0]
    #     processed_api_mention_entries[0] = globa_var_alias 
    #     processed_api_mention = '.'.join(processed_api_mention_entries)
    
    return processed_api_mention
def fetch_question_id_from_answer(answer_id, site="stackoverflow",api_key='rl_eTtryCzQUmfzfBLWmJV6QFnD3'):
    db = pymysql.connect(host='10.0.0.193',
                         #port=3306,
                         user='root',
                         password='0.mysql',
                         database='StackOverflow')
    cursor = db.cursor()
    sql_cmd = "SELECT ParentId FROM Posts where id={}".format(answer_id)
    cursor.execute(sql_cmd)

    data = cursor.fetchall()
    for element in data:
        if(not element[0] is None):
            return element[0]
        else:
            return None
    # os.environ["http_proxy"] = "http://192.168.16.1:7890"
    # os.environ["https_proxy"] = "http://192.168.16.1:7890"
    # url = f"https://api.stackexchange.com/2.3/answers/{answer_id}"
    # params = {
    #     'site': site,
    #     'filter': 'withbody',
    #     'key':api_key
    # }
    # response = requests.get(url, params=params)
    # data = response.json()
    # if data['items']:
    #     question_id = data['items'][0]['question_id']
    #     # 更新answer的question id
    #     sql_cmd = "UPDATE Posts SET ParentId = {} WHERE id ={}".format(question_id,answer_id)
    #     cursor.execute(sql_cmd)
    #     db.commit()
    #     db.close()
    #     return question_id
    # else:
    #     return None


def process_string_by_adding_special_markers(input_string, index_list,start_symbol="</BEGIN>",end_symbol="<END/>"):
    """
    处理给定的字符串，并在指定的索引区间范围内为子字符串添加标记。
    
    1. 首先，函数会去掉嵌套的索引区间（即一个区间的开始索引大于等于另一个区间的开始索引，
       且终止索引小于等于另一个区间的终止索引，则删除这个嵌套的区间）。
    2. 其次，过滤后的区间列表按照开始索引从大到小排序，以确保后续替换操作不会影响尚未处理的区间。
    3. 最后，遍历处理过的区间列表，并在每个区间对应的子字符串前后分别添加 `</BEGIN>` 和 `<END/>` 标签，
       每次替换都在前一次修改后的字符串基础上进行。

    参数：
        input_string (str): 待处理的原始字符串。
        index_list (list): 每个元素是一个二元组，包含开始索引和终止索引，表示要替换的子字符串的区间。
    
    返回：
        str: 经过标记处理后的字符串，指定区间的子字符串被 `</BEGIN>` 和 `<END/>` 标签包围。
    
    示例：
        input_string = "The quick brown fox jumps over the lazy dog."
        index_list = [(4, 9), (10, 19), (16, 19), (35, 39)]
        输出:
        'The </BEGIN>quick<END/> </BEGIN>brown fox<END/> jumps over </BEGIN>the lazy<END/> dog.'
    """
    # 去掉嵌套的索引区间
    filtered_list = []
    for i in range(len(index_list)):
        start_i, end_i = index_list[i]
        is_nested = False
        for j in range(len(index_list)):
            start_j, end_j = index_list[j]
            if i != j and start_j <= start_i and end_i <= end_j:
                is_nested = True
                break
        if not is_nested:
            filtered_list.append((start_i, end_i))
    
    # 按开始索引从大到小排序
    filtered_list.sort(key=lambda x: x[0], reverse=True)

    # 遍历list并替换子字符串
    for start, end in filtered_list:
        #input_string = input_string[:start] + "</BEGIN>" + input_string[start:end] + "<END/>" + input_string[end:]
        input_string = input_string[:start] + start_symbol + input_string[start:end] + end_symbol + input_string[end:]
    
    return input_string
def split_texts_and_codes_from_thread(post_id):
    db = pymysql.connect(host='10.0.0.193',
                         #port=3306,
                         user='root',
                         password='0.mysql',
                         database='StackOverflow')
    cursor = db.cursor()
    
    
    sql_cmd = "SELECT body FROM Posts where id={}".format(post_id)
    cursor.execute(sql_cmd)

    data = cursor.fetchall()
    for element in data:
        post_body = element[0]
        
        # 使用 html.unescape() 来解码转义的HTML字符
        decoded_text = html.unescape(post_body)
        
        
        decoded_text_str = decoded_text.encode("utf-8").strip()
        extracted_codes = []
        
        soup = BeautifulSoup(decoded_text_str, "lxml")
        
        body_text = []
        try:
            for para in soup.body:
                text_for_current_block = str(para)

                temp_soup = BeautifulSoup(text_for_current_block, "lxml")
                list_of_tags = [tag.name for tag in temp_soup.find_all()]


                if set(list_of_tags) == set(['html', 'body', 'pre', 'code']):
                    if(len(temp_soup.get_text().strip())>0):
                        body_text.append('Code: {}\n'.format(temp_soup.get_text()))
                else:
                    if(len(temp_soup.get_text().strip())>0):
                        body_text.append('Text: {}\n'.format(temp_soup.get_text()))
        except TypeError as e:

            #print(soup.body,"title is null!")
            print(post_id)
    db.close()            
    return body_text    
def fetch_thread_content_from_SO(post_id,post_content,label,mark_API_flag=False):
    marked_indices_list = []
    for cur_single_label in label:
        if(cur_single_label[2]=='Mention'):
            marked_indices_list.append((cur_single_label[0],cur_single_label[1]))
    #fetch post_content
    #if(fetch_question_id_from_answer(post_id) is None):
    if(1):
        question_id = post_id
        question_body = post_content
        if(mark_API_flag):
            question_body = process_string_by_adding_special_markers(question_body,marked_indices_list)
        # 使用正则表达式按 'Text:' 或 'Code:' 分割
        segments = re.split(r'\s*(Text:|Code:)', question_body)
        # 将标签和内容合并成完整元素
        question_body = [segments[i] + segments[i + 1] for i in range(1, len(segments) - 1, 2)]
        # 处理列表
        processed_question_body = [
            item[5:] if item.startswith('Text:') else f"<pre><code>\n{item[6:]}\n</code></pre>"
            for item in question_body
        ]
        
        
        processed_answer_body = []
        #post_content = '\n'.join(processed_question_body+processed_answer_body)
        
        post_content = '\n'.join(processed_question_body)
        
    else:
        question_id =  fetch_question_id_from_answer(post_id)      
        question_body = split_texts_and_codes_from_thread(question_id)
        answer_body = post_content
        if(mark_API_flag):
            answer_body = process_string_by_adding_special_markers(answer_body,marked_indices_list)
        # 使用正则表达式按 'Text:' 或 'Code:' 分割
        segments = re.split(r'\s*(Text:|Code:)', answer_body)
        # 将标签和内容合并成完整元素
        answer_body = [segments[i] + segments[i + 1] for i in range(1, len(segments) - 1, 2)]
        # 处理列表
        processed_answer_body = [
            item[5:] if item.startswith('Text:') else f"<pre><code>\n{item[6:]}\n</code></pre>"
            for item in answer_body
        ]
        processed_question_body = [
            item[5:] if item.startswith('Text:') else f"<pre><code>\n{item[6:]}\n</code></pre>"
            for item in question_body
        ]
        # post_content = '\n'.join(processed_question_body+processed_answer_body)
        post_content = '\n'.join(processed_answer_body)    
        #print(post_content)
    db = pymysql.connect(host='10.0.0.193',
                         #port=3306,
                         user='root',
                         password='0.mysql',
                         database='StackOverflow')
    cursor = db.cursor()
    
    
    sql_cmd = "SELECT Title, Tags FROM Posts where id={}".format(question_id)
    cursor.execute(sql_cmd)

    data = cursor.fetchall()
    for element in data:
        post_title = element[0]
        post_tags = element[1]
    
    return post_content,post_title,post_tags


def jaro_winkler_similarity(s1, s2, scaling=0.1):
    """
    高效计算 Jaro-Winkler 相似度
    :param s1: 第一个字符串
    :param s2: 第二个字符串
    :param scaling: 前缀缩放因子（默认 0.1）
    :return: Jaro-Winkler 相似度值（0 到 1）
    """
    len1, len2 = len(s1), len(s2)
    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0

    # 最大匹配距离
    match_distance = max(len1, len2) // 2 - 1
    s1_matches = np.zeros(len1, dtype=bool)
    s2_matches = np.zeros(len2, dtype=bool)
    matches, transpositions = 0, 0

    # 查找匹配字符
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(len2, i + match_distance + 1)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    # 计算转置数量
    s1_match_indices = np.where(s1_matches)[0]
    s2_match_indices = np.where(s2_matches)[0]
    for k1, k2 in zip(s1_match_indices, s2_match_indices):
        if s1[k1] != s2[k2]:
            transpositions += 1
    transpositions //= 2

    # Jaro 相似度公式
    jaro_sim = (
        matches / len1 +
        matches / len2 +
        (matches - transpositions) / matches
    ) / 3.0

    # 前缀奖励（最多 4 个字符）
    prefix_length = sum(1 for c1, c2 in zip(s1, s2) if c1 == c2)
    prefix_length = min(prefix_length, 4)

    # Jaro-Winkler 相似度公式
    return jaro_sim + scaling * prefix_length * (1 - jaro_sim)


def match_simplename_candidate(api_mention,API_full_name_list):
    candidate_list = []
    api_mention_entries = api_mention.split('.')
    for cur_standard_api in API_full_name_list:
        standard_api_entries = cur_standard_api.split('.')
        if(standard_api_entries[-1][:2].lower()==api_mention_entries[-1][:2].lower()):
            candidate_list.append(cur_standard_api)
    return candidate_list


def damerau_levenshtein_similarity(s1,s2):
    # 计算 Damerau-Levenshtein 距离
    distance = jellyfish.levenshtein_distance(s1,s2)
    #distance = jellyfish.damerau_levenshtein_distance(s1, s2)
    # 计算标准化相似度
    max_len = max(len(s1), len(s2))
    similarity = 1 - (distance / max_len)
    return similarity

def jaccard_similarity(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)