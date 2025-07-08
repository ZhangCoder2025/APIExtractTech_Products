import time
import openai
import json
import random

from datetime import datetime
import pytz

# 设置时区
timezone = pytz.timezone('Asia/Shanghai')

# 获取当前时间
now = datetime.now(timezone)

# 格式化输出为指定的字符串格式
formatted_time = now.strftime('%a %b %d %Y %H:%M:%S GMT%z') + " (中国标准时间)"


def gpt_api_recognize(message,model_type="gpt-4"):
    openai.api_base = ""
    openai.api_key = ""
    response = openai.ChatCompletion.create(
    model = "{}".format(model_type),
    messages=message,
    #temperature=1.0,
    temperature=0.5,
    #temperature=0.2,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    )
    if(isinstance(response,str)):
        response = json.loads(response)
    return response

def send_request_with_retry(msg, max_retries=30, retry_interval=60,model_type="gpt-4",Link_type="Direct"):
    """
    发送请求并在遇到 RateLimitError 时自动重试。

    :param url: 请求的 URL
    :param headers: 请求头
    :param max_retries: 最大重试次数
    :param retry_interval: 每次重试之间等待的时间（秒）
    :return: Response 对象，如果重试次数用尽仍失败，则为 None
    """
    retries = 0
    while retries < max_retries:
        try:
            response = gpt_api_recognize(message=msg,model_type=model_type,Link_type=Link_type)
            return response
        except openai.error.RateLimitError:
            print(f"Rate limit exceeded. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
            retries += 1
        except openai.error.APIConnectionError:
            # 处理与 API 连接有关的错误
            print(f"API connection error. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
            retries += 1
        except openai.error.OpenAIError as e:
            # 处理其他可能的 OpenAI 错误
            if '502 Bad Gateway' in str(e):
                print(f"Bad Gateway. Retrying in {retry_interval/2} seconds...")
                time.sleep(retry_interval/2)
                retries += 1
            else:
                print(f"An error occurred: {e}")
                break
    print("Max retries exceeded.")
    return None


def generate_API_link_oneStep_prompt(post_body):
    msg = [
        {
        "role":"system",
        "content": 
            """
            ### Task:

            I have a Stack Overflow post that contains both text and code segments. Please analyze the post and extract only the API mentions from the text (ignore mentions in code). For each API mentioned in the text, provide output the following information:
            
            1. API Name
            2. Full API Name (e.g., including the module or library it belongs to)
            3. The sentence or context in which the API is mentioned
            
            Please provide the output in **JSON format**. Each API mention should be represented as a JSON object with the following fields:
            - `api_name`: The name of the API.
            - `full_api_name`: The full name of the API (including the library or module).
            - `context`: The sentence or context where the API is mentioned.
            
            ### Example
            **Input:**
            Text: In this case, where the DataFrame is long but not too wide, you can simply slice it:
            Code: >>> df = pd.DataFrame({"A": range(1000), "B": range(1000)})
            Text: ix is deprecated. If it's both wide and long, I tend to use .ix:
            Code: >>> df = pd.DataFrame({i: range(1000) for i in range(100)})
                  >>> df.ix[:5, :10]

            **Output in JSON format:**
            [
            {
                "api_name": "DataFrame",
                "full_api_name": "pandas.DataFrame",
                "context": "In this case, where the DataFrame is long but not too wide, you can simply slice it:"
            },
            {
                "api_name": "ix",
                "full_api_name": "pandas.DataFrame.ix",
                "context": "If it's both wide and long, I tend to use .ix:"
            }
            ]
            
            **Now, please analyze the following post:**
            
            Please provide the extracted information in the JSON format as shown in the example.
            
            
            """
            },
           {
               "role":"user",
               "content":"Now, please analyze the following post: {}, and please provide the extracted information in the JSON format as shown in the example.".format(post_body)
           }
           ]
    return msg


def generate_API_link_provideExactTokens_prompt(post):
    msg = [
        {
            "role":"system",
            "content":"\nYou are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent model: gpt-4\nCurrent time: {}\nLatex inline: \\(x^2\\) \nLatex block: $$e=mc^2$$\n\n".format(formatted_time)
        },
        {
        "role":"user",
        "content": 
            f"""
            ### Task:
            You are provided with a Stack Overflow post where each API mentions are marked with special tags ("</BEGIN>" and "<END/>"), Your task is to provide the full name of each API mentions, including the module or library it belongs to, by analyzing its context and the given Stack Overflow post. 
            
            ### Output Requirements:
            
            (1) Output the full name of each API mentions in a list in order of appearance in posts.
            (2) The number of elements in the output list must be equal to the number of the API mentions with special tags.
            (3) Only the judgments are output, without any explanatory or introductory content!!!
            ### Example
            **Input:**
            Post: "In this case, where the </BEGIN>DataFrame<END/> is long but not too wide, you can simply slice it. If it's both wide and long, I tend to use </BEGIN>.ix<END/>:"
            
            
            **Output:**
            [pandas.DataFrame,pandas.DataFrame.ix]

            Now, please analye the following post and output the result:\n
            POST:{post}
            """
        }
           ]
    return msg



def generate_API_link_provideExactTokens_prompt_withCandidate(post,candidate_list):
    msg = [
        {
            "role":"system",
            "content":"\nYou are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent model: gpt-4\nCurrent time: {}\nLatex inline: \\(x^2\\) \nLatex block: $$e=mc^2$$\n\n".format(formatted_time)
        },
        {
        "role":"user",
        "content": 
            f"""
            ### Task:            
            You are provided with a Stack Overflow post where each API mention is marked with special tags ("</BEGIN>" and "<END/>"). Additionally, for each API mention, you are given a list of candidate API names with their full names. Your task is to select the correct full name of each API mention from the given candidate list by analyzing its context within the Stack Overflow post. If none of the candidates match, return "None" for that mention.
            
            ### Input Format:
            (1) The post text with API mentions marked using "</BEGIN>" and "<END/>" tags.
            (2) A candidate list for each API mention, provided in the order of appearance.

            ### Output Requirements:
            (1) Output the full name of each API mention in a list in order of appearance in the post.
            (2) The number of elements in the output list must be equal to the number of API mentions with special tags.
            (3) Only output the selected API names from the candidates list or "None" if none match, without any explanatory or introductory content!!!

            ### Example

            **Input:**

            Post: "In this case, where the </BEGIN>DataFrame<END/> is long but not too wide, you can simply slice it. If it's both wide and long, I tend to use </BEGIN>.ix<END/>."

            Candidates:
            1. ['pandas.DataFrame', 'pyspark.sql.DataFrame', 'dask.DataFrame']
            2. ['pandas.DataFrame.ix', 'pandas.DataFrame.loc', 'numpy.ix_', 'dask.ix']

            **Output:**
            [pandas.DataFrame, pandas.DataFrame.ix]
            
            
            Now, please analye the following post and output the result:\n
            POST:{post}\n
            Candidates:
            {candidate_list}\n
            """
        }
           ]
    return msg


def generate_API_link_provideExactTokens_prompt_withCandidate_withoutNone(post,candidate_list):
    msg = [
        {
            "role":"system",
            "content":"\nYou are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent model: gpt-4\nCurrent time: {}\nLatex inline: \\(x^2\\) \nLatex block: $$e=mc^2$$\n\n".format(formatted_time)
        },
        {
        "role":"user",
        "content": 
            f"""
            ### Task:            
            You are provided with a Stack Overflow post where each API mention is marked with special tags ("</BEGIN>" and "<END/>"). Additionally, for each API mention, you are given a list of candidate API names with their full names. Your task is to select the correct full name of each API mention from the given candidate list by analyzing its context within the Stack Overflow post. 
            
            ### Input Format:
            (1) The post text with API mentions marked using "</BEGIN>" and "<END/>" tags.
            (2) A candidate list for each API mention, provided in the order of appearance.

            ### Output Requirements:
            (1) Output the full name of each API mention in a list in order of appearance in the post.
            (2) The number of elements in the output list must be equal to the number of API mentions with special tags.
            (3) Only output the selected API names from the candidates list, without any explanatory or introductory content!!!

            ### Example

            **Input:**

            Post: "In this case, where the </BEGIN>DataFrame<END/> is long but not too wide, you can simply slice it. If it's both wide and long, I tend to use </BEGIN>.ix<END/>."

            Candidates:
            1. ['pandas.DataFrame', 'pyspark.sql.DataFrame', 'dask.DataFrame']
            2. ['pandas.DataFrame.ix', 'pandas.DataFrame.loc', 'numpy.ix_', 'dask.ix']

            **Output:**
            [pandas.DataFrame, pandas.DataFrame.ix]
            
            
            Now, please analye the following post and output the result:\n
            POST:{post}\n
            Candidates:
            {candidate_list}\n
            """
        }
           ]
    return msg



def generate_API_link_withRAG(post,candidate_list,Processed_API_mentions):
    msg = [
        {
            "role":"system",
            "content":"\nYou are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent model: gpt-4\nCurrent time: {}\nLatex inline: \\(x^2\\) \nLatex block: $$e=mc^2$$\n\n".format(formatted_time)
        },
        {
        "role":"user",
        "content": 
            f"""
            ### Task:
            You are provided with a Stack Overflow post where some API mentions are marked with special tags ("</BEGIN>" and "<END/>"), Additionally, for each API mention, you are given a list of candidate API names with their full names, and the processed API mention obtained by analyzing the code snippets in the Stack Overflow post and restoring the possible abbreviations and aliases of these mentions. These processed mentions can be used for reference, but they are not necessarily absolutely correct. 
            
            Your task is to refer the full name of these API mentions, including the module or library it belongs to, by analyzing its context within the Stack Overflow post, the given candidate list and the processed API mentions. 
            
            ### Input Format:
            (1) The post text with API mentions marked using "</BEGIN>" and "<END/>" tags.
            (2) A candidate list for each API mention, provided in the order of appearance.
            (3) Processed API mention with code snippet for each API mention, provided in the order of appearance.
            
            ### Output Requirements:
            (1) Output the full name of each API mention in a list in order of appearance in the post!!!
            (2) The number of elements in the output list must be equal to the number of API mentions with special tags!!!
            (3) Only output the selected API names from the candidates list or "None" if none match, without any explanatory or introductory content!!!

            ### Example

            **Input:**

            Post: "In this case, where the </BEGIN>DataFrame<END/> is long but not too wide, you can simply slice it. If it's both wide and long, I tend to use </BEGIN>.ix<END/>."

            Candidates:
            1. ['pandas.DataFrame', 'pyspark.sql.DataFrame', 'dask.DataFrame']
            2. ['pandas.DataFrame.ix', 'pandas.DataFrame.loc', 'numpy.ix_', 'dask.ix']

            Processed API mentions:
            1. 'pandas.DataFrame'
            2. 'pandas.DataFrame.ix'
            
            **Output:**
            [pandas.DataFrame, pandas.DataFrame.ix]
            
            
            Now, please analye the following post and output the result:\n
            POST:{post}\n
            Candidates:
            {candidate_list}\n
            Processed API mentions:
            {Processed_API_mentions}\n
            """
        }
           ]
    return msg



def generate_API_link_withText_andCode(post,candidate_list,Text_Predict_apis,Code_Predict_apis):
    msg = [
        {
            "role":"system",
            "content":"\nYou are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent model: gpt-4\nCurrent time: {}\nLatex inline: \\(x^2\\) \nLatex block: $$e=mc^2$$\n\n".format(formatted_time)
        },
        {
        "role":"user",
        "content": 
            f"""
            ### Task
            You are given a Stack Overflow post where certain API mentions are enclosed by special tags `</BEGIN>` and `<END/>`. For each mentioned API, you also have a corresponding list of candidate full API names, provided in the order the mentions appear.
            
            Your task is to determine the full names of these mentioned APIs (including their modules or libraries) by analyzing the context of the Stack Overflow post and the given candidate lists. If none of the candidates match, return "None".
            
            ### Input Format
            1. The post text, where API mentions are marked with `</BEGIN>` and `<END/>` tags.
            2. A candidate list for each API mention, given in the same order as the mentions appear in the post.
            
            ### Output Requirements
            1. Output the chosen full API names in a list, in the same order as their appearance in the post!!!
            2. The number of output elements must match the number of marked API mentions!!!
            3. For each API mention, only output one selected full name from the candidate list, or "None" if no candidates match!!!
            4. Do not provide any explanations or additional content, only output the selected names or "None"!!!!!
            
            ### Example

            **Input:**
            Post:  
            "In this case, where the </BEGIN>DataFrame<END/> is long but not too wide, you can simply slice it. If it's both wide and long, I tend to use </BEGIN>.ix<END/>."

            Candidates:  
            1. `['pandas.DataFrame', 'pyspark.sql.DataFrame', 'dask.DataFrame']`  
            2. `['pandas.DataFrame.ix', 'pandas.DataFrame.loc', 'numpy.ix_', 'dask.ix']`

            Auxiliary information (for reference only, not guaranteed correct):
            - Text analysis results:  
            - For "DataFrame": Possible predictions include ["pandas.DataFrame", "pyspark.sql.DataFrame"].
            - For ".ix": Possible predictions include ["pandas.DataFrame.ix", "numpy.ix_"].
            
            - Code analysis results:  
            - The API full name of "DataFrame" might end with "pandas.DataFrame".
            - The API full name of ".ix" might end with "pandas.DataFrame.ix".
            
            
            **Output:**
            [pandas.DataFrame, pandas.DataFrame.ix]
            
            
            Now, please analye the following post and output the result:\n
            POST:
            {post}\n
            Candidates:
            {candidate_list}\n
            Auxiliary information (for reference only, not guaranteed correct):\n
            {Text_Predict_apis}\n\n
            {Code_Predict_apis}\n\n
            
            
            """
        }
           ]
    return msg

def generate_API_link_for_RALLA(post,candidate_list):
    msg = [
        {
            "role":"system",
            "content":"\nYou are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent model: gpt-4\nCurrent time: {}\nLatex inline: \\(x^2\\) \nLatex block: $$e=mc^2$$\n\n".format(formatted_time)
        },
        {
        "role":"user",
        "content": 
            f"""
            ### Task
            You are given a Stack Overflow post where certain API mentions are enclosed by special tags `</BEGIN>` and `<END/>`. For each mentioned API, you also have a corresponding list of candidate API fully qualified names, provided in the order the mentions appear.
            
            Your task is to determine the fully qualified names of these mentioned APIs (including their modules or libraries) by analyzing the context of the Stack Overflow post and the given candidate lists. If none of the candidates match, return "None".
            
            ### Additional Information
            1. The candidate lists have been pre-processed based on an analysis of the API mention's surrounding context.
            2. The candidates in each list are sorted in descending order of probability, meaning the most likely API fully qualified names appears first.
            
            ### Input Format
            1. The post text, where API mentions are marked with `</BEGIN>` and `<END/>` tags.
            2. A candidate list for each API mention, given in the same order as the mentions appear in the post.
            
            ### Output Requirements
            1. Output the chosen API fully qualified names in a list, in the same order as their appearance in the post!!!
            2. The number of output elements must match the number of marked API mentions!!!
            3. For each API mention, only output one selected fully qualified name from the candidate list, or "None" if no candidates match!!!
            4. Do not provide any explanations or additional content, only output the selected names or "None"!!!!!
            
            ### Example

            **Input:**
            Post:  
            "In this case, where the </BEGIN>DataFrame<END/> is long but not too wide, you can simply slice it. If it's both wide and long, I tend to use </BEGIN>.ix<END/>."

            Candidates:  
            1. `['pandas.DataFrame', 'pyspark.sql.DataFrame', 'dask.DataFrame']`  
            2. `['pandas.DataFrame.ix', 'pandas.DataFrame.loc', 'numpy.ix_', 'dask.ix']`

            **Output:**
            [pandas.DataFrame, pandas.DataFrame.ix]
            
            
            Now, please analye the following post and output the result:\n
            POST:
            {post}\n
            Candidates:
            {candidate_list}\n
            """
        }
           ]
    return msg

def generate_API_link_onlywithRule(post,Processed_API_mentions):
    msg = [
        {
            "role":"system",
            "content":"\nYou are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent model: gpt-4\nCurrent time: {}\nLatex inline: \\(x^2\\) \nLatex block: $$e=mc^2$$\n\n".format(formatted_time)
        },
        {
        "role":"user",
        "content": 
            f"""
            ### Task:
            You are provided with a Stack Overflow post where some API mentions are marked with special tags ("</BEGIN>" and "<END/>"), Additionally, for each API mention, you are given the processed API mention obtained by analyzing the code snippets in the Stack Overflow post and restoring the possible abbreviations and aliases of these mentions. These processed mentions can be used for reference, but they are not necessarily absolutely correct. 
            
            Your task is to refer the full name of these API mentions, including the module or library it belongs to, by analyzing its context within the Stack Overflow post and the processed API mentions. 
            
            ### Input Format:
            (1) The post text with API mentions marked using "</BEGIN>" and "<END/>" tags.
            (2) Processed API mention with code snippet for each API mention, provided in the order of appearance.
            
            ### Output Requirements:
            (1) Output the full name of each API mention in a list in order of appearance in the post, without any explanatory or introductory content!!!
            (2) The number of elements in the output list must be equal to the number of API mentions with special tags!!!

            ### Example

            **Input:**

            Post: "In this case, where the </BEGIN>DataFrame<END/> is long but not too wide, you can simply slice it. If it's both wide and long, I tend to use </BEGIN>.ix<END/>."

            Processed API mentions:
            1. 'pandas.DataFrame'
            2. 'pandas.DataFrame.ix'
            
            **Output:**
            [pandas.DataFrame, pandas.DataFrame.ix]
            
            
            Now, please analye the following post and output the result:\n
            POST:{post}\n
            Processed API mentions:
            {Processed_API_mentions}\n
            """
        }
           ]
    return msg


