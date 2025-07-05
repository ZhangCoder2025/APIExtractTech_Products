import json
from CEDCC.utils import Extract_Codes_and_Data_From_XML,clean_jupyter_markup,output_identifiers_from_Codes
from CEDCC.utils import contains_demonstrative
from spacy.tokenizer import Tokenizer
import spacy
from CEDCC.LLM_prompt import generate_prompt_for_Building_Coreference_Chains_withGivenEntity
from CEDCC.utils import process_string_by_adding_special_markers
from CEDCC.utils import send_request_with_retry
from CEDCC.utils import extract_json_from_LLM,extract_complete_context_from_LLM
from CEDCC.LLM_prompt import generate_prompt_for_Complete_step_by_step_step1,generate_prompt_for_Complete_step_by_step_step2
def load_spacy_processor():
    
    nlp = spacy.load("en_core_web_sm")
    
    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=None)
    return nlp

def candidate_entity_selection(sample_sentence,context_body):
    key_words_list = ['he', 'she', 'it', 'they', 'him', 'her', 'them','this', 'that', 'these', 'those','my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours','what','such','above','below']
    

    
    all_code_Identifier_list = []
    all_code_Assigned_variables_list = []
    all_code_Function_parameters_list = []
    for cur_so_post in context_body:
        extracted_codes = Extract_Codes_and_Data_From_XML(cur_so_post)
        for cur_code_block in extracted_codes:
            cur_code_block = clean_jupyter_markup(cur_code_block)
            cur_code_identifiers,cur_code_assign_vars,cur_code_func_params = output_identifiers_from_Codes(source_code=cur_code_block)
            all_code_Identifier_list.extend(cur_code_identifiers)
            all_code_Assigned_variables_list.extend(cur_code_assign_vars)
            all_code_Function_parameters_list.extend(cur_code_func_params)
    related_code_Identifier_list = list(set(all_code_Identifier_list))
    related_code_Assigned_variables_list = list(set(all_code_Assigned_variables_list))
    related_code_function_parameters_list = list(set(all_code_Function_parameters_list))
    
    
    with open('CEDCC/API_elements.json', 'r') as file:
        loaded_API_elements_list = json.load(file)
    
    filtered_code_Assigned_variables_list = [item for item in related_code_Assigned_variables_list if item not in loaded_API_elements_list]
    
    candidate_entity_list = []
    nlp = load_spacy_processor()
    doc = nlp(sample_sentence)
    for noun_phrase in doc.noun_chunks:
        
        if(contains_demonstrative(noun_phrase,key_words_list)):
            candidate_entity_list.append([noun_phrase.text, (noun_phrase.start_char,noun_phrase.end_char)])
        core_word = noun_phrase.root
        core_word_lemma = core_word.text
        if('=' in core_word_lemma):
            core_word_lemma = core_word_lemma.split('=')[0].strip()
        if('[' in core_word_lemma):
            core_word_lemma = core_word_lemma.split('[')[0].strip()
        if(core_word_lemma in filtered_code_Assigned_variables_list):
            candidate_entity_list.append([noun_phrase.text, (noun_phrase.start_char,noun_phrase.end_char)])
    return candidate_entity_list

def predict_unclear_tokens_with_CorefChain_after_given_entity(sample_process_sentence,to_be_replace_for_predict_tokens):
    
    to_prompt_sentence = process_string_by_adding_special_markers(sample_process_sentence,to_be_replace_for_predict_tokens)
    msg = generate_prompt_for_Building_Coreference_Chains_withGivenEntity(to_prompt_sentence)
    response = send_request_with_retry(msg, api_key="",api_url="",model_type="",max_retries=5, retry_interval=60)
    annotation_result = response["choices"][0]["message"]["content"]
    return annotation_result
def CEDCC_step_1(sample_sentence,context_body):

    candidate_entity_list = candidate_entity_selection(sample_sentence,context_body)

    
    if(len(candidate_entity_list)>0):
        raw_CorefChain_result = predict_unclear_tokens_with_CorefChain_after_given_entity(sample_sentence,[element[1] for element in candidate_entity_list])
        CorefChain_result = extract_json_from_LLM(raw_CorefChain_result)
    
    
    candidate_entity_tokens = ["</BEGIN>{}<END/>".format(element[0]) for element in candidate_entity_list]
    unclear_tokens_indices_list = [] 
    for ref_entity in candidate_entity_list:
        ref_entity_token = "</BEGIN>{}<END/>".format(ref_entity[0])
        found_in_chain = False
        all_in_list = True
        for chain_index,chain_value in CorefChain_result.items():
            if ref_entity_token in chain_value:
                found_in_chain = True
                for element in chain_value:
                    if element not in candidate_entity_tokens:
                        all_in_list = False
                        break
                if all_in_list:
                    break
        if all_in_list:
            unclear_tokens_indices_list.append(ref_entity)

    return unclear_tokens_indices_list

def complete_context_step_by_step(to_prompt_sentence,context_body):
    
    
    msg_step_1 = generate_prompt_for_Complete_step_by_step_step1(to_prompt_sentence,context_body)
    response_msg_step_1 = send_request_with_retry(msg_step_1, api_key="",api_url="",model_type="",max_retries=5, retry_interval=60)
    raw_related_context = response_msg_step_1["choices"][0]["message"]["content"]
    
    msg_step_2 = generate_prompt_for_Complete_step_by_step_step2(to_prompt_sentence,raw_related_context)
    response_msg_step_2 = send_request_with_retry(msg_step_2, api_key="",api_url="",model_type="",max_retries=5, retry_interval=60)
    raw_completed_context = response_msg_step_2["choices"][0]["message"]["content"]
    
    return raw_completed_context

def CEDCC_step_2(sample_sentence,unclear_tokens_indices_list,context_body):
    to_prompt_sentence = process_string_by_adding_special_markers(sample_sentence,[element [1] for element in unclear_tokens_indices_list])
    raw_completed_context = complete_context_step_by_step(to_prompt_sentence,context_body)
    completed_context = extract_complete_context_from_LLM(raw_completed_context)
    # if(not isinstance(completed_context,list)):
    #     print("parse error")
    # else:
    #     for cur_context_pairs in completed_context:
    #         print("'{}' in the sentence '{}' is context-dependent entity, it refers to {}".format(cur_context_pairs['NP'],sample_sentence,cur_context_pairs['Value']))
    return completed_context
if __name__ == '__main__':
    sample_sentence = 'To fit a parabola to those points, use numpy.polyfit()'
    context_body = '''
    <p>Assume you have some data points</p><pre><code>x = numpy.array([0.0, 1.0, 2.0, 3.0])
y = numpy.array([3.6, 1.3, 0.2, 0.9])</code></pre>
<p>To fit a parabola to those points, use polyfit()</p>
<pre><code>darr = numpy.array([1, 3.14159, 1e100, -2.71828])</code></pre>
<p>darr.argmin() will give you the index corresponding to the minimum.</p>
    '''
    unclear_tokens_indices_list = CEDCC_step_1(sample_sentence,context_body)
    completed_context = CEDCC_step_2(sample_sentence,unclear_tokens_indices_list,context_body)
    