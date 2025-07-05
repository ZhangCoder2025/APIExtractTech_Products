from Preprocess.twokenize import tokenize
import re
import spacy
def text_process_func(html_text):
    # 匹配 <p>...</p> 和 <pre><code>...</code></pre> 并记录标签类型和内容
    pattern = re.compile(r'(<p>(.*?)</p>)|(<pre><code>(.*?)</code></pre>)', flags=re.DOTALL)

    result = []
    for match in pattern.finditer(html_text):
        if match.group(2):  # <p> 匹配内容
            result.append(f"Text: {match.group(2).strip()}")
        elif match.group(4):  # <code> 匹配内容
            result.append(f"Code: {match.group(4).strip()}")
    
    tagged_post_content = []
    nlp = spacy.load("en_core_web_sm")
    tokens_list = {}
    item_index = 0
    for item in result:
        if(item.startswith('Text:')):
            text_fragment = item[6:]
            doc= nlp(text_fragment)
            sentences = [sent.text for sent in doc.sents]
            for cur_sent in sentences:
                tokens = tokenize(cur_sent)
                tokens_list[item_index] = tokens
                item_index += 1
                #tokens_list.append(tokens)
                tagged_post_content.append('Text: {}'.format(cur_sent))
        else:
            tagged_post_content.append(item)
            item_index += 1
    
    
    return tokens_list,tagged_post_content

if __name__ == '__main__':
    text_process_func("""<p>Assume you have some data points</p><pre><code>x = numpy.array([0.0, 1.0, 2.0, 3.0])
y = numpy.array([3.6, 1.3, 0.2, 0.9])</code></pre>
<p>To fit a parabola to those points, use polyfit()</p>
<pre><code>darr = numpy.array([1, 3.14159, 1e100, -2.71828])</code></pre>
<p>darr.argmin() will give you the index corresponding to the minimum.</p>""")