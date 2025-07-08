import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from final_utils import clean_jupyter_markup,remove_parentheses_content




class Thread:
    def __init__(self, input_text,title="",post_id=""):
        self.input_text = input_text
        self.code_block = []
        self.text_block = []
        self.title = title
        self.post_id = post_id
        self._extract_content()
        self._analyse_import_module()
        self._analyze_var_assignment()
    def _extract_content(self):
        self.text_block.append(self.title)
        start = 0
        while start < len(self.input_text):
            # 找到下一个 <pre><code> 标签
            code_start = self.input_text.find("<pre><code>", start)
            # 找到下一个 </code></pre> 标签
            code_end = self.input_text.find("</code></pre>", start)
            
            if code_start != -1 and (code_end != -1 and code_start < code_end):
                # 先添加 <pre><code> 之前的文本部分
                if start != code_start:
                    self.text_block.append(self.input_text[start:code_start].strip())
                
                # 提取代码块内容
                code_content = self.input_text[code_start + len("<pre><code>"):code_end].strip()
                self.code_block.append(code_content)
                
                # 更新起始位置，跳过 </code></pre> 标签
                start = code_end + len("</code></pre>")
            else:
                # 没有更多的代码块，添加剩下的文本
                self.text_block.append(self.input_text[start:].strip())
                break
        #self.text_block.append(self.title)
    def _analyse_import_module(self):

        
        imports = []
        PY_LANGUAGE = Language(tspython.language())

        
        # 创建Parser对象
        parser = Parser(PY_LANGUAGE)
        for cur_code in self.code_block:
            wildcard_imports = []
            imported_funcs = []
            
            
            cur_code = clean_jupyter_markup(cur_code)
            tree = parser.parse(bytes(cur_code, "utf8"))
            root_node = tree.root_node
            
            # 直接遍历 AST 找到所有 import 语句
            for node in root_node.children:
                if node.type == 'import_statement':
                    # 处理 import 语句
                    for child in node.children:
                        if child.type == 'dotted_name':
                            # 直接模块名称
                            imports.append([cur_code[child.start_byte:child.end_byte]])
                            
                            imported_funcs.append([cur_code[child.start_byte:child.end_byte]])
                            
                        elif child.type == 'aliased_import':
                            # 带有别名的模块
                            dotted_name = child.child_by_field_name('name')
                            alias_name = child.child_by_field_name('alias')
                            if dotted_name and alias_name:
                                full_import = [cur_code[dotted_name.start_byte:dotted_name.end_byte],cur_code[alias_name.start_byte:alias_name.end_byte]]
                                imports.append(full_import)
                                
                                imported_funcs.append(cur_code[alias_name.start_byte:alias_name.end_byte])
                                
                    #imports.append({"type": "import", "modules": import_names})

                elif node.type == 'import_from_statement':
                    # 处理 from ... import ... 语句
                    module_name = None

                    # 手动查找第一个 identifier 作为模块名称
                    for child in node.children:
                        child_type = child.type
                        child_text = (cur_code[child.start_byte:child.end_byte].decode("utf-8")
                      if isinstance(cur_code, bytes)
                      else cur_code[child.start_byte:child.end_byte])
                        
                        # 第一个 `dotted_name` 是模块名
                        if child_type == "dotted_name" and module_name is None:
                            module_name = child_text
                        elif child_type == "dotted_name":
                            # 导入项（不带别名）
                            full_name = f"{module_name}.{child_text}"
                            #imports.append([full_name])
                            imports.append([full_name, child_text] if child_text else [full_name])
                            
                            imported_funcs.append(child_text)
                        elif child_type == "aliased_import":
                            # 带别名的导入项
                            name_node = child.child_by_field_name("name")
                            alias_node = child.child_by_field_name("alias")
                            name = (cur_code[name_node.start_byte:name_node.end_byte].decode("utf-8")
                                    if isinstance(cur_code, bytes)
                                    else cur_code[name_node.start_byte:name_node.end_byte])
                            alias = (cur_code[alias_node.start_byte:alias_node.end_byte].decode("utf-8")
                                    if alias_node and isinstance(cur_code, bytes)
                                    else cur_code[alias_node.start_byte:alias_node.end_byte] if alias_node else None)
                            full_name = f"{module_name}.{name}"
                            imports.append([full_name, alias] if alias else [full_name])
                            
                            imported_funcs.append(alias)
                        elif child.type == "wildcard_import":
                            wildcard_imports.append(module_name)
            #处理通配符导入
            if(len(wildcard_imports)>0):
                wildcard_imports = list(set(wildcard_imports))
                calls = []
                custom_functions = []
                # 遍历所有方法调用
                stack = [root_node]  # 初始化栈
                while stack:
                    node = stack.pop()
                    # 处理当前节点
                    #print(f"Node type: {node.type}, Text: {cur_code[node.start_byte:node.end_byte]}")
                    
                    # 将子节点加入栈中（从右到左，这样弹出时是从左到右）
                    stack.extend(reversed(node.children))
                    if node.type == "call":
                        function_name = node.child_by_field_name("function")
                        if function_name and function_name.type == "identifier":
                            calls.append(function_name.text.decode("utf-8"))
                    if(node.type == 'function_definition'):
                        # 提取函数名称
                        name_node = node.child_by_field_name("name")
                        if name_node:
                            custom_functions.append(name_node.text.decode("utf-8"))
                calls = list(set(calls))
                custom_functions = list(set(custom_functions))
                for cur_wildward_module in wildcard_imports:
                    for cur_func in calls:
                        if(cur_func not in custom_functions and cur_func not in imported_funcs):
                            full_name = f"{cur_wildward_module}.{cur_func}"
                            #imports.append([full_name])
                            imports.append([full_name, cur_func])
        dep_tracing_dict = {}
        for entry in imports:
            if len(entry) == 2:
                full_module, alias = entry
                if alias not in dep_tracing_dict:
                    dep_tracing_dict[alias] = []
                dep_tracing_dict[alias].append(full_module)
            elif len(entry) == 1:
                full_module = entry[0]
                if full_module not in dep_tracing_dict:
                    dep_tracing_dict[full_module] = []
                dep_tracing_dict[full_module].append(full_module)
        for key,value in dep_tracing_dict.items():
            dep_tracing_dict[key] = list(set(value))
        
        self.import_types = dep_tracing_dict
    
    def _analyze_var_assignment(self):
        var_type_dict = {}
        
        PY_LANGUAGE = Language(tspython.language())

        # 创建Parser对象
        parser = Parser(PY_LANGUAGE)
        # if(self.post_id=='6004738'):
        #     print("wait")
        for cur_code in self.code_block:
            cur_code = clean_jupyter_markup(cur_code)
            tree = parser.parse(bytes(cur_code, "utf8"))
            root_node = tree.root_node
            # 遍历所有方法调用
            stack = [root_node]  # 初始化栈
            while stack:
                node = stack.pop()
                stack.extend(reversed(node.children))

            
        
            
            #for node in root_node.descendants:
                if node.type == "assignment":
                    # 获取赋值左侧的变量
                    left_node = node.child_by_field_name('left')
                    right_node = node.child_by_field_name('right')

                    if left_node and right_node:
                        # 获取变量名称
                        # 提取左侧变量名
                        variable_name = cur_code[left_node.start_byte:left_node.end_byte].decode("utf-8") if isinstance(cur_code, bytes) else cur_code[left_node.start_byte:left_node.end_byte]
                
                        if right_node.type == 'call':  # 右侧是一个函数调用
                            function_node = right_node.child_by_field_name('function')
                            if function_node:
                                if function_node.type == "attribute":
                                    # 提取方法的前缀和名称
                                    object_node = function_node.child_by_field_name('object')
                                    method_node = function_node.child_by_field_name('attribute')
                                    if object_node and method_node:
                                        prefix = cur_code[object_node.start_byte:object_node.end_byte].decode("utf-8") if isinstance(cur_code, bytes) else cur_code[object_node.start_byte:object_node.end_byte]
                                        method = cur_code[method_node.start_byte:method_node.end_byte].decode("utf-8") if isinstance(cur_code, bytes) else cur_code[method_node.start_byte:method_node.end_byte]
                                        process_prefix = remove_parentheses_content(prefix)
                                        process_method = remove_parentheses_content(method)
                                        if(variable_name not in var_type_dict.keys()):
                                            var_type_dict[variable_name] = ["{}.{}".format(process_prefix,process_method)]
                                        else:
                                            if("{}.{}".format(process_prefix,process_method) not in var_type_dict[variable_name]):
                                                var_type_dict[variable_name].append("{}.{}".format(process_prefix,process_method))
                                elif function_node.type == "identifier":
                                    method = cur_code[function_node.start_byte:function_node.end_byte].decode("utf-8") if isinstance(cur_code, bytes) else cur_code[function_node.start_byte:function_node.end_byte]    
                                    process_method = remove_parentheses_content(method)
                                    if(variable_name not in var_type_dict.keys()):
                                        var_type_dict[variable_name] = [process_method]
                                    else:
                                        if(process_method not in var_type_dict[variable_name]):
                                            var_type_dict[variable_name].append(process_method)
        for key,value in var_type_dict.items():
            #var_type_dict[key] = list(set(value))
            #去重且保持顺序
            var_type_dict[key] = list(dict.fromkeys(value))
        
        
        assign_body = {}
        for key_list,value_list in var_type_dict.items():
            key_list = [key.strip() for key in key_list.split(',') if key.strip()]
            for cur_value in value_list:
                for var_index,cur_var in enumerate(key_list):
                    if(cur_var in assign_body.keys()):
                        assign_body[cur_var].append([cur_value,var_index])
                    else:
                        assign_body[cur_var] = [[cur_value,var_index]]
        
        # # 去重方法
        
        # for key,value in assign_body.items():  
        #     unique_data = []
        #     for item in value:
        #         if item not in unique_data:
        #             unique_data.append(item)
        #     assign_body[key] = unique_data
        
        
        self.assign_body = assign_body                        
    
                        

    
if __name__=='__main__':
    # 测试用例
    input_string = """ I am just getting started with pandas in the IPython Notebook and encountering the following problem: ...
    <pre><code>
    In [27]:

    evaluation = readCSV("evaluation_MO_without_VNS_quality.csv").filter(["solver", "instance", "runtime", "objective"])
    </code></pre>
    I would like to see a small portion of the data frame as a table just to make sure it is in the right format.
    <pre><code>
    >>> df = pd.DataFrame({"A": range(1000), "B": range(1000)})
    >>> df[:5]
    </code></pre>
    <pre><code>
    import pandas as pd
    import numpy as np
    from collections import defaultdict, namedtuple
    from mypackage import mymodule
    from module.submodule import function as func, ClassName
    df = pd.DataFrame({'A':[9,10]*6,
                    'B':range(23,35),
                    'C':range(-6,6)})

    print(df)
    #      A   B  C
    # 0    9  23 -6
    # 1   10  24 -5
    # 2    9  25 -4
    # 3   10  26 -3
    # 4    9  27 -2
    # 5   10  28 -1
    # 6    9  29  0
    # 7   10  30  1
    # 8    9  31  2
    # 9   10  32  3
    # 10   9  33  4
    # 11  10  34  5
    </code></pre>
    <pre><code>
    a("param1").b("param2").c("param3")
    DataFrame()
    </code></pre>
    <pre><code>
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate some data
    yellow_data, green_data = np.random.random((2,2000))
    yellow_data += np.linspace(0, 3, yellow_data.size)
    green_data -= np.linspace(0, 3, yellow_data.size)

    # Plot the data
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax2 = ax1.twinx()

    ax1.plot(yellow_data, 'y-')
    ax2.plot(green_data, 'g-')

    # Change the axis colors...
    for ax, color in zip([ax1, ax2], ['yellow', 'green']):
        for label in ax.yaxis.get_ticklabels():
            label.set_color(color)

    plt.show()
    </code></pre>
    <pre><code>
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.multiclass import OneVsRestClassifier
    from nltk.classify import SklearnClassifier
    from sklearn.feature_extraction import DictVectorizer
    </code></pre>
    """
    
    
    
    
    extractor = Thread(input_string)
    print("Code Blocks:", extractor.code_block)
    print("Text:", extractor.text_block)
    
