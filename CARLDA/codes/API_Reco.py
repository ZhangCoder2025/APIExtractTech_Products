'''
import packages
'''
import torch
from tqdm.auto import tqdm
import os
from datasets import load_dataset
from Models import CAREER
from transformers import AutoTokenizer,AutoConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW,SGD
from transformers import get_linear_schedule_with_warmup
from utils import read_parameters,adjust_Hierarchical_lr,SpeciDataCollator,train, validate, test,preprocess_Datasets
import copy
import pandas as pd
import ast
from numpy import mean
from utils import Alphabet
import pickle
'''
define functions
'''

'''
Data Process

Adding the special tokens [CLS] and [SEP] and subword tokenization creates a mismatch between the input and labels.
1. Mapping all tokens to their corresponding word with the word_ids method.
2. Assigning the label -100 to the special tokens [CLS] and [SEP] so the PyTorch loss function ignores them.
3. Only labeling the first token of a given word. Assign -100 to other subtokens from the same word.
#set is_split_into_words=True to tokenize the words into subwords:
#tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
'''
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True,add_special_tokens=False)

    token_word_ids = []
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        token_word_ids.append(word_ids)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    tokenized_inputs["word_ids"] = token_word_ids
    chars = []
    for i, word in enumerate(examples["tokens"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        char_ids = []
        for word_idx in word_ids:
            char_ids.append([char2index.get_index(char)  for char in word[word_idx]])

        chars.append(char_ids)

    tokenized_inputs["chars"] = chars

    return tokenized_inputs








if __name__=='__main__':
    #print(torch.cuda.is_available())
    args = read_parameters()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_device
    label_list = ['0','B-API']

    # free memeory
    torch.cuda.empty_cache()

    # load and moditify config
    model_config = AutoConfig.from_pretrained(args.model_location,num_labels = len(label_list))

    # build char alphabet
    global char2index


    char2index = Alphabet('character')
    with open("char_alphabet.pkl", 'rb') as char_file:
        char2index = pickle.loads(char_file.read())

    #char2index = build_alphabet(input_file=args.char_alphabet_files_location)
    args.char_alphabet_size = char2index.size()

    #load model
    model = CAREER.from_pretrained(args.model_location,config=model_config,args=args)


    # Load the tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_location,model_max_length=512)


    # prepare a dataset
    raw_train_datasets = load_dataset('json', data_files=args.train_files_location)['train']
    raw_valid_datasets = load_dataset('json', data_files=args.valid_files_location)['train']
    raw_test_datasets = load_dataset('json', data_files=args.test_files_location)['train']

    test_fullnames_list = raw_test_datasets['full_names']
    test_mention_morphological_list = raw_test_datasets['mention_morphological']
    test_mention_common_list = raw_test_datasets['mention_common']
    test_api_common_list = raw_test_datasets['api_common']

    raw_train_datasets = preprocess_Datasets(raw_train_datasets)
    raw_valid_datasets = preprocess_Datasets(raw_valid_datasets)
    raw_test_datasets = preprocess_Datasets(raw_test_datasets)


    # whether labeling the first token of a given word
    global label_all_tokens
    label_all_tokens = True


    # add chars
    new_column = [[0]] * len(raw_train_datasets['tokens'])
    raw_train_datasets = raw_train_datasets.add_column("chars",new_column)

    train_tokenized_datasets = raw_train_datasets.map(tokenize_and_align_labels, batched=True)
    train_tokenized_datasets = train_tokenized_datasets.remove_columns(["tokens","ner_tags"])

    new_column = [[0]] * len(raw_valid_datasets['tokens'])
    raw_valid_datasets = raw_valid_datasets.add_column("chars",new_column)

    valid_tokenized_datasets = raw_valid_datasets.map(tokenize_and_align_labels, batched=True)
    valid_tokenized_datasets = valid_tokenized_datasets.remove_columns(["tokens","ner_tags"])

    new_column = [[0]] * len(raw_test_datasets['tokens'])
    raw_test_datasets = raw_test_datasets.add_column("chars",new_column)

    test_tokenized_datasets = raw_test_datasets.map(tokenize_and_align_labels, batched=True)
    test_tokenized_datasets = test_tokenized_datasets.remove_columns(["tokens","ner_tags"])

    # Create a DataLoader
    data_collator = SpeciDataCollator(tokenizer = tokenizer,padding="longest")

    train_dataloader = DataLoader(train_tokenized_datasets, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)


    valid_dataloader = DataLoader(valid_tokenized_datasets, batch_size=args.batch_size//2, collate_fn=data_collator)
    eval_dataloader = DataLoader(test_tokenized_datasets, batch_size=args.batch_size//2, collate_fn=data_collator)



    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = adjust_Hierarchical_lr(args=args,model=model)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,eps = 1e-8)

    num_training_steps = args.n_epochs * len(train_dataloader) // args.gradient_accumulation_steps

    # whether use warmup
    lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=int(args.warmup_ratio*num_training_steps), num_training_steps=num_training_steps)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    # train loop
    print("----------Begin to Train--------")
    progress_bar = tqdm(range(num_training_steps))

    # set best performance
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_epoch = 0
    best_model = None

    #early_stops_limit = 15

    for epoch in range(args.n_epochs):

        print("\n")
        print("Epoch: %d"%(epoch+1))
        train(model=model,
            train_dataloader=train_dataloader,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            progress_bar=progress_bar)


        candidate_model, API_Precision, API_Recall, API_F1 = validate(model=model,
                                                                    valid_dataloader=valid_dataloader,
                                                                    device=device)

        print("\n")
        print("Current Performence: Epoch: {}, Percison:{:.4f}, Recall:{:.4f}, F1:{:.4f}".format(epoch+1, API_Precision, API_Recall,API_F1))


        if API_F1 >= best_f1:
            best_f1 = API_F1
            best_precision = API_Precision
            best_recall = API_Recall
            no_improve = 0
            best_model = copy.deepcopy(candidate_model)
            best_epoch = epoch + 1
            print("new best score!")

        else:
            no_improve += 1
            print("no_improve: " + str(no_improve))
            '''
            if no_improve >= early_stops_limit:
                print("early stopping")
                break
            '''
        print("\n")
        print("Best Performence: Epoch: {}, Percison:{:.4f}, Recall:{:.4f}, F1:{:.4f}".format(best_epoch, best_precision, best_recall,best_f1))

    print("\n")
    print("----------------------------------------Begin to Eval------------------------------------")
    test_Precision, test_Recall, test_F1,test_MCC = test(model=best_model,eval_dataloader=eval_dataloader,device=device)
    print("Test: Percison:{:.4f}, Recall:{:.4f}, F1:{:.4f}".format(test_Precision, test_Recall, test_F1))


    print("----------Task Finished----------")



    model_save_path = args.model_save_path
    save_model_name = "{}best_model.pt".format(model_save_path)
    torch.save(best_model,save_model_name)




    excel_name = "CAREER_result.xlsx"
    if(not os.path.exists(excel_name)):
        dfData = {
        'Precision':[],
        'Recall':[],
        'F1':[],
        'MCC':[]
        }
        df = pd.DataFrame(dfData)
        df.to_excel(excel_name, index=False)
    old_df = pd.read_excel(excel_name)
    new_df = {
        'Precision':[test_Precision],
        'Recall':[test_Recall],
        'F1':[test_F1],
        'MCC':[test_MCC]
    }
    new_df = pd.DataFrame(new_df)
    df = pd.concat([old_df, new_df])
    df.to_excel(excel_name, index=False)