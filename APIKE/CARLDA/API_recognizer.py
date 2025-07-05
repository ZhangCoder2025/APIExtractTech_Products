import torch
from transformers import AutoTokenizer
import pickle
import sys
import json



def recognize_api_within_text(example):
    def pad_sequences_with_lengths(sequences, padding_value=0):
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        original_lengths = []

        for seq in sequences:
            original_lengths.append(len(seq))
            padded_seq = seq + [padding_value] * (max_len - len(seq))
            padded_sequences.append(padded_seq)

        return padded_sequences, original_lengths
    
    def predict_single(tokens: list[str]) -> list[tuple[str, str]]:
        """
        tokens: e.g. ["import", "numpy", "as", "np"]
        returns list of (token, predicted_label)
        """
        # 4.1) tokenize & align word_ids
        enc = tokenizer(tokens,
                        truncation=True,
                        is_split_into_words=True,
                        add_special_tokens=False,
                        return_tensors="pt")
        word_ids = enc.word_ids(batch_index=0)

        # 4.2) build per‐token character indices
        chars = []
        for word_idx in word_ids:
            if word_idx is None:
                chars.append([])  # special token
            else:
                w = tokens[word_idx]
                chars.append([char2index.get_index(c) for c in w])
        padded_chars,char_lengths = pad_sequences_with_lengths(chars, padding_value=0)
        feature = {
            "input_ids":     torch.tensor([enc["input_ids"][0].tolist()]),
            "attention_mask":torch.tensor([enc["attention_mask"][0].tolist()]),
            #"token_type_ids":enc.get("token_type_ids", None) and enc["token_type_ids"][0].tolist(),
            "word_ids":      torch.tensor([word_ids]),
            "chars":         torch.tensor([padded_chars]),
            "char_seq_lengths": torch.tensor([char_lengths])
        }

        # 4.3) collate into a batch
        #batch = collator([feature])
        batch = {k: v.to(DEVICE) for k, v in feature.items()}
        word_ids = batch.pop('word_ids')
        # 4.4) forward
        with torch.no_grad():
            tags = model(**batch, is_train=False)  # [batch, seq_len] of label‐ids
        tags = tags[0].cpu().tolist()

        # 5) map back to words (skipping special tokens)
        API_word_id_list = []
        for cur_tag,cur_word_id in zip(tags,word_ids[0]):
            if cur_tag==1:
                API_word_id_list.append(cur_word_id.item())
        API_word_id_list = list(set(API_word_id_list))
        return API_word_id_list
    with open("CARLDA/char_alphabet.pkl", 'rb') as char_file:
        char2index = pickle.loads(char_file.read())
    MODEL_DIR = "models/CAREER.pt"          # where best_model.pt lives
    model = torch.load(MODEL_DIR)    
    PRETRAINED_NAME   = "models/codebert-base/"         # or your args.model_location
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME,model_max_length=512,add_prefix_space=True)
    DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE).eval()
    preds = predict_single(example)
    return preds
    
if __name__ == '__main__':
    example = sys.argv[1]
    input_example = json.loads(example)
    preds = recognize_api_within_text(input_example)
    print(json.dumps(preds))
    # import torch
    # print(torch.__version__)
    # print(torch.version.cuda)
    # print(torch.cuda.is_available())

    # print(torch.cuda.is_available())  # 应该是 True
    # print(torch.cuda.get_device_name(0))  # 应该输出你的GPU型号

    # example = ["Assume", "you", "import", "numpy", "as", "np", "and", "call", "np.array", "("]
    # recognize_api_within_text(example)