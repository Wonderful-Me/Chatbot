from transformers import BertTokenizerFast
import argparse
import pickle
from tqdm import tqdm 

# each block's structure after perprocess:
# [CLS] sentence1 [SEP] sentence2 [SEP] sentence3 [SEP]

def set_args():
    # parameters
    parser = argparse.ArgumentParser()
    # list_path
    parser.add_argument('--list_path', default='list/list.txt', type=str, required=False)
    # train_data
    parser.add_argument('--train_path', default='data/train.txt', type=str, required=False)
    # train_result
    parser.add_argument('--save_path', default='data/train.pkg', type=str, required=False)
    args = parser.parse_args()
    return args

def preprocess():

    args = set_args()

    # initial tokenizer
    tokenizer = BertTokenizerFast(vocab_file=args.list_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id

    # read train_data
    with open(args.train_path, 'rb') as f:
        data = f.read().decode("utf-8")

    # tokenize
    sentence_list = []
    
    # Linux or Windows
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")

    with open(args.save_path, "w", encoding="utf-8") as f:
        for index, block in enumerate(tqdm(train_data)):
            # Linux or Windows
            if "\r\n" in data:
                sentences = block.split("\r\n")
            else:
                sentences = block.split("\n")
            
            # add [CLS]
            input_ids = [cls_id]
            for sentence in sentences:
                input_ids += tokenizer.encode(sentence, add_special_tokens=False)

                # add [SEP]
                input_ids.append(sep_id)

            # save the sentence
            sentence_list.append(input_ids)

        # end
        print('data preprocess done successfully!')
    with open(args.save_path, "wb") as f:
        pickle.dump(sentence_list, f)

if __name__ == '__main__':
    preprocess()

