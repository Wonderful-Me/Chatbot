from operator import index
import torch
import os
import argparse
from datetime import datetime
import logging
from transformers import GPT2LMHeadModel
from transformers import BertTokenizerFast
import torch.nn.functional as F

def set_args():
    
    # Sets up the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False) # device to generate
    parser.add_argument('--temperature', default=1, type=float, required=False) # temperature generated
    parser.add_argument('--topk', default=8, type=int, required=False)
    parser.add_argument('--topp', default=0, type=float, required=False)
    parser.add_argument('--log_path', default='data/interact.log', type=str, required=False)
    parser.add_argument('--list_path', default='list/list.txt', type=str, required=False)
    parser.add_argument('--model_path', default='model/epoch40', type=str, required=False)
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False)
    parser.add_argument('--max_len', type=int, default=25) # max_input_length
    parser.add_argument('--max_history_len', type=int, default=3) # max dialogue history
    parser.add_argument('--no_cuda', action='store_true')
    # 重复惩罚参数，若生成的对话重复性较高，可适当提高该参数
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)
    return parser.parse_args()


def create_logger(args):
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # handlers for outputs
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


"""
    Learning Note:

    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

        Args:
            logits: logits distribution shape (vocab size)

            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al.
            
            articles: (http://arxiv.org/abs/1904.09751)

            or find it at this way: ./References/The Curious Case of Neural Text Degeneration.pdf

        From: https://github.com/thomwolf?tab=repositories

"""

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    
    # My text is generated one character by another
    
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk() return the last token of the top-k like: (values,indices)
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')  # other elements are all set to -float('Inf')

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')
    return logits


def main():
    args = set_args()
    # store the Info
    logger = create_logger(args)

    # choose the device
    args.cuda = torch.cuda.is_available() and not args.no_cuda and torch.cuda.device_count() >= 1
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    # logging function
    logger.info('using device:{}'.format(device))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tokenizer = BertTokenizerFast(
        vocab_file=args.list_path, 
        sep_token="[SEP]", 
        pad_token="[PAD]", 
        cls_token="[CLS]"
    )

    # load my model
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    
    # initial the record directory and files
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
   
    # save the conversations
    history = []
    print('开始和chatbot聊天，输入CTRL + Z以退出')

    while True:
        try:
            text = input("user:")
            if args.save_samples_path:
                samples_file.write("user:{}\n".format(text))
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            history.append(text_ids)
            input_ids = [tokenizer.cls_token_id]  # input begin with [CLS]

            # get the len num of history dialogs to generate sentence for this time
            for index, history_dialogue in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_dialogue)
                # sentences are divided by [SEP] (sep_token_id)
                input_ids.append(tokenizer.sep_token_id)
            input_ids = torch.tensor(input_ids).long().to(device)
            
            # clear
            input_ids = input_ids.unsqueeze(0)
            
            # store the output response
            response = []
            # generate the response according to the history_dialogue
            for _ in range(args.max_len):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]

                # give the toke that generated a penalty value
                # if the dailogues generated are highly similar with each other, we can raise this value slightly through the args
                for id in set(response):
                    next_token_logits[id] /= args.repetition_penalty
                
                next_token_logits = next_token_logits / args.temperature

                # token can not be [UNK]
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                
                '''
                    Learning Note:

                    这里用中文做注释，怕说不清楚

                    torch.multinomial表示从集合中无放回地抽取num_samples个元素
                    说明：权重越高→抽到的几率越高 
                    返回元素的下标
                    这里做softmax的归一化操作 而后生成运用的是贪心算法
                    相当于对于每个中文字符的生成都是根据上下文得出一个可能性最大的值作为输出的结果
                    当遇到[SEP]即预示着生成的结束

                    top_k_top_p_filtering:

                    这个函数的处理思路参照了以下作者thomwolf的仓库内的GPT2的项目源码:
                    From: https://github.com/thomwolf/gpt-2-simple

                    论文依据:
                    articles: (http://arxiv.org/abs/1904.09751)

                '''
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                if next_token == tokenizer.sep_token_id:  # [SEP] → generate response comes to an end
                    break
                response.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                # print("his_text:{}".format(his_text))
            
            history.append(response)
            text = tokenizer.convert_ids_to_tokens(response)
            print("chatbot:" + "".join(text))
            if args.save_samples_path:
                samples_file.write("chatbot:{}\n".format("".join(text)))

        except KeyboardInterrupt:
            if args.save_samples_path:
                samples_file.close()
            break


if __name__ == '__main__':
    main()
