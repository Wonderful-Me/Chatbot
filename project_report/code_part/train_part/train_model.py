import argparse
import torch
import pickle
import torch.nn.functional as F
from datetime import datetime
import os
import logging
from torch.utils.data import DataLoader
from os.path import join
from torch.nn import DataParallel
import transformers
from transformers import AdamW
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset



# data_set
class MyDataset(Dataset):

    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels


def validate_epoch(model, validate_dataloader, logger, epoch, args):
    logger.info("validating part")
    model.eval()
    device = args.device
    epoch_start_time = datetime.now()

    total_loss = 0
    # cuda out of memory exception
    try:
        with torch.no_grad():
            for index, (input_ids, labels) in enumerate(validate_dataloader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                loss = outputs.loss
                loss = loss.mean()

                # clear the data 
                total_loss += loss.item()
                del input_ids, outputs

            # record the average loss for this epoch
            epoch_mean_loss = total_loss / len(validate_dataloader)
            logger.info(
                "validate epoch {}: loss {}".format(epoch+1, epoch_mean_loss))
            epoch_finish_time = datetime.now()
            logger.info('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
            return epoch_mean_loss

    
    except RuntimeError as exc:
        '''
        Learning Note:

        Pytorch out of memory:
        
        Reference Link: https://blog.csdn.net/Dooonald/article/details/89875620

        Code: 
        try:

            # calculate
            
            ... # loss, outputs = model(src, lengths, dec, targets)
            
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
        raise e
        '''

        if "out of memory" in str(exc):
            print('| WARNING: ran out of memory')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else: 
            raise exc

def train_epoch(model, train_dataloader, optimizer, scheduler, logger, epoch, args):
    model.train()
    device = args.device
    ignore_index = args.ignore_index
    total_loss = 0

    # epoch_total_num → sum of epoch_correct_num
    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        # get the cuda out of memory exception
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            # batch based record
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
            # epoch based record
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # calculate the accuracy
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # to be decided
            # print('batch ', batch_idx, 'of epoch ',epoch, ': acc:[', batch_acc, '] loss:[', loss, ']')

            loss.backward()
            # gradient value clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # update the args timely
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # update the args
                optimizer.step()
                # update the lr
                scheduler.step()
                # clear the grad (reset to 0)
                optimizer.zero_grad()

            del input_ids, outputs
            
            # write into the log file
            if (batch_idx + 1) % args.log_step == 0:
                logger.info(
                    "epoch: {}, batch: {}\n loss: {}, batch_acc: {}, lr: {}".format(
                        epoch + 1,
                        batch_idx + 1,
                        loss.item() * args.gradient_accumulation_steps,
                        batch_acc,
                        scheduler.get_lr()
                    )
                )

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception

    # record the average loss for this epoch
    epoch_mean_loss = total_loss / len(train_dataloader)
    
    # save the model
    model_path = join(args.save_model_path, 'epoch{}'.format(epoch + 1))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_path)
    return epoch_mean_loss

def train(model, logger, train_dataset, validate_dataset, args):
    '''
    Learning Note:

    torch.utils.data.dataloader:

    parameter_parsing: 
    https://zhuanlan.zhihu.com/p/346332974
    
    Official Reference Menu: 
    https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
    
    main_point: shuffle, collate_fn, drop_last
    '''

    # load data for train
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,   # swap the batch sequence randomly
        num_workers=args.num_workers, #default: 0
        collate_fn=collate_fn,
        drop_last=True
    )
    
    # loda data for validation
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=args.batch_size,
        shuffle=True,   # swap the batch sequence randomly
        num_workers=args.num_workers, #default: 0
        collate_fn=collate_fn,
        drop_last=True
    )
    
    # # early stop module
    # early_stopping = EarlyStopping(
    #     args.patience,
    #     verbose=True,
    #     save_path=args.save_model_path
    # )
    

    '''
    Learning Note:

    1. len(dataloder())
    The length of dataloder will change according to the batch_size and data_set
    data_set = 1000, batch_size = 100  →  len(dataloder) = 10
    
    2. // 
    Returns the integer part of the quotient (rounded down)  →  [x]

    '''
    total_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps, # default warmup_steps = 4000
        num_training_steps=total_steps
    )

    # begin the training
    logger.info('starting training')

    # record losses of training and validating
    train_losses, validate_losses = [], []
    # record the minimum loss
    best_val_loss = 10000
    
    # main_body for training
    for epoch in range(args.epochs):
        # train
        train_loss = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            epoch=epoch,
            args=args
        )
        train_losses.append(train_loss)

        # validate
        validate_loss = validate_epoch(
            model=model,
            validate_dataloader=validate_dataloader,
            epoch=epoch,
            args=args
        )
        validate_losses.append(validate_loss)

        logger.info('training finished')
        logger.info("train_losses:{}".format(train_losses))
        logger.info("validate_losses:{}".format(validate_losses))
        
        # # There's no early stop if patience == 0
        # if args.patience == 0:
        #     continue
        # # decide whether to early stop or not
        # early_stopping(validate_loss, model)
        # if early_stopping.early_stop:
        #     break

def caculate_loss(logit, target, pad_idx, smoothing=True):
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)

        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        # loss = F.cross_entropy(predict_logit, target, ignore_index=pad_idx)
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # return the max index for each one
    # generate a mask_tenser
    # padding_part → 0, data_part → 1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='3', type=str, required=False, help='devices to use')
    parser.add_argument('--no_cuda', action='store_true', help='not to use GPU')
    parser.add_argument('--list_path', default='list/list.txt', type=str, required=False, help='path to vocab list')
    parser.add_argument('--model_config', default='config/config.json', type=str, required=False, help='configs of the model')
    parser.add_argument('--train_path', default='data/train.pkg', type=str, required=False, help='path to train.pkg')
    parser.add_argument('--max_len', default=150, type=int, required=False, help='the max length of the input info')
    parser.add_argument('--ignore_index', default=-100, type=int, required=False, help='taken with ignore_index can avoid the calculation of gradient')
    parser.add_argument('--epochs', default=100, type=int, required=False, help='the epochs to train')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='batch size for training')
    parser.add_argument('--gpu0_bsz', default=10, type=int, required=False, help='batch size for device 0')
    parser.add_argument('--lr', default=2.6e-5, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='衰减率')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='report loss every * steps')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False, help='gradient accumulation')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, required=False)
    parser.add_argument('--save_model_path', default='model', type=str, required=False, help='output path for trained_model')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='path to the pretrained_model')
    parser.add_argument('--num_workers', type=int, default=0, help="number of threads used by the dataloader to load data")
    parser.add_argument('--patience', type=int, default=0, help="symbol of early stopping")
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warm up steps')
    parser.add_argument('--val_num', type=int, default=8000, help='size of the validation set')
    parser.add_argument('--log_path', default='data/train.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--log', default=True, help="是否记录日志")
    args = parser.parse_args()
    return args

def create_logger(args):
    
    #create logger
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

def main():

    # initial the arguments
    args = set_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device # devices for training
    
    if( args.no_cuda ):
        args.cuda = 0
    else:
        args.cuda = 1

    if torch.cuda.is_available() and args.cuda :
        device = 'cuda:0'
    else:
        device = 'cpu'
    args.device = device

    # create the logger
    logger = create_logger(args)
    logger.info('using device:{}'.format(device))


    # initial tokenizer
    # bert_name = 'bert-base-chinese'
    # Delimiter encoding
    tokenizer = BertTokenizerFast(vocab_file=args.list_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    args.sep_id = tokenizer.sep_token_id    # seperate one sentence        [SEP] code: 102
    args.pad_id = tokenizer.pad_token_id    # padding to a exact length    [PAD] code: 0
    args.cls_id = tokenizer.cls_token_id    # seperate one conversation    [CLS] code: 101
    
    '''
    Learning Note:

    The same:
    
    if not model.config.vocab_size == tokenizer.vocab_size:
        raise AssertionError
    
    '''

    # output path for the model
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # creat the train_model
    if not args.pretrained_model: 
        # creat a new one
        model_config = GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model = model.to(device)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))

    # detect if the sizes match, or it will cause errors!
    assert model.config.vocab_size == tokenizer.vocab_size
    
    # train the model with devices ( > 1 )
    if args.cuda and torch.cuda.device_count() > 1:
        '''
        Learning Note:

        Pytorch.nn.DataParrallel
        References: https://zhuanlan.zhihu.com/p/102697821

        the usage of the device is like: 2 : 1 : 1 : 1 : ...
        thus, device-0 is always two times as the othier devices

        The usage of device 0 is more than the other devices ( about two times )

        '''
        model = DataParallel(model).cuda()
        logger.info("use GPU {} to train".format(args.device))

    # load dataset    
    train_path = args.train_path

    with open(train_path, "rb") as f:
        input_list = pickle.load(f)

    # devide the dataset for train and calidation
    val_num = args.val_num
    input_list_train = input_list[val_num:]
    input_list_val = input_list[:val_num]

    train_dataset = MyDataset(input_list_train, args.max_len)
    validate_dataset = MyDataset(input_list_val, args.max_len)

    # calculate the number of parameters
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))
    logger.info("args:{}".format(args))

    train(model, logger, train_dataset, validate_dataset, args)

if __name__ == '__main__':
    main()
