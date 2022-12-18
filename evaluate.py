import argparse
import glob
import logging
import os
import random
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,

    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    get_linear_schedule_with_warmup,
)
# # import sys
# print(sys.path)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
#from transformers.data.metrics import acc_and_f1
#from transformers import xnli_output_modes as output_modes
#from transformers import xnli_processors as processors
from data_processor_for_kon import appreview_processors as processors
from data_processor_for_kon import appreview_output_modes as output_modes
from sklearn import metrics
from new_modeling_bert import BertForSequenceClassification

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
output_dir='output/'
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}".format(
            "test" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(len(processor.get_labels())),
            str(task)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        label_list1 = processor.get_labels1()
        label_list2 = processor.get_labels2()
       
        if evaluate:
            examples1,examples2,examples3,content=(processor.get_test_examples(args.data_dir))
        else:
            examples1,examples2,examples3=(processor.get_train_examples(args.data_dir))
       
        features = convert_examples_to_features(
            examples1, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
        )
        features1 = convert_examples_to_features(
            examples2, tokenizer, max_length=args.max_seq_length, label_list=label_list1, output_mode=output_mode,
        )
        features2 = convert_examples_to_features(
            examples3, tokenizer, max_length=args.max_seq_length, label_list=label_list2, output_mode=output_mode,
        )
      
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        all_labels1 = torch.tensor([f.label for f in features1], dtype=torch.long)
        all_labels2 = torch.tensor([f.label for f in features2], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` for XNLI.")
    
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels,all_labels1,all_labels2)
    return dataset

def confusion_matrix(y_true, y_pred):
    # return metrics.confusion_matrix(y_true, y_pred, range(len(processors['appreview']().get_labels())))
    return metrics.confusion_matrix(y_true, y_pred)

def basic_metrics(y_true, y_pred):
    return {'Accuracy': metrics.accuracy_score(y_true, y_pred),
            'Precision': metrics.precision_score(y_true, y_pred, average='macro'),
            'Recall': metrics.recall_score(y_true, y_pred, average='macro'),
            'Macro-F1': metrics.f1_score(y_true, y_pred, average='macro'),
            'Micro-F1': metrics.f1_score(y_true, y_pred, average='micro'),
            'ConfMat': confusion_matrix(y_true, y_pred)}

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return basic_metrics(labels, preds)

logger = logging.getLogger(__name__)

#ALL_MODELS = sum(
#    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, DistilBertConfig, XLMConfig)), ()
#)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
}


# def set_seed(args):
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.n_gpu > 0:
#         torch.cuda.manual_seed_all(args.seed)

eval_task_names = ('appreview',)
eval_outputs_dirs = (output_dir,)
local_rank=-1
def evaluate(args, model, tokenizer, prefix=""):
    results = {}
    results1={}
    results2={}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        preds1=None
        preds2=None
        out_label_ids = None
        out_label_ids1=None
        out_label_ids2=None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],"labels1":batch[4],"labels2":batch[5]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
                outputs,outputs1,outputs2 = model(**inputs)

                tmp_eval_loss, logits = outputs[:2]
                tmp_eval_loss1, logits1 = outputs1[:2]
                tmp_eval_loss2, logits2 = outputs2[:2]
              
                tmp_eval_loss=tmp_eval_loss+tmp_eval_loss1+tmp_eval_loss2
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None and preds1 is None and preds2 is None:
                preds = logits.detach().cpu().numpy()
                preds1 = logits1.detach().cpu().numpy()
                preds2 = logits2.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                out_label_ids1 = inputs["labels1"].detach().cpu().numpy()
                out_label_ids2 = inputs["labels2"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                preds1 = np.append(preds1, logits1.detach().cpu().numpy(), axis=0)
                preds2 = np.append(preds2, logits2.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                out_label_ids1 = np.append(out_label_ids1, inputs["labels1"].detach().cpu().numpy(), axis=0)
                out_label_ids2 = np.append(out_label_ids2, inputs["labels2"].detach().cpu().numpy(), axis=0)
              
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1) 
            #preds1 = np.argmax(preds1, axis=1)
            preds2 = np.argmax(preds2, axis=1)
        else:
            raise ValueError("No other `output_mode` for XNLI.")
       
        newpreds1=[]
        oldpreds=preds
        preds = preds.tolist()
        preds1 = preds1.tolist()
        
        for ind,i in enumerate(preds):
            if i==0:
                c=preds1[ind][:3].index(max(preds1[ind][:3]))
                if c==3:
                    print(preds1[ind])
                    print(c)
                newpreds1.append(c)
            if i==1 :
                c=3
                newpreds1.append(c)
                
        preds=np.array(preds)
        newpreds1=np.array(newpreds1)
        result = compute_metrics(eval_task, preds, out_label_ids)
        result1 = compute_metrics(eval_task, newpreds1, out_label_ids1)
        result2 = compute_metrics(eval_task, preds2, out_label_ids2)
        results.update(result)
        results1.update(result1)
        results2.update(result2)
        preds=preds.tolist()
        newpreds1=newpreds1.tolist()
        out_label_ids=out_label_ids.tolist()
        out_label_ids1=out_label_ids1.tolist()

        with open('output/preds.txt', 'w') as f:
            for item in preds:
                f.write("%s\n" % item)
        with open('output/preds1.txt', 'w') as f:
            for item in newpreds1:
                f.write("%s\n" % item)
        with open('output/out_label_ids.txt', 'w') as f:
            for item in out_label_ids:
                f.write("%s\n" % item)
        with open('output/out_label_ids1.txt', 'w') as f:
            for item in out_label_ids1:
                f.write("%s\n" % item)
        processor=processors['appreview']()
        examples1,examples2,examples3,content = (
            processor.get_test_examples(args.data_dir))
    
        concate=list(zip(content,out_label_ids,preds,out_label_ids1,newpreds1,out_label_ids2,preds2))
        with open('output/prediction.tsv','w') as f :
            writer=csv.writer(f,delimiter='\t')
            writer.writerow(['content', '是否有關核能','預測','立場','預測','是否有關核廢料','預測'])
            writer.writerows(concate)        
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            writer.write('是否有關核能')
            writer.write('\n')
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write('全文立場')   
            writer.write('\n')
            for key in sorted(result1.keys()):
                logger.info("  %s = %s", key, str(result1[key]))
                writer.write("%s = %s\n" % (key, str(result1[key])))
            writer.write('是否有關核廢料')
            writer.write('\n')
            for key in sorted(result2.keys()):
                logger.info("  %s = %s", key, str(result2[key]))
                writer.write("%s = %s\n" % (key, str(result2[key])))

    return results
def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='data/',
        type=str,
        
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default='bert-base-chinese',
        type=str,
    
        help="Path to pre-trained model or shortcut name selected in the list: " ,
    )
    parser.add_argument(
        "--output_dir",
        default='output/',
        type=str,
        
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()
    args.do_train=False
    args.do_eval=True
    args.overwrite_output_dir=True

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare XNLI task
    args.task_name = "appreview"
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    label_list1 = processor.get_labels1()
    label_list2 = processor.get_labels2()
    num_labels = len(label_list)
    num_labels1 = len(label_list1)
    num_labels2 = len(label_list2)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        num_labels1=num_labels1,
        num_labels2=num_labels2,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.num_labels = len(label_list)
    config.num_labels1 = len(label_list1)

    config.num_labels2 = len(label_list2)

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    
    # model.num_labels = len(label_list)
    #model.classifier = torch.nn.Linear(config.hidden_size, num_labels)
	
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    
    # Evaluation
    results = {}
    results1 = {}
    results2 = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results
if __name__ == "__main__":
    main()
