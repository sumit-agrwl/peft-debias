import json
import sys
import logging
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

group_bias_words = [
        "jewish",
        "christian",
        "muslim",
        "sikh",
        "buddhist",
        "hindu",
        "atheist",
        "torah",
        "bible",
        "quran",
        "gita",
        "synagogue",
        "church",
        "mosque",
        "temple",
        "monastery",
        "judaism",
        "christianity",
        "islam",
        "sikhism",
        "buddhism",
        "hinduism",
        "atheism",
        "allah",
        "god",
        "christ",
        "caucasian",
        "asian",
        "indian",
        "mexican",
        "american",
        "african",
        "european",
        "chinese",
        "hispanic",
        "black",
        "white",
        "brown",
        "africa",
        "america",
        "asia",
        "europe",
        "mexico",
        "india",
        "china",
        "gay",
        "lesbian",
        "straight",
        "homosexual",
        "asexual",
        "transgender",
        "trans",
        "queer",
        "bisexual"
    ]

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_options()

    def add_options(self):
        self.parser.add_argument('--dataset', type=str, default=None, required=True)
        self.parser.add_argument('--output_dir', type=str, default=None, required=True)
        self.parser.add_argument('--axis', type=str, default=None, required=True)


    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir)/ opt.name
        model_dir = expr_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()
        return opt


def init_logger(filename=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logger

def get_fprd(data_attr, df, pred, text_column='text', label_column='label'):
    overall_fpr = []
    _fpr = defaultdict(lambda : [])
    FPRD = 0

    for i in tqdm(range(len(df))):
        text = df[i][text_column]
        fpr = int(df[i][label_column]==0 and int(pred[i][1])==1)
        overall_fpr.append(fpr)
        for attr in data_attr:
            if attr in text:
                _fpr[attr].append(fpr)    

    for k in _fpr:
        _fpr[k] = sum(_fpr[k])/ len(_fpr[k])
    overall_fpr = sum(overall_fpr)/len(overall_fpr)

    import pdb
    pdb.set_trace()


    for k in _fpr:
        FPRD = FPRD + abs(_fpr[k] - overall_fpr)
    
    return FPRD

def calc_gab_group(opt):

    # calculate in domain metric

    pred = [item.strip().split('\t') for item in open(os.path.join(opt.output_dir, 'predictions_in_domain.txt')).readlines()][1:]

    data = [json.loads(item) for item in open('data/gab/test.jsonl','r').readlines()]
    assert (len(pred) == len(data))

    in_domain_FPRD = get_fprd(group_bias_words, data, pred)

    # calculate out domain metric

    pred_metrics = json.load(open(os.path.join(opt.output_dir, 'predict_out_of_domain_results.json')))
    pred = [item.strip().split('\t') for item in open(os.path.join(opt.output_dir, 'predictions_out_of_domain.txt')).readlines()][1:]
    

    df = pd.read_csv('data/iptts77k/test.csv')
    data = [json.loads(df.loc[index].to_json()) for index in range(len(df))]
    assert (len(pred) == len(data))


    out_domain_FPRD = get_fprd(group_bias_words, data, pred, text_column='Text', label_column='Label')

    in_domain_pred_metrics = json.load(open(os.path.join(opt.output_dir, 'predict_in_domain_results.json')))
    out_domain_pred_metrics = json.load(open(os.path.join(opt.output_dir, 'predict_out_of_domain_results.json')))

    logger.info(f"In Domain F1 : {in_domain_pred_metrics['predict_f1']*100:.2f}")
    logger.info(f"In Domain FPRD : {in_domain_FPRD:.2f}")
    logger.info(f"IPTTS F1 : {out_domain_pred_metrics['predict_f1']*100:.2f}")
    logger.info(f"IPTTS FPRD : {out_domain_FPRD:.2f}")


def calc_ws_group(opt):

    pred = [item.strip().split('\t') for item in open(os.path.join(opt.output_dir, 'predictions_in_domain.txt')).readlines()][1:]

    data = [json.loads(item) for item in open('data/ws/test.jsonl','r').readlines()]
    assert (len(pred) == len(data))

    in_domain_FPRD = get_fprd(group_bias_words, data, pred)

    # calculate out domain metric

    pred_metrics = json.load(open(os.path.join(opt.output_dir, 'predict_out_of_domain_results.json')))
    pred = [item.strip().split('\t') for item in open(os.path.join(opt.output_dir, 'predictions_out_of_domain.txt')).readlines()][1:]
    

    df = pd.read_csv('data/iptts77k/test.csv')
    data = [json.loads(df.loc[index].to_json()) for index in range(len(df))]
    assert (len(pred) == len(data))


    out_domain_FPRD = get_fprd(group_bias_words, data, pred, text_column='Text', label_column='Label')

    in_domain_pred_metrics = json.load(open(os.path.join(opt.output_dir, 'predict_in_domain_results.json')))
    out_domain_pred_metrics = json.load(open(os.path.join(opt.output_dir, 'predict_out_of_domain_results.json')))

    logger.info(f"In Domain F1 : {in_domain_pred_metrics['predict_f1']*100:.2f}")
    logger.info(f"In Domain FPRD : {in_domain_FPRD:.2f}")
    logger.info(f"IPTTS F1 : {out_domain_pred_metrics['predict_f1']*100:.2f}")
    logger.info(f"IPTTS FPRD : {out_domain_FPRD:.2f}")


def calc_fdcl_group(opt):

    # calculate in domain metric
    pred = [item.strip().split('\t') for item in open(os.path.join(opt.output_dir, 'predictions_in_domain.txt')).readlines()][1:]

    data = [json.loads(item) for item in open('data/fdcl/test.jsonl','r').readlines()]
    assert (len(pred) == len(data))

    in_domain_FPRD = get_fprd(group_bias_words, data, pred, label_column='ND_label')

    # calculate out domain metric

    pred_metrics = json.load(open(os.path.join(opt.output_dir, 'predict_out_of_domain_results.json')))
    pred = [item.strip().split('\t') for item in open(os.path.join(opt.output_dir, 'predictions_out_of_domain.txt')).readlines()][1:]
    

    df = pd.read_csv('data/iptts77k/test.csv')
    data = [json.loads(df.loc[index].to_json()) for index in range(len(df))]
    assert (len(pred) == len(data))

    out_domain_FPRD = get_fprd(group_bias_words, data, pred, text_column='Text', label_column='Label')
    # out_domain_FPRD = get_fprd(group_bias_words, data, pred, text_column='Text', label_column='Label')

    in_domain_pred_metrics = json.load(open(os.path.join(opt.output_dir, 'predict_in_domain_results.json')))
    out_domain_pred_metrics = json.load(open(os.path.join(opt.output_dir, 'predict_out_of_domain_results.json')))

    logger.info(f"In Domain F1 : {in_domain_pred_metrics['predict_f1']*100:.2f}")
    logger.info(f"In Domain FPRD : {in_domain_FPRD:.2f}")
    logger.info(f"IPTTS F1 : {out_domain_pred_metrics['predict_f1']*100:.2f}")
    logger.info(f"IPTTS FPRD : {out_domain_FPRD:.2f}")


def calc_bias_bios_gender(opt):
    _tpr = defaultdict(lambda : {'m' : [], 'f' : []})

    pred = [item.strip().split('\t') for item in open(os.path.join(opt.output_dir, 'predictions_in_domain.txt')).readlines()][1:]
    data = [json.loads(item) for item in open('data/bias-bios/test.jsonl','r').readlines()]
    
    assert len(pred) == len(data)
    for i in tqdm(range(len(data))):
        gender = data[i]['g']
        label = data[i]['p']
        prediction = pred[i][1]
        _tpr[label][gender].append(int(label==prediction))

    for k in _tpr:
        _tpr[k]['m'] = sum(_tpr[k]['m']) / len(_tpr[k]['m'])
        _tpr[k]['f'] = sum(_tpr[k]['f']) / len(_tpr[k]['f'])
        _tpr[k] = _tpr[k]['m'] - _tpr[k]['f']

    RMSE = 0.0
    for k in _tpr:
        RMSE = RMSE + (_tpr[k])**2
    
    RMSE = (RMSE / len(_tpr))**0.5
    in_domain_pred_metrics = json.load(open(os.path.join(opt.output_dir, 'predict_in_domain_results.json')))
    logger.info(f"In Domain Acc : {in_domain_pred_metrics['predict_accuracy']*100:.2f}")
    logger.info(f"In Domain RMSE : {RMSE*100:.2f}")


def main():
    
    options = Options()
    opt = options.parse()
    logger = init_logger(os.path.join(opt.output_dir, 'extrinsic.log'))

    if opt.dataset == 'gab':
        opt.axis = 'group'
        calc_gab_group(opt)
    if opt.dataset == 'bias-bios':
        opt.axis = 'gender'
        calc_bias_bios_gender(opt)
    if opt.dataset == 'ws':
        opt.axis = 'group'
        calc_ws_group(opt)
    if opt.dataset == 'fdcl':
        if opt.axis == 'group':
            calc_fdcl_group(opt)
        elif opt.axis == 'dialect':
            calc_fdcl_dialect(opt)


if __name__ == "__main__":
    main()





