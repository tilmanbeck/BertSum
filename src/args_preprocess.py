import json
import random
import os
import pandas as pd
from nltk import word_tokenize

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def leave_one_out_splits_crosstopics(topics):
    print(topics)
    splits = {}
    random.seed(42)
    for i, test_topic in enumerate(topics):
        train_eval_topics = [i for i in topics if i != test_topic]  # df[df['topic'].isin()]
        random.shuffle(train_eval_topics)
        train_topics = train_eval_topics[:-1]
        eval_topics = [train_eval_topics[-1]]
        print('Split: {}, Train topics: {}, Valid topics: {}, Test topic: {}'.
              format(i, train_topics, eval_topics, test_topic))

        splits[i] = {'test': [test_topic], 'train': train_topics, 'valid': eval_topics}
    return splits

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument("-input_path", default='../json_data/')
    parser.add_argument("-output_path", default='../bert_data/')

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_nsents', default=3, type=int)
    parser.add_argument('-max_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens', default=5, type=int)
    parser.add_argument('-max_src_ntokens', default=200, type=int)

    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)

    #parser.add_argument('-log_file', default='../logs/cnndm.log')

    parser.add_argument('-dataset', default='', help='train, valid or test, defaul will process all datasets')

    parser.add_argument('-n_cpus', default=2, type=int)
    args = parser.parse_args()
    topics = ['nuclear energy', 'minimum wage', 'abortion', 
    'marijuana legalization', 'gun control', 'cloning', 'death penalty', 'school uniforms']

    splits = leave_one_out_splits_crosstopics(topics)
    with open('../raw_data/procon_debatepedia_gold.json', 'r') as fp:
        gold_args = json.load(fp)


    for i in splits.keys():
        for corpus_type in ['train', 'valid', 'test']:
            final = []
            for topic in splits[i][corpus_type]:
                pro_gold = [i['title'] for i in gold_args[topic]['pro_points']]
                con_gold = [i['title'] for i in gold_args[topic]['contra_points']]
                for stance in ['pro', 'con']:
                    res = {
                        'src': [],
                        'tgt': []
                    }
                    sentences = []
                    with open(os.path.join(args.input_path, topic.replace(' ', '') + '_' + stance + '.txt'), 'r') as fp:
                        for line in fp:
                            sentences.append(line.strip().lower())
                    for sent in sentences:
                        res['src'].append(word_tokenize(sent))
                    if stance == 'pro':
                        for sent in pro_gold:
                            res['tgt'].append(word_tokenize(sent))
                    else:
                        for sent in con_gold:
                            res['tgt'].append(word_tokenize(sent))
                final.append(res)
            pt_file = "{:s}.{:s}.{:d}.json".format('args', corpus_type, i)
            with open(os.path.join(args.output_path, pt_file), mode='w', encoding='utf-8') as fp:
                json.dump(final, fp)



