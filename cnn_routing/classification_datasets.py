import re
import os
import random
import codecs
from torchtext import data
import torchtext.datasets as datasets
random.seed(800)

dev_batch_size = 50
test_batch_size = 50

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class MR(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with codecs.open(os.path.join(path, 'rt-polarity.neg'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with codecs.open(os.path.join(path, 'rt-polarity.pos'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True ,root='.',path="./dataset/", **kwargs):
        examples = cls(text_field, label_field, path=path, **kwargs).examples        
        random.shuffle(examples)
        train_end_index = int(0.8*len(examples))
        dev_end_index = int(0.9*len(examples))

        print('total train: ', len(examples))
        
        return (cls(text_field, label_field, examples=examples[:train_end_index]),
                cls(text_field, label_field, examples=examples[train_end_index:dev_end_index]),
                cls(text_field, label_field, examples=examples[dev_end_index:]),
                )
        
        
class SST2(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, examples=None):     
        
        fields = [('text', text_field), ('label', label_field)]        
        super(SST2, self).__init__(examples, fields)

    @classmethod
    def splits(cls, text_field, label_field, t_train, t_dev, t_test): 
        train_examples = []
        dev_examples = []
        test_examples = []
        
        for x in t_train:
            if 'positive'in x.label:
                x.label = 'positive'
                train_examples.append(x)
            if 'negative' in x.label:
                x.label = 'negative'
                train_examples.append(x)
                
        for x in t_dev:            
            if 'positive'in x.label:
                x.label = 'positive'
                dev_examples.append(x)
            if 'negative' in x.label:
                x.label = 'negative'
                dev_examples.append(x)
                
        for x in t_test:            
            if 'positive'in x.label:
                x.label = 'positive'
                test_examples.append(x)
            if 'negative' in x.label:
                x.label = 'negative'
                test_examples.append(x)
            

        random.shuffle(train_examples)
        random.shuffle(dev_examples)
        random.shuffle(test_examples)        
        print('2: ', len(train_examples), len(dev_examples), len(test_examples))
        return (cls(text_field, label_field, examples=train_examples),
                cls(text_field, label_field, examples=dev_examples),
                cls(text_field, label_field, examples=test_examples),
                ) 
        
        
class TREC_WITH_VALDATE(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, examples=None):     
        
        fields = [('text', text_field), ('label', label_field)]        
        super(TREC_WITH_VALDATE, self).__init__(examples, fields)

    @classmethod
    def splits(cls, text_field, label_field, t_train, t_test): 
        
        temp_t = []
        for item in t_train:
            temp_t.append(item)
            
        random.shuffle(temp_t)
        
        dict_v = {}
        for x in temp_t:            
            key = x.label.split(':')[0]
            if key not in dict_v.keys():
                dict_v[key] = 0
            else:
                dict_v[key] += 1
        print('1. dec: ', dict_v)
        
        dev_index = 0
        
        train_examples = []
        dev_examples = []
        test_examples = []
        
        for x in temp_t:
            dev_index += 1
            key = x.label.split(':')[0]
            x.label = key            
            if dev_index < len(t_train) * 0.90:                
                train_examples.append(x)
            else:
                dev_examples.append(x)
        
        for x in t_test:            
            key = x.label.split(':')[0]
            x.label = key
            test_examples.append(x)

        dict_v = {}
        for x in train_examples:
            k = x.label            
            if k not in dict_v.keys():
                dict_v[x.label] = 0
            else:
                dict_v[x.label] += 1
        print('2. dec: ', dict_v)
        random.shuffle(train_examples)
        random.shuffle(dev_examples)
        random.shuffle(test_examples)  
        
        print('3: ', len(train_examples), len(dev_examples), len(test_examples))
        
        return (cls(text_field, label_field, examples=train_examples),
                cls(text_field, label_field, examples=dev_examples),
                cls(text_field, label_field, examples=test_examples),
                )
 
class CR(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with codecs.open(os.path.join(path, 'custrev.neg'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with codecs.open(os.path.join(path, 'custrev.pos'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(CR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True ,root='.',path="./dataset/", **kwargs):
        
        examples = cls(text_field, label_field, path=path, **kwargs).examples

        random.shuffle(examples)
        
        train_end_index = int(0.8*len(examples))
        dev_end_index = int(0.9*len(examples))

        print('total train: ', len(examples))
        
        return (cls(text_field, label_field, examples=examples[:train_end_index]),
                cls(text_field, label_field, examples=examples[train_end_index:dev_end_index]),
                cls(text_field, label_field, examples=examples[dev_end_index:]),
                )  


class MPQA(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with codecs.open(os.path.join(path, 'mpqa.neg'), 'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with codecs.open(os.path.join(path, 'mpqa.pos'), 'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MPQA, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True ,root='.',path="./dataset/", **kwargs):
        
        examples = cls(text_field, label_field, path=path, **kwargs).examples

        random.shuffle(examples)
        
        train_end_index = int(0.8*len(examples))
        dev_end_index = int(0.9*len(examples))

        print('total train: ', len(examples))
        
        return (cls(text_field, label_field, examples=examples[:train_end_index]),
                cls(text_field, label_field, examples=examples[train_end_index:dev_end_index]),
                cls(text_field, label_field, examples=examples[dev_end_index:]),
                )  
        
        
class SUBJ(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with codecs.open(os.path.join(path, 'quote.tok.gt9.5000'), 'r','iso-8859-15') as f:
                examples += [
                    data.Example.fromlist([line, 'subjective'], fields) for line in f]            
            with codecs.open(os.path.join(path, 'plot.tok.gt9.5000'), 'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line, 'objective'], fields) for line in f]
            
        super(SUBJ, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True ,root='.',path="./dataset/", **kwargs):
        
        examples = cls(text_field, label_field, path=path, **kwargs).examples

        random.shuffle(examples)
        
        train_end_index = int(0.8*len(examples))
        dev_end_index = int(0.9*len(examples))

        
        print('total train: ', len(examples))
        
        return (cls(text_field, label_field, examples=examples[:train_end_index]),
                cls(text_field, label_field, examples=examples[train_end_index:dev_end_index]),
                cls(text_field, label_field, examples=examples[dev_end_index:]),
                )  
    


class IMDB(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, examples=None):     
        
        fields = [('text', text_field), ('label', label_field)]        
        super(IMDB, self).__init__(examples, fields)

    @classmethod
    def splits(cls, text_field, label_field, t_train, t_test): 
        
        temp_t = []
        for item in t_train:
            temp_t.append(item)
            
        random.shuffle(temp_t)
        
        dict_v = {}
        for x in temp_t:            
            key = x.label.split(':')[0]
            if key not in dict_v.keys():
                dict_v[key] = 0
            else:
                dict_v[key] += 1
        print('1. dec: ', dict_v)
        
        dev_index = 0
        
        train_examples = []
        dev_examples = []
        test_examples = []
        
        for x in temp_t:
            dev_index += 1
            key = x.label.split(':')[0]
            x.label = key            
            if dev_index < len(t_train) * 0.90:                
                train_examples.append(x)
            else:
                dev_examples.append(x)
        
        for x in t_test:            
            key = x.label.split(':')[0]
            x.label = key
            test_examples.append(x)

        dict_v = {}
        for x in train_examples:
            k = x.label            
            if k not in dict_v.keys():
                dict_v[x.label] = 0
            else:
                dict_v[x.label] += 1
        print('2. dec: ', dict_v)
        random.shuffle(train_examples)
        random.shuffle(dev_examples)
        random.shuffle(test_examples)  
        
        print('3: ', len(train_examples), len(dev_examples), len(test_examples))
        
        return (cls(text_field, label_field, examples=train_examples),
                cls(text_field, label_field, examples=dev_examples),
                cls(text_field, label_field, examples=test_examples),
                )


    
# load MR dataset
def load_mr(text_field, label_field, batch_size):
    print('mr, loading data...')    
    train_data, dev_data, test_data = MR.splits(text_field, label_field)    
    dict_v = {}
    for x in train_data:
        k = x.label        
        if k not in dict_v.keys():
            dict_v[x.label] = 0
        else:
            dict_v[x.label] += 1
    print('train: ', dict_v)
    
    dict_v = {}
    for x in dev_data:
        k = x.label            
        if k not in dict_v.keys():
            dict_v[x.label] = 0
        else:
            dict_v[x.label] += 1
    print('dev: ', dict_v)
    
    dict_v = {}
    for x in test_data:
        k = x.label            
        if k not in dict_v.keys():
            dict_v[x.label] = 0
        else:
            dict_v[x.label] += 1
    print('test: ', dict_v)
    
    print('train: ', len(train_data), ' dev: ', len(dev_data), ' test: ', len(test_data))
    
    max_length = 0
    for data_item in train_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
        
    for data_item in dev_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
            
    for data_item in test_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
            
    text_field.fix_length = max_length 
    
    
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    
    print('label_field vocab: ', len(label_field.vocab),label_field.vocab.itos[0])
    print('label_field vocab: ', len(label_field.vocab),label_field.vocab.itos[1])
    print('label_field vocab: ', len(label_field.vocab),label_field.vocab.itos[2])
    
    print('vocab: ', len(text_field.vocab),text_field.vocab.itos[0])
    print('vocab: ', len(text_field.vocab),text_field.vocab.itos[1])
    print('vocab: ', len(text_field.vocab),text_field.vocab.itos[2])
    print('vocab: ', len(text_field.vocab),text_field.vocab.itos[3])
       
    
    print('building batches')
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_data, dev_data, test_data), batch_sizes=(batch_size, dev_batch_size, test_batch_size), repeat=False, device = -1)
    #max_length = max_length*4 - 10
    return train_iter, dev_iter, test_iter, max_length 

# load SST-5 dataset
def load_sst_5(text_field, label_field,  batch_size):
    print('sst5, loading data...')
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)    
    print('sst5: ', len(train_data), len(dev_data), len(test_data))
    
    max_length = 0
    for data_item in train_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
            
    for data_item in dev_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
            
    for data_item in test_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)        
            
            
    text_field.fix_length = max_length 
    
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(batch_size, dev_batch_size, test_batch_size), repeat=False, device = -1)
    #max_length = max_length*4 - 10
    return train_iter, dev_iter, test_iter, max_length 

# load SST-2 dataset
def load_sst_2(text_field, label_field,  batch_size):   
    print('sst2, loading data...')
    t_train, t_dev, t_test = datasets.SST.splits(text_field, label_field, fine_grained=True)  
    train_data , dev_data , test_data = SST2.splits(text_field, label_field, t_train, t_dev, t_test) 
    print('sst2: ', len(train_data), len(dev_data), len(test_data))
    
    max_length = 0
    for data_item in train_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
    text_field.fix_length = max_length 
    
    for data_item in dev_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
            
    for data_item in test_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
    
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(batch_size, dev_batch_size, test_batch_size), repeat=False, device = -1)
    #max_length = max_length*4 - 10
    return train_iter, dev_iter, test_iter, max_length

# load TREC dataset
def load_trec(text_field, label_field,  batch_size):
    print('trec, loading data...')
    t_train, t_test = datasets.TREC.splits(text_field, label_field, fine_grained=True)        
    train_data , dev_data , test_data = TREC_WITH_VALDATE.splits(text_field, label_field, t_train, t_test)    
    print('trec: ', len(train_data), len(dev_data), len(test_data))
    
    
    max_length = 0
    for data_item in train_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
    
    for data_item in dev_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
            
    for data_item in test_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
    
    text_field.fix_length = max_length
    
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(batch_size, dev_batch_size, test_batch_size), repeat=False, device = -1)
    #max_length = max_length*4 - 10
    return train_iter, dev_iter, test_iter, max_length


# load CR dataset
def load_cr(text_field, label_field, batch_size):
    print('cr, loading data...')
    train_data, dev_data, test_data = CR.splits(text_field, label_field)    
    
    dict_v = {}
    for x in train_data:
        k = x.label        
        if k not in dict_v.keys():
            dict_v[x.label] = 0
        else:
            dict_v[x.label] += 1
    print('train: ', dict_v)
    
    dict_v = {}
    for x in dev_data:
        k = x.label            
        if k not in dict_v.keys():
            dict_v[x.label] = 0
        else:
            dict_v[x.label] += 1
    print('dev: ', dict_v)
    
    dict_v = {}
    for x in test_data:
        k = x.label            
        if k not in dict_v.keys():
            dict_v[x.label] = 0
        else:
            dict_v[x.label] += 1
    print('test: ', dict_v)
    
    print('train: ', len(train_data), ' dev: ', len(dev_data), ' test: ', len(test_data))
    
    max_length = 0
    for data_item in train_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
    
    for data_item in dev_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
            
    for data_item in test_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
    
    text_field.fix_length = max_length
    
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    print('building batches')
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_data, dev_data, test_data), batch_sizes=(batch_size, dev_batch_size, test_batch_size), repeat=False, device = -1, sort=False)
    #max_length = max_length*4 - 10
    return train_iter, dev_iter, test_iter, max_length

# load MPQA dataset
def load_mpqa(text_field, label_field, batch_size):
    print('mpqa, loading data...')
    train_data, dev_data, test_data = MPQA.splits(text_field, label_field)    
    
    dict_v = {}
    for x in train_data:
        k = x.label        
        if k not in dict_v.keys():
            dict_v[x.label] = 0
        else:
            dict_v[x.label] += 1
    print('train: ', dict_v)
    
    dict_v = {}
    for x in dev_data:
        k = x.label            
        if k not in dict_v.keys():
            dict_v[x.label] = 0
        else:
            dict_v[x.label] += 1
    print('dev: ', dict_v)
    
    dict_v = {}
    for x in test_data:
        k = x.label            
        if k not in dict_v.keys():
            dict_v[x.label] = 0
        else:
            dict_v[x.label] += 1
    print('test: ', dict_v)
    
    print('train: ', len(train_data), ' dev: ', len(dev_data), ' test: ', len(test_data))
    
    max_length = 0
    for data_item in train_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
    
    for data_item in dev_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
            
    for data_item in test_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
    
    text_field.fix_length = max_length
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    print('building batches')
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_data, dev_data, test_data), batch_sizes=(batch_size, dev_batch_size, test_batch_size), repeat=False, device = -1)
    #max_length = max_length*4 - 10
    return train_iter, dev_iter, test_iter, max_length


# load subj dataset
def load_subj(text_field, label_field, batch_size):
    print('subj, loading data...')
    train_data, dev_data, test_data = SUBJ.splits(text_field, label_field)    
    
    dict_v = {}
    for x in train_data:
        k = x.label        
        if k not in dict_v.keys():
            dict_v[x.label] = 0
        else:
            dict_v[x.label] += 1
    print('train: ', dict_v)
    
    dict_v = {}
    for x in dev_data:
        k = x.label            
        if k not in dict_v.keys():
            dict_v[x.label] = 0
        else:
            dict_v[x.label] += 1
    print('dev: ', dict_v)
    
    dict_v = {}
    for x in test_data:
        k = x.label            
        if k not in dict_v.keys():
            dict_v[x.label] = 0
        else:
            dict_v[x.label] += 1
    print('test: ', dict_v)
    
    print('train: ', len(train_data), ' dev: ', len(dev_data), ' test: ', len(test_data))
    
    max_length = 0
    for data_item in train_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
    
    for data_item in dev_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
            
    for data_item in test_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
    
    text_field.fix_length = max_length
    
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    print('building batches')
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_data, dev_data, test_data), batch_sizes=(batch_size, dev_batch_size, test_batch_size), repeat=False, device = -1)
    #max_length = max_length*4 - 10
    return train_iter, dev_iter, test_iter, max_length

# load imdb dataset
def load_imdb(text_field, label_field,  batch_size):
    print('imdb, loading data...')
    t_train, t_test = datasets.IMDB.splits(text_field, label_field)        
    train_data , dev_data , test_data = IMDB.splits(text_field, label_field, t_train, t_test)    
    print('imdb: ', len(train_data), len(dev_data), len(test_data))
    
    max_length = 0
    for data_item in train_data:
        if len(data_item.text) > max_length:
            max_length = len(data_item.text)
    text_field.fix_length = max_length
    
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(batch_size, dev_batch_size, test_batch_size), repeat=False, device = -1)
    
    return train_iter, dev_iter, test_iter 