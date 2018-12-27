# -*- coding: utf-8 -*-
import torch
import argparse
import torch.autograd as autograd
import torch.nn as nn
import utils
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import classification_datasets as cld
import random
import torch.utils.data as Data
from capsule_layer import CapsuleLayer
from torch.autograd import Variable
from tqdm import tqdm
from timeit import default_timer as timer

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(800)

use_cuda = torch.cuda.is_available()
#use_cuda = False

parser = argparse.ArgumentParser(description='text classificer')
parser.add_argument('-dataset', type=str, default='mr', help='data set selection')
parser.add_argument('-mode', type=str, default='train', help='mode')
parser.add_argument('-use-pre-train', type=bool, default=True, help='use w2v')
parser.add_argument('-w2v', type=str, default='glove.6B.300d', help='w2v')
parser.add_argument('-unit-size', type=int, default=100, help='output unit size')
parser.add_argument('-num-routing', type=int, default=5, help='number of routing')
parser.add_argument('-batch-size', type=int, default=50, help='batch size')
parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('-weight-decay', type=float, default=0.0, help='weight decay')
parser.add_argument('-kernel-num', type=int, default=5, help='number of each kind of kernel')
parser.add_argument('-num-classes', type=int, default=2, help='number of class')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-log-interval', type=int, default=10, help='how many batches to wait before logging training status. default=10')
parser.add_argument('-dropout', type=float, default=0.3, help='dropout')
parser.add_argument('-epochs', type=int, default=30, help='epochs. default=30')


args = parser.parse_args()
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]


print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))
print('-----------------------------------------')


        
        
class SimpleClassifier(nn.Module):

    def __init__(self, embedding_dim, vocab_size, label_size, batch_size, vocab_vectors, routing_unit_num, vocab):
        super(SimpleClassifier, self).__init__()
        
        self.vocab = vocab
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)        
        if args.use_pre_train:
            self.word_embeddings.weight.data.copy_(vocab_vectors)
        else:
            print('random initiallize')
        #self.word_embeddings.weight.requires_grad = False
        print(self.word_embeddings.weight.requires_grad)        
        
        Ks = args.kernel_sizes    
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        D = embedding_dim
        
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        
        self.digits = CapsuleLayer(in_unit = args.kernel_num,
                                   in_channel = routing_unit_num,
                                   num_unit = 2,
                                   unit_size = args.unit_size, 
                                   vocab = vocab,
                                   num_routing = args.num_routing,
                                   cuda_enabled = use_cuda)        
        self.dropout = nn.Dropout(args.dropout)  
       
    def forward(self, x, mode, epoch):        
        x = x.permute(1,0)
        x_original = x
        #print('1.x: ', x.data.shape)
        #for d in x.data[0]:
        #    print(self.vocab.itos[d])
        x = self.word_embeddings(x)     
        #print('2.x: ', x.data.shape)        
        #x.contiguous()
        x = x.unsqueeze(1)
        #print('3.x: ', x.shape)
        x = [conv(x).squeeze(3) for conv in self.convs1]
        #print(x[0].shape, x[1].shape, x[2].shape)
        x = torch.cat(x, 2)        
        #print('x0: ', x.data.shape)
        x = utils.squash(x, dim=1)      
        #print('5 x: ', x.shape) 
        x = self.digits(x, x_original, mode, epoch)
        x = self.dropout(x)
        #print('6 x: ', x.shape) 
        
        return x             

def train():   
    
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    print('text_field: ', type(text_field))
    
    if 'mr' == args.dataset:
        train_iter, dev_iter , test_iter, routing_unit_num = cld.load_mr(text_field, label_field, batch_size=args.batch_size)
    elif 'cr' == args.dataset:
        train_iter, dev_iter , test_iter, routing_unit_num = cld.load_cr(text_field, label_field, batch_size=args.batch_size)
    elif 'subj' == args.dataset:
        train_iter, dev_iter , test_iter, routing_unit_num = cld.load_subj(text_field, label_field, batch_size=args.batch_size)
    elif 'sst2' == args.dataset:
        train_iter, dev_iter , test_iter, routing_unit_num = cld.load_sst_2(text_field, label_field, batch_size=args.batch_size)
    elif 'sst5' == args.dataset:
        train_iter, dev_iter , test_iter, routing_unit_num = cld.load_sst_5(text_field, label_field, batch_size=args.batch_size)
    elif 'mpqa' == args.dataset:
        train_iter, dev_iter , test_iter, routing_unit_num = cld.load_mpqa(text_field, label_field, batch_size=args.batch_size)
    elif 'trec' == args.dataset:
        train_iter, dev_iter , test_iter, routing_unit_num = cld.load_trec(text_field, label_field, batch_size=args.batch_size)

    routing_unit_num = routing_unit_num*len(args.kernel_sizes) - 9
    
    if args.use_pre_train:
        text_field.vocab.load_vectors(args.w2v)
        EMBEDDING_DIM = text_field.vocab.vectors.shape[1]
    else:
        EMBEDDING_DIM = 300
        print('1 . random initiallize')
        
    model = SimpleClassifier(embedding_dim=EMBEDDING_DIM,
                           vocab_size=len(text_field.vocab),label_size=len(label_field.vocab)-1,
                            batch_size=args.batch_size,
                            vocab_vectors=text_field.vocab.vectors,
                            routing_unit_num = routing_unit_num,
                            vocab=text_field.vocab)
    print('Parameters and size:')
    for name, param in model.named_parameters():
       print('{}: {}'.format(name, list(param.size())))    
    
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

    if use_cuda:
        print('------use cuda------')
        model = model.cuda()  
    # Set the logger
    
    
    for i in range(args.epochs):
        print('=============================== epoch: %d start!===============================' % (i+1))
        train_epoch(model, train_iter, optimizer, text_field, label_field, i+1, dev_iter, mode='train')        
        test(model, test_iter, i + 1, mode='test') 
        print('=============================== epoch: %d done!================================'%(i+1))
               
    
def test(model, test_iter, epoch, mode):
    print("Test Mode")   

    steps = 0 
    correct = 0
    t_correct = 0
    for batch in test_iter:
        steps += 1        
        sent, labels = batch.text, batch.label          
                
        sent = sent.cuda() if use_cuda else sent        
        labels = labels.cuda() if use_cuda else labels
        
        labels.data.sub_(1)        
        #model.batch_size = len(labels.data)           
        output = model(sent, mode, epoch)
        
        # Count number of correct predictions
        # v_magnitude shape: [128, 10, 1, 1]
        v_magnitude = torch.sqrt((output**2).sum(dim=2, keepdim=True))
        #print('v_magnitude: ', v_magnitude.data.shape)
        # pred shape: [128, 1, 1, 1]
        pred = v_magnitude.data.max(1, keepdim=True)[1]
        #print('pred: ', pred.shape)
        correct += pred.eq(labels.data.view_as(pred)).sum()
                
        pred = pred.squeeze(1)
        pred = pred.squeeze(1)
        pred = pred.squeeze(1)       
            
        sent = sent.permute(1,0)        
        log_path = 'log/' + str(epoch) + '_stat.txt'
        fo = open(log_path, "a")
        
        for i in range(0, len(sent)):
            sentence_length = 0
            sentence = ''           
            for j in range(0, 59):
                index = sent.data[i][j]        
                if 1 != index:
                    sentence_length += 1                            
                    word = model.vocab.itos[index]
                    sentence += ' ' + word
                
            if pred[i] == labels.data[i]:
                sentence = '1@@@' + str(sentence_length) + '@@@' + sentence + '\n'
                t_correct+=1
            else:
                sentence = '2@@@' + str(sentence_length) + '@@@' + sentence + '\n'
            fo.write(sentence)             
            
            
    num_test_data = len(test_iter.dataset)
    accuracy = correct / num_test_data
    accuracy_percentage = 100. * accuracy    
    
    print('t_correct: ', t_correct)
    print('Test Accuracy: {}/{} ({:.0f}%)\n'.format(correct, num_test_data, accuracy_percentage))


def train_epoch(model, train_iter, optimizer, text_field, label_field, epoch, dev_iter, mode):
    model.train()
    print("Train Mode")    
    steps = 0     
    
    num_batches = len(train_iter) # iteration per epoch. e.g: 469
    total_step = args.epochs * num_batches
    epoch_tot_acc = 0.0    

    start_time = timer()   
    
    
    for batch in train_iter:
        steps += 1
        global_step = steps + (epoch * num_batches) - num_batches
        
        sent, labels = batch.text, batch.label     
        
        target = labels.data
        batch_size = len(labels.data)   
        #print('target: ', target)
        target_one_hot = utils.one_hot_encode(target, length=args.num_classes)
        assert target_one_hot.size() == torch.Size([batch_size, 2])
        target = Variable(target_one_hot)
        #print('target_one_hot: ', target_one_hot)
        sent = sent.cuda() if use_cuda else sent
        labels = labels.cuda() if use_cuda else labels
        target = target.cuda() if use_cuda else target
        
        labels.data.sub_(1)        
        model.batch_size = len(labels.data)    
        
        model.zero_grad()        
        output = model(sent,mode,epoch)
        m_loss = margin_loss(output, target)        
        m_loss = m_loss.mean()        
        m_loss.backward()
        optimizer.step()    
                   
        
        # Calculate accuracy for each step and average accuracy for each epoch
        acc = utils.accuracy(output, labels.data.long(), use_cuda)
        
        epoch_tot_acc += acc
        epoch_avg_acc = epoch_tot_acc / steps     
        
        
        if steps % args.log_interval == 0:
            template = 'Epoch {}/{},  Step {}/{}: [Margin loss: {:.6f}, \tBatch accuracy: {:.6f}, \tAccuracy: {:.6f}]'
            tqdm.write(template.format(epoch, args.epochs,  global_step,  total_step, m_loss.data[0], acc, epoch_avg_acc))
            

    end_time = timer()
    print('Time elapsed for epoch {}: {:.0f}s.'.format(epoch, end_time - start_time))      
  
def margin_loss(input, target):
    """
    Class loss
    Implement equation 4 in section 3 'Margin loss for digit existence' in the paper.
    Args:
        input: [batch_size, 10, 16, 1] The output from `DigitCaps` layer.
        target: target: [batch_size, 10] One-hot MNIST labels.
    Returns:
        l_c: A scalar of class loss or also know as margin loss.
    """
    batch_size = input.size(0)

    # ||vc|| also known as norm.
    v_c = torch.sqrt((input**2).sum(dim=2, keepdim=True))

    # Calculate left and right max() terms.
    zero = Variable(torch.zeros(1))
    
    zero = zero.cuda() if use_cuda else zero

    m_plus = 0.9
    m_minus = 0.1
    loss_lambda = 0.5
    max_left = torch.max(m_plus - v_c, zero).view(batch_size, -1)**2
    max_right = torch.max(v_c - m_minus, zero).view(batch_size, -1)**2
    t_c = target
    # Lc is margin loss for each digit of class c
    l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
    l_c = l_c.sum(dim=1)
    return l_c    


train()