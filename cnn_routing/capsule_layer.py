import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import torch.nn.functional as F


class CapsuleLayer(nn.Module):
    """
    The core implementation of the idea of capsules
    """

    def __init__(self, in_unit, in_channel, num_unit, unit_size, vocab,
                 num_routing, cuda_enabled):
        super(CapsuleLayer, self).__init__()

        self.in_unit = in_unit
        self.in_channel = in_channel
        self.num_unit = num_unit
        self.vocab = vocab
        self.num_routing = num_routing
        self.cuda_enabled = cuda_enabled
        
        """
        Based on the paper, DigitCaps which is capsule layer(s) with
        capsule inputs use a routing algorithm that uses this weight matrix, Wij
        """
        # weight shape:
        # [1 x primary_unit_size x num_classes x output_unit_size x num_primary_unit]
        # == [1 x 1152 x 10 x 16 x 8]
        self.weight = nn.Parameter(torch.randn(1, in_channel, num_unit, unit_size, in_unit))
        print('weight: ' , self.weight.shape)

    def forward(self, x, x_original, mode, epoch):
        return self.routing(x,x_original,mode,epoch)
        

    def routing(self, x, x_original,mode,epoch):
        #print('x: ', x.data.shape)
        #print('x: ', len(x))
        #print('x_original: ', x_original.data.shape)
        
        """
        Routing algorithm for capsule.

        :input: tensor x of shape [128, 8, 1152]

        :return: vector output of capsule j
        """
        batch_size = x.size(0)

        x = x.transpose(1, 2) # dim 1 and dim 2 are swapped. out tensor shape: [128, 1152, 8]
        #print('in routing: ', x)
        # Stacking and adding a dimension to a tensor.
        # stack ops output shape: [128, 1152, 10, 8]
        # unsqueeze ops output shape: [128, 1152, 10, 8, 1]
        x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)

        # Convert single weight to batch weight.
        # [1 x 1152 x 10 x 16 x 8] to: [128, 1152, 10, 16, 8]
        batch_weight = torch.cat([self.weight] * batch_size, dim=0)

        # u_hat is "prediction vectors" from the capsules in the layer below.
        # Transform inputs by weight matrix.
        # Matrix product of 2 tensors with shape: [128, 1152, 10, 16, 8] x [128, 1152, 10, 8, 1]
        # u_hat shape: [128, 1152, 10, 16, 1]
        #print('u_hat x: ', x)
        u_hat = torch.matmul(batch_weight, x)

        # All the routing logits (b_ij in the paper) are initialized to zero.
        # self.in_channel = primary_unit_size = 32 * 6 * 6 = 1152
        # self.num_unit = num_classes = 10
        # b_ij shape: [1, 1152, 10, 1]
        b_ij = Variable(torch.zeros(1, self.in_channel, self.num_unit, 1))
        if self.cuda_enabled:
            #print('@@@@@@@@@@@@@@@@@@@  b_ij GPU in use    @@@@@@@@@@@@@@@@@')
            b_ij = b_ij.cuda()
        #else:
            #print("#################     b_ij GPU NOT in use #################")

        # From the paper in the "Capsules on MNIST" section,
        # the sample MNIST test reconstructions of a CapsNet with 3 routing iterations.
        num_iterations = self.num_routing

        for iteration in range(num_iterations):
            # Routing algorithm

            # Calculate routing or also known as coupling coefficients (c_ij).
            # c_ij shape: [1, 1152, 10, 1]
            c_ij = F.softmax(b_ij, dim=2)  # Convert routing logits (b_ij) to softmax.
            #print('c_ij: ',iteration, c_ij)
            # c_ij shape from: [128, 1152, 10, 1] to: [128, 1152, 10, 1, 1]
            
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            
            if mode == 'test':                
                log_path_3_gram = 'log/' + str(epoch) + ' 3_gram.txt'
                log_path_4_gram = 'log/' + str(epoch) + ' 4_gram.txt'
                log_path_5_gram = 'log/' + str(epoch) + ' 5_gram.txt'
                fo3 = open(log_path_3_gram, "a")
                fo4 = open(log_path_4_gram, "a")
                fo5 = open(log_path_5_gram, "a")
                for i in range(0, len(x)):
                    #print('b_data=====================================================================', i)    
                    fo3.write('Data =====================================================================  ' + str(i))
                    fo4.write('Data =====================================================================  ' + str(i))
                    fo5.write('Data =====================================================================  ' + str(i))
                    
                    sentence_with_c_3 = ''
                    sentence_with_c_4 = ''
                    sentence_with_c_5 = ''
                    
                    for count in range(0, 3):  
                        sentence = ''
                        if count == 0:
                            for j in range(0, 59):
                                index = x_original.data[i][j]                           
                                word = self.vocab.itos[index]
                                if j < 57 and index != 1:
                                    index_1 = x_original.data[i][j+1]
                                    index_2 = x_original.data[i][j+2]
                                    word_1 = self.vocab.itos[index_1]
                                    word_2 = self.vocab.itos[index_2]
                                    c_value = str(c_ij[i][j].data[0].numpy()[0][0]) + ', ' + str(c_ij[i][j].data[1].numpy()[0][0])                                
                                    sentence_with_c_3 += word + ' ' +  word_1 + ' ' + word_2 + '        c_value : '  + c_value  + '\n'
                                    sentence += ' ' + word
                            #print('iteration  ===========================: ', iteration)
                            #print('3-gram: \n' , sentence)
                            #print('\n' , sentence_with_c_3)
                            fo3.write('\niteration  =========================== ' + str(iteration))
                            fo3.write('\n 3-gram: \n')
                            fo3.write(sentence)
                            fo3.write('\n')
                            fo3.write(sentence_with_c_3)
                        if count == 1:
                            for j in range(0, 59):
                                index = x_original.data[i][j]
                                word = self.vocab.itos[index]
                                if j < 56 and index != 1:
                                    index_1 = x_original.data[i][j+1]
                                    index_2 = x_original.data[i][j+2]
                                    index_3 = x_original.data[i][j+3]
                                    word_1 = self.vocab.itos[index_1]
                                    word_2 = self.vocab.itos[index_2]
                                    word_3 = self.vocab.itos[index_3]
                                    c_value = str(c_ij[i][j].data[0].numpy()[0][0]) + ', ' + str(c_ij[i][j].data[1].numpy()[0][0])     
                                    sentence_with_c_4 += word + ' ' +  word_1 + ' ' + word_2 + ' ' + word_3 + '        c_value : '  + c_value + '\n'                            
                                    sentence += ' ' + word
                            #print('iteration  ===========================: ', iteration)
                            #print('4-gram: \n' , sentence)
                            #print('\n' , sentence_with_c_4)
                            fo4.write('\niteration  =========================== ' + str(iteration))
                            fo4.write('\n4-gram: \n')
                            fo4.write(sentence)
                            fo4.write('\n')
                            fo4.write(sentence_with_c_4)
                        if count == 2:
                            for j in range(0, 59):
                                index = x_original.data[i][j]
                                word = self.vocab.itos[index]
                                if j < 55 and index != 1:
                                    index_1 = x_original.data[i][j+1]
                                    index_2 = x_original.data[i][j+2]
                                    index_3 = x_original.data[i][j+3]
                                    index_4 = x_original.data[i][j+4]
                                    word_1 = self.vocab.itos[index_1]
                                    word_2 = self.vocab.itos[index_2]
                                    word_3 = self.vocab.itos[index_3]
                                    word_4 = self.vocab.itos[index_4]
                                    c_value = str(c_ij[i][j].data[0].numpy()[0][0]) + ', ' + str(c_ij[i][j].data[1].numpy()[0][0])
                                    sentence_with_c_5 += word + ' ' +  word_1 + ' ' + word_2 + ' ' + word_3 + ' ' +word_4 + '        c_value : '  + c_value + '\n'
                                    sentence += ' ' + word
                            #print('iteration  ===========================: ', iteration)
                            #print('5-gram: \n' , sentence)
                            #print('\n' , sentence_with_c_5)
                            fo5.write('\niteration  =========================== ' + str(iteration))
                            fo5.write('\n5-gram: \n')
                            fo5.write(sentence)
                            fo5.write('\n')
                            fo5.write(sentence_with_c_5)                    
                
            #print('c_ij: ', c_ij.data.shape)
            # Implement equation 2 in the paper.
            # s_j is total input to a capsule, is a weigthed sum over all "prediction vectors".
            # u_hat is weighted inputs, prediction ˆuj|i made by capsule i.
            # c_ij * u_hat shape: [128, 1152, 10, 16, 1]
            # s_j output shape: [batch_size=128, 1, 10, 16, 1]
            # Sum of Primary Capsules outputs, 1152D becomes 1D.
            #print('c_ij type: ', type(c_ij))
            #print('u_hat type: ', type(u_hat))
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # Squash the vector output of capsule j.
            # v_j shape: [batch_size, weighted sum of PrimaryCaps output,
            #             num_classes, output_unit_size from u_hat, 1]
            # == [128, 1, 10, 16, 1]
            # So, the length of the output vector of a capsule is 16, which is in dim 3.
            v_j = utils.squash(s_j, dim=3)

            # in_channel is 1152.
            # v_j1 shape: [128, 1152, 10, 16, 1]
            v_j1 = torch.cat([v_j] * self.in_channel, dim=1)

            # The agreement.
            # Transpose u_hat with shape [128, 1152, 10, 16, 1] to [128, 1152, 10, 1, 16],
            # so we can do matrix product u_hat and v_j1.
            # u_vj1 shape: [1, 1152, 10, 1]
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update routing (b_ij) by adding the agreement to the initial logit.
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1) # shape: [128, 10, 16, 1]