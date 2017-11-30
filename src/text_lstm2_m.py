# coding:utf-8
import tensorflow as tf
from config import *
from tensorflow.contrib import  rnn
import os
from optparse import OptionParser
from TextLoader import TextLoader
import time
import pypinyin
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
class text_lstm():
    def __init__(self,opt,is_training=True):
        self.data_path=opt.data_path
        self.model_dir=opt.save_dir
        self.rnn_size=opt.rnn_size
        self.num_epochs=opt.num_epochs
        self.learning_rate=opt.learning_rate
        self.decay_rate=opt.decay_rate
        self.num_layers=opt.num_layers
        self.seq_length=opt.seq_length
        self.keep_prob=opt.keep_prob
        self.batch_size=opt.batch_size
        self.letter_size=len(letters)
        self.is_training=is_training
        self.le2id={}
        for i in range(len(letters)):
            self.le2id[letters[i]]=i+1
        if self.is_training:
            self.data_loader=TextLoader(self.batch_size,self.seq_length)
            self.build_graph()
            self.run()
        else:
            self.batch_size=1
            self.build_graph()
    def build_graph(self):
        cell=rnn.BasicLSTMCell(self.rnn_size,state_is_tuple=True)
        self.cell=rnn.MultiRNNCell([cell]*self.num_layers)
        if self.is_training:
            self.input_data1=tf.placeholder(tf.int32,[self.batch_size,self.seq_length])
            self.input_data2=tf.placeholder(tf.int32,[self.batch_size,self.seq_length])
            self.input_data3=tf.placeholder(tf.int32,[self.batch_size,self.seq_length])
            self.input_data4=tf.placeholder(tf.int32,[self.batch_size,self.seq_length])

            with tf.variable_scope('embedding_layer'):
                weights=tf.get_variable('weights',initializer=tf.random_normal(shape=[self.letter_size+1,self.rnn_size],stddev=0.1))
                input1=tf.nn.embedding_lookup(weights,self.input_data1)
                input2=tf.nn.embedding_lookup(weights,self.input_data2)
                input3=tf.nn.embedding_lookup(weights,self.input_data3)
                input4=tf.nn.embedding_lookup(weights,self.input_data4)

                input1 = tf.split(input1, self.seq_length, 1)
                input1 = [tf.squeeze(input_, [1]) for input_ in input1]

                input2 = tf.split(input2, self.seq_length, 1)
                input2 = [tf.squeeze(input_, [1]) for input_ in input2]

                input3 = tf.split(input3, self.seq_length, 1)
                input3 = [tf.squeeze(input_, [1]) for input_ in input3]

                input4 = tf.split(input4, self.seq_length, 1)
                input4 = [tf.squeeze(input_, [1]) for input_ in input4]

            self.outputs1,last_state1=rnn.static_rnn(self.cell,input1,dtype=tf.float32,scope='rnn_layer')
            self.outputs2,last_state2=rnn.static_rnn(self.cell,input2,dtype=tf.float32,scope='rnn_layer')
            self.outputs3,last_state3=rnn.static_rnn(self.cell,input3,dtype=tf.float32,scope='rnn_layer')
            self.outputs4,last_state4=rnn.static_rnn(self.cell,input4,dtype=tf.float32,scope='rnn_layer')

            output1=self.outputs1[-1]
            output2=self.outputs2[-1]
            output3=self.outputs3[-1]
            output4=self.outputs4[-1]

            self.cost1=tf.reduce_sum(tf.square(output1-output2),-1)
            self.cost2=tf.reduce_sum(tf.square(output3-output4),-1)

            cost=self.cost1-self.cost2
            cost=tf.clip_by_value(cost,clip_value_min=-10,clip_value_max=100)
            self.cost=tf.reduce_mean(cost)
            '''
            self.cost=tf.reduce_mean(self.cost1-self.cost2)
            self.cost = tf.clip_by_value(self.cost, clip_value_min=-10, clip_value_max=10)
            # self.optimizer = tf.train.AdadeltaOptimizer().minimize(self.cost)  # Adam Optimizer
            '''
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.cost)  # Adam Optimizer
        else:
            self.input_data1=tf.placeholder(tf.int32,[self.batch_size,self.seq_length])

            with tf.variable_scope('embedding_layer'):
                weights=tf.get_variable('weights',initializer=tf.random_normal(shape=[self.letter_size+1,self.rnn_size],stddev=0.1))
                input1=tf.nn.embedding_lookup(weights,self.input_data1)


                input1 = tf.split(input1, self.seq_length, 1)
                input1 = [tf.squeeze(input_, [1]) for input_ in input1]


            self.outputs1,last_state1=rnn.static_rnn(self.cell,input1,dtype=tf.float32,scope='rnn_layer')

            self.output1=self.outputs1[-1]
    def get_loc_num(self):
        loc_num=TextLoader(data_path=os.path.join(data_dir,'part_loc.txt'),is_training=False).get_data()
        return loc_num
    def run(self):
        with tf.Session() as sess:
            init=tf.global_variables_initializer()
            sess.run(init)
            self.saver=tf.train.Saver(max_to_keep=1)
            minv=100
            for i in range(self.num_epochs):
                self.data_loader.reset_batch_pointer()
                for j in range(self.data_loader.num_batches):
                    start=time.time()
                    x1,x2,x3,x4=self.data_loader.next_batch()
                    feed={self.input_data1:x1,self.input_data2:x2,self.input_data3:x3,self.input_data4:x4}
                    cost1,cost2,train_loss,_ =sess.run([self.cost1,self.cost2,self.cost,self.optimizer], feed_dict=feed)
                    print (np.mean(cost1),np.mean(cost2),train_loss)
                    end = time.time()
                    print ('{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.3f}'\
                    .format(i * self.data_loader.num_batches + j + 1,
                            self.num_epochs * self.data_loader.num_batches,
                            i + 1,
                            train_loss,
                            end - start))
                    if train_loss<=minv:
                        minv=train_loss
                        self.saver.save(sess,os.path.join(self.model_dir,'model.ckpt'))
    def benum(self,part_list):
        tmp=[]
        for line in part_list:
            tmp.append([])
            line=pypinyin.lazy_pinyin(line,0,errors='ignore')
            line=' '.join(line)
            tmpp=[]
            for le in line:
                tmpp.append(self.le2id.get(le))
            tmp[-1].append(tmpp)
        return tmp
    def get_asrlist(self,asr):
        sub_num={}
        sub_pos={}
        lens=len(asr)
        sco=[2,3,4,5,6,7,8,9,10]
        for i in range(lens):
            for j in sco:
                if i+j<=lens:
                    tmp=asr[i:i+j]
                    tmp_id=[]
                    tmp_pinyin=pypinyin.lazy_pinyin(tmp,0,errors='ignore')
                    tmp_pinyin=' '.join(tmp_pinyin)
                    for k in tmp_pinyin:
                        tmp_id.append(self.le2id.get(k))
                    while len(tmp_id)<self.seq_length:
                        tmp_id.append(0)
                    if len(tmp_id)>self.seq_length:
                        tmp_id=tmp_id[:self.seq_length]
                    sub_pos[tmp]=(i,i+j)
                    sub_num[tmp]=tmp_id
        return sub_num,sub_pos
    def compute_dis(self,va,vb,lens):
        #score=np.sqrt(np.sum(((np.array(va)-np.array(vb))**2)))/lens
        score=np.sqrt(np.sum(np.square(np.array(va)-np.array(vb))))/lens
        return score
    def getone(self, mn_matrix, s_loc_vec, s_sub_vec):
        imin, jmin=np.unravel_index(mn_matrix.argmin(), mn_matrix.shape)
        # print("imin:{}, jmin:{}".format(imin,jmin))
        loc=s_loc_vec[imin]
        sub=s_sub_vec[jmin]
        # print("loc:{}\n sub:{}\n".format(loc, sub))
        self.list.append(loc)
        self.loc_vec.pop(loc)
        i,j=self.sub_pos[sub]
        for k in range(i,j):
            self.cov[k]+=1
        keys=[]
        for key,value in self.sub_vec.items():
            i,j=self.sub_pos[key]
            for k in range(i,j):
                if self.cov[k]>3:
                    if key not in keys:
                        keys.append(key)
                    self.cov[k]=0
        for k in keys:
            if k in self.sub_vec:
                self.sub_vec.pop(k)
        return imin
    def getcandi(self):
        #self.vec_loc,self.vec_sub,self.sub_pos
        self.list=[]
        loc_vec_arr = [self.loc_vec[ikey] for ikey in sorted(self.loc_vec.keys(), key=lambda x: len(x), reverse=True)]
        loc_vec_np = np.array(loc_vec_arr)
        loc_vec_np = np.reshape(loc_vec_np, (loc_vec_np.shape[0], loc_vec_np.shape[-1]))
        sub_vec_arr = [self.sub_vec[jkey] for jkey in sorted(self.sub_vec.keys(), key=lambda x: len(x), reverse=True)]
        sub_vec_np = np.array(sub_vec_arr)
        sub_vec_np = np.reshape(sub_vec_np, (sub_vec_np.shape[0], sub_vec_np.shape[-1]))
        mn_matrix = pairwise_distances(loc_vec_np, sub_vec_np, metric='euclidean', n_jobs=-1)
        set_inf=np.array([float("inf") for i in range(mn_matrix.shape[-1]) ])

        s_loc_vec=sorted(self.loc_vec.keys(), key=lambda x: len(x), reverse=True)
        s_sub_vec=sorted(self.sub_vec.keys(), key=lambda x: len(x), reverse=True)

        k=0
        while k<K:
            imin =self.getone(mn_matrix, s_loc_vec, s_sub_vec)
            mn_matrix[imin]=set_inf
            k=k+1
        for i in self.list:
            print (i)
    def predict(self,asr_list, fw):
        self.ans={}
        self.saver=tf.train.Saver()
        #self.loc_num=self.get_loc_num()#得到{名字:向量}
        self.loc_num=self.get_loc_num()#
        #print("loc_num:\n{}".format(self.loc_num))
        with tf.Session() as sess:
            self.saver.restore(sess,os.path.join(self.model_dir,'model.ckpt'))
            self.loc_vec={}
            for key,value in self.loc_num.items():
                tmp=[]
                tmp.append(value)
                self.loc_vec[key]=sess.run([self.output1],feed_dict={self.input_data1:tmp})
            print("loc_vec:".format(self.loc_vec))
            #得到loc的向量表示 {loc的vec:loc的汉字}
            for asr in asr_list:
                print (asr)
                start_time=time.time()
                self.cov=np.zeros(len(asr))
                self.sub_num,self.sub_pos=self.get_asrlist(asr)
                #print("sub_num:\n{}\n sub_pos:\n{}".format(self.sub_num, self.sub_pos))
                asr_list_t=time.time()
                print("asr_sub_list time:{:.2f}".format(asr_list_t-start_time))
                self.sub_vec={}#得到sub的向量表示,[num_sub,hidden_size]
                for key,value in self.sub_num.items():
                    tmp=[]
                    tmp.append(value)
                    self.sub_vec[key]=sess.run([self.output1],feed_dict={self.input_data1:tmp})
                sub_vec_t=time.time()
                print("sub_vec:{:.2f} seconds".format(sub_vec_t-start_time))
                self.getcandi()
                cand_vec_t=time.time()
                print("cand_vec_t:{:.2f} seconds".format(cand_vec_t-start_time))
                self.ans[asr]=self.list
                #erint(self.ans[asr])
                fw.write("{}".format(asr))
                fw.write('\n'.join(self.ans[asr])+'\n\n')
    def test(self):
        asr=[]
        with open(os.path.join(data_dir,'asr.txt')) as f:
            for line in f:
                asr.append(line)
        return asr

if __name__=='__main__':
    optparser=OptionParser()
    optparser.add_option("-d","--data_path",dest="data_path",type="string",default=data_path)
    optparser.add_option("-s","--save_path",dest="save_dir",type="string",default=save_dir)
    optparser.add_option("-r","--rnn_size",dest="rnn_size",type="int",default=128)
    optparser.add_option("-q","--seq_length",dest="seq_length",type="int",default=64)
    optparser.add_option("-n","--num_layers",dest="num_layers",type="int",default=1)
    optparser.add_option("-e","--num_epochs",dest="num_epochs",type="int",default=100)
    optparser.add_option("-l","--learning_rate",dest="learning_rate",type="float",default=0.01)
    optparser.add_option("-y","--decay_rate",dest="decay_rate",type="float",default=0.9)
    optparser.add_option("-c","--batch_size",dest="batch_size",type="int",default=1000)
    optparser.add_option("-k","--keep_prob",dest="keep_prob",type="float",default=1)
    opt,args=optparser.parse_args()
    #model=text_lstm(opt)
    model=text_lstm(opt,is_training=False)
    #model.predict(['调出广汇汇景小区东门的预案。'])
    asr=model.test()
    fw=open('ans.txt','w')
    model.predict(asr, fw)
    # for i,j in model.ans.items():
    #     fw.write(i)
    #     fw.write('\n'.join(j)+'\n')




