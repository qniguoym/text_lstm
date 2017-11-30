import numpy as np
import pypinyin
from config import *
class TextLoader(object):
    def __init__(self, batch_size=3, seq_length=50,is_training=True,data_path=data_path):
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.le2id={}
        for i in range(len(letters)):
            self.le2id[letters[i]]=i+1

        if is_training:
            data=[]
            with open(data_path) as f:
                for line in f:
                    data1=[]
                    line=line.strip()
                    line=line.split('.')
                    for part in line:
                        data2=[]
                        for letter in part:
                            if letter in letters:
                                data2.append(self.le2id.get(letter))
                        while len(data2)<self.seq_length:
                            data2.append(0)
                        if len(data2)>self.seq_length:
                            data2=data2[:self.seq_length]
                        data1.append(data2)
                    data.append(data1)
            self.data=data
            self.num_data=len(self.data)
            self.reset_batch_pointer()
        else:
            data_pinyin={}#{loc的汉字:pinyin的num}
            data=[]
            with open(data_path) as f:
                for line in f:
                    data.append([])
                    line=line.strip()
                    line_pinyin=pypinyin.lazy_pinyin(line,0,errors='ignore')
                    line_pinyin=' '.join(line_pinyin)
                    for le in line_pinyin:
                        data[-1].append(self.le2id.get(le))
                    while len(data[-1])<self.seq_length:
                            data[-1].append(0)
                    if len(data[-1])>self.seq_length:
                            data[-1]=data[-1][:self.seq_length]
                    data_pinyin[line]=data[-1]
            self.data=data
            self.data_pinyin=data_pinyin
            self.num_data=len(self.data)


    def create_batches(self):
        self.num_batches = int(self.num_data / self.batch_size)
        if self.num_batches == 0:
            assert False, 'Not enough data, make batch_size small.'
        self.x_tensor = self.data[:self.num_batches * self.batch_size]

    def next_batch(self):
        x = self.x_tensor[self.batch_size*self.pointer:self.batch_size*(self.pointer+1)]
        self.pointer += 1
        x=np.array(x)
        x1=x[:,0]
        x2=x[:,1]
        x3=x[:,2]
        x4=x[:,3]
        return x1,x2,x3,x4

    def reset_batch_pointer(self):
        self.create_batches()
        self.pointer = 0

    def get_data(self):#返回的是向量,{向量:名字}
        return self.data_pinyin

if __name__=='__main__':
    pass
