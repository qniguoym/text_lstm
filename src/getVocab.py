from config import *
def creatvocab():
    fw=open(os.path.join(data_dir,'vocab.txt'),'w')
    vocab={}
    with open(os.path.join(data_dir,'trainData_pinyin.txt')) as f:
        for line in f:
            line=line.strip()
            for i in line:
                if i in vocab:
                    vocab[i]+1
                else:
                    vocab[i]=1

    vocab=sorted(vocab.keys(), key=lambda x:vocab.get(x),reverse=True)
    fw.write(str(vocab))

if __name__=='__main__':
    creatvocab()
