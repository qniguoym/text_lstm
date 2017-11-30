from config import *
def getseqlength():
    maxv=0
    with open(data_path) as f:
        for line in f:
            line=line.strip()
            line=line.split('.')
            for i in line:
                sum=0
                for j in i:
                    if j in letters:
                        sum=sum+1
                maxv=max(maxv,sum)
        print (maxv)
def getasrlength():
    maxv=0
    with open(os.path.join(data_dir,'part_loc.txt')) as f:
        for line in f:
            line=line.strip()
            maxv=max(maxv,len(line))
        print (maxv)
if __name__=='__main__':
    getseqlength()