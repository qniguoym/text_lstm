import os
root_dir=os.path.dirname(__file__)
data_dir=os.path.join(root_dir,'../data')
data_path=os.path.join(data_dir,'trainData_pinyin.txt')
save_dir=os.path.join(root_dir,'../model')
letters=[' ','a','b','c','d','e','f','j','h','i','g','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
K=10
