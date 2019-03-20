import sys
user_taggedartists = 'E:/code/recommend/data/hetrec2011-lastfm-2k/user_taggedartists.dat'
tags = 'E:/code/recommend/data/hetrec2011-lastfm-2k/tags.dat'
# user_data = '../data/user_data'
# item_data = '../data/item_data'

output_file = 'E:/code/recommend/data/output_data.txt'
ofile = open(output_file,'w')


#处理用户元数据，将处理后的结果放入字典里面，key是用户id，value是用户信息
with open(user_taggedartists,'r') as fd :
    for line in fd:
        ss = line.strip().split('\t')
        if len(ss) != 6:
            continue
        userID,artistID,tagID,day,month,year = ss
        ofile.write('\001'.join([userID,artistID,tagID]))
        ofile.write('\n')

ofile.close
