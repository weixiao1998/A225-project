import jieba,re

f_in_path = "text.txt"
f_out_path = "segm.txt"

with open(f_in_path,"r") as f_in,open(f_out_path,"w",encoding="utf-8") as f_out:
    for line in f_in:
        line = re.sub(r"\s{2,}","，",line)#去多余空格
        #line = re.sub('[，。]+','',line)#去标点
        seg_list = jieba.cut(line)
        f_out.write(" ".join(seg_list))
