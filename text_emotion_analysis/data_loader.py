import gensim.models,jieba,os,math,random
import matplotlib.pyplot as plt
import numpy as np

class DataLoader(object):
    def __init__(self,stop_words_path):
        super().__init__()

        print("开始载入停用词...")
        self.stop_words = self.get_stop_words(stop_words_path)
        print("停用词数：{}".format(len(self.stop_words)))

    def get_words_vec(self,model_path,vocab,vec_dim):
        '''
        获取词向量字典
        :param model_path: 模型路径
        :param vocab: VocabularyProcessor中的vocabulary
        :param vec_dim: 词向量维度
        :return: 词向量字典
        '''
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path,binary=True)
        vec = model.wv
        del model

        words_vec = {}
        for word in vocab._reverse_mapping:
            try:
                words_vec[word] = vec[word]
            except:
                words_vec[word] = np.random.uniform(-1.0, 1.0, vec_dim).astype(np.float32)
        return words_vec

    def get_stop_words(self,fpath):
        '''
        获取停用词列表
        :param fpath: 停用词文件路径
        :return: 停用词列表
        '''
        stopwords = []
        with open(fpath, "r", encoding='gbk') as f:
            lines = f.readlines()
            for line in lines:
                stopwords.append(line.strip())
        return stopwords

    def check_word_limit(self,data,min_word,max_word):
        '''
        剔除不在词数范围内的数据
        :param data: 原始数据（分词后）
        :param min_word: 词数最小值
        :param max_word: 词数最大值
        :return: 新数据
        '''
        new_data = []
        for each in data:
            e_len = len(each.split(" "))
            if e_len<min_word or e_len>max_word:
                continue
            else:
                new_data.append(each)
        return new_data

    def get_seg_list(self,sentence):
        '''
        获取分词后生成的列表
        :param sentence: 字符串
        :return: 分词列表
        '''
        first_seg_list = list(jieba.cut(sentence))
        seg_list = []
        for each in first_seg_list:
            if each not in self.stop_words:
                seg_list.append(each)
        return seg_list

    def participle(self,data):
        '''
        分词
        :param data:
        :return:
        '''
        new_data = []
        for each in data:
            seg_list = self.get_seg_list(each)
            if len(seg_list)>0:
                sentence = " ".join(seg_list)
                new_data.append(sentence)
        return new_data

    def tokenizer(self,data):
        for each in data:
            yield each.split(" ")

    def load_data(self,dir,file_names):
        '''
        读入数据
        :param dir: 数据储存文件夹
        :param file_names: 消极与积极数据的文件名
        :return: 消极与积极数据
        '''
        neg_path = os.path.join(dir,file_names[0])
        pos_path = os.path.join(dir,file_names[1])

        def read_data_file(path):
            '''
            读入文件内容
            :param path: 文件路径
            :return: 文件内容
            '''
            data = []
            temp = ""
            with open(path,"r") as f:
                for line in f:
                    if line=="\n":
                        data.append(temp.strip("\n").strip())
                        temp = ""
                    else:
                        temp+=line
            return data

        neg_data = read_data_file(neg_path)
        pos_data = read_data_file(pos_path)
        return neg_data,pos_data

    def balance_data(self,neg_data,pos_data,allow_bias=1000):
        '''
        平衡数据（允许偏差）
        :param neg_data: 消极数据
        :param pos_data: 积极数据
        :param allow_bias: 允许的偏差
        :return: 平衡后的数据
        '''
        neg_size = len(neg_data)
        pos_size = len(pos_data)
        if abs(neg_size-pos_size)<=allow_bias:
            return neg_data,pos_data
        elif neg_size-pos_size>allow_bias:
            return random.sample(neg_data, pos_size+allow_bias),pos_data
        else:
            return neg_data,random.sample(pos_data, neg_size+allow_bias)

    def random_choose_data(self,neg_data,pos_data,max_num):
        '''
        平衡数据（不允许有偏差）
        :param neg_data: 消极数据
        :param pos_data: 积极数据
        :param max_num: 最大数量
        :return: 平衡后的数据
        '''
        min_size = min(len(neg_data),len(pos_data))
        final_size = min(min_size,max_num)
        return random.sample(neg_data,final_size), random.sample(pos_data,final_size)

    def show_len_distribution_1(self,neg_data,pos_data):
        '''
        数据长度条状图
        :param neg_data:消极数据
        :param pos_data:积极数据
        :return:
        '''
        def add_count(text, dict):
            text_len = len(text)
            if text_len in dict:
                dict[text_len] += 1
            else:
                dict[text_len] = 1

        count_neg = {}
        count_pos = {}

        for each in pos_data:
            add_count(each, count_pos)
        for each in neg_data:
            add_count(each, count_neg)

        neg_len_list = sorted(count_neg.items(), key=lambda item: item[1])
        pos_len_list = sorted(count_pos.items(), key=lambda item: item[1])

        neg_keys = [item[0] for item in neg_len_list]
        neg_counts = [item[1] for item in neg_len_list]
        pos_keys = [item[0] for item in pos_len_list]
        pos_counts = [item[1] for item in pos_len_list]

        max_len = 300
        max_cnt = max(max(neg_counts), max(pos_counts))
        neg_list = []
        pos_list = []

        print(max_len,max_cnt)

        for i in range(0, max_len + 1):
            if i in neg_keys:
                neg_list.append(count_neg[i])
            else:
                neg_list.append(0)
            if i in pos_keys:
                pos_list.append(count_pos[i])
            else:
                pos_list.append(0)

        x = range(max_len + 1)

        plt.bar(x, neg_list, width=0.4, alpha=0.8, color='red', label="pos")
        plt.bar([i + 0.4 for i in x], pos_list, width=0.4, alpha=0.8, color='green', label="neg")

        plt.ylim(0, max_cnt + 5)
        plt.ylabel("counts")

        plt.xlabel("length")
        plt.xticks([index + 0.2 for index in x], x)

        plt.show()

    def show_len_distribution_2(self, neg_data, pos_data):
        '''
        数据长度直方图
        :param neg_data:消极数据
        :param pos_data:积极数据
        :return:
        '''
        data = [len(each) for each in neg_data]
        data.extend([len(each) for each in pos_data])
        plt.hist(data, bins=200, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.show()

    def load_x(self,data,words_vec,max_text_dim,vec_dim):
        '''
        根据数据内容生成词向量矩阵
        :param data: 文本数据
        :param max_text_dim: 数据中最大文本词数
        :return: x 词向量矩阵
        '''
        data_size = len(data)
        x_data = []
        for idx in range(data_size):
            sentence = data[idx]
            words = sentence.split(" ")
            each_x = []
            for word in words:
                try:
                    each_x.append(words_vec[word].tolist())  # 将每个词映射到词向量，并将其加入矩阵中
                except:
                    pass  # 若没有该词则忽略

            list_zero = [0] * vec_dim  # vec_dim维零向量(1*vec_dim大小的0矩阵)
            for cnt in range(max_text_dim - len(each_x)):  # 不足长度的用零向量填充
                each_x.append(list_zero)

            x_data.append(each_x)
        return x_data


    def process_original_data(self,file_path,save_files):
        '''
        处理原始数据，将数据分类加入积极或消极数据文件中
        :param file_path: 要处理的原始数据文件
        :param save_files: 积极和消极数据文件
        :return:
        '''
        neg_file = open(save_files[0],"a+")
        pos_file = open(save_files[1],"a+")
        with open(file_path, "r", encoding="gbk") as f:
            pos = neg = 0
            content = ""
            for line in f:
                if len(line) == 1:
                    continue
                if line.find(':') != -1:
                    temp = line.split(":")
                    if (temp[0] == "content"):
                        content=temp[1]
                    elif(temp[0] == "score"):
                        if temp[1][0] == '1':
                            neg_file.write(content)
                            neg_file.write("\n")
                            neg += 1
                        elif temp[1][0] == '5':
                            pos_file.write(content)
                            pos_file.write("\n")
                            pos += 1
                    else:
                        content += line#处理换行情况
                else:
                    content += line#处理换行情况
        neg_file.close()
        pos_file.close()
        print("已加入 消极数据{}条 积极数据{}条".format(neg,pos))

    def init_data(self):
        '''
        初始化两极数据文件
        :return:
        '''
        save_files = ["./data/neg.txt", "./data/pos.txt"]
        origin_path = "./data/origin"

        if os.path.exists(save_files[0]) or os.path.exists(save_files[1]):
            print("数据已初始化！")
            return

        data_list = os.listdir(origin_path)#获取文件夹下所有文件名
        data_list = [os.path.join(origin_path, each) for each in data_list]#补充为完整路径
        for each in data_list:
            self.process_original_data(each, save_files)#逐一进行处理


    def batch_iter(self,data,batch_size,allow_absence=True):
        '''
        批次生成器
        :param data: 数据
        :param batch_size: 批次大小
        :param allow_absence: 允许最后批次不达批次大小
        :return:
        '''
        data_size = len(data)

        if allow_absence:
            epoch_num = math.ceil(data_size / batch_size)
        else:
            epoch_num = math.floor(data_size / batch_size)

        for epoch in range(int(epoch_num)):
            start = epoch*batch_size
            end = min(start+batch_size,data_size)
            yield data[start:end]
