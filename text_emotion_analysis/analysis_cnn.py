import tensorflow as tf
import os,time,random,math,pickle
from data_loader import DataLoader
from text_cnn import TextCNN
from tensorflow.contrib import learn

# -----------------------------------预处理参数---------------------------------------

MIN_WORD_LEN = 0    #数据的词数下限
MAX_WORD_LEN = 100  #数据的词数上限
VEC_DIM = 100   #词向量维度大小
MAX_TEXT_DIM = 100  #文本词数最大值
MAX_DATA_NUM = 35000 #允许的积极或消极数据大小
W2V_MODEL_PATH = "./vecs/vectors_100.bin"
STOP_WORDS_PATH = "stop_words.txt"

# -----------------------------------------------------------------------------------

loader = DataLoader(STOP_WORDS_PATH) #初始化数据输入工具

#数据读入
print("开始读入数据...")

neg_data, pos_data = loader.load_data("./data",["neg.txt","pos.txt"])
neg_size,pos_size = len(neg_data),len(pos_data)
print("【初始读入】 总数据量:{}  消极观点数据量：{}  积极观点数据量：{}".format(neg_size+pos_size,neg_size,pos_size))

print("开始分词...")
neg_data = loader.participle(neg_data)
pos_data = loader.participle(pos_data)
neg_size,pos_size = len(neg_data),len(pos_data)
print("【初次筛选】 总数据量:{}  消极观点数据量：{}  积极观点数据量：{}".format(neg_size+pos_size,neg_size,pos_size))

print("开始剔除过长过短的句子...")
neg_data = loader.check_word_limit(neg_data,min_word=MIN_WORD_LEN,max_word=MAX_WORD_LEN)
pos_data = loader.check_word_limit(pos_data,min_word=MIN_WORD_LEN,max_word=MAX_WORD_LEN)
neg_size,pos_size = len(neg_data),len(pos_data)
print("【二次筛选】 总数据量:{}  消极观点数据量：{}  积极观点数据量：{}".format(neg_size+pos_size,neg_size,pos_size))

print("开始平衡数据...")
neg_data,pos_data = loader.random_choose_data(neg_data,pos_data,max_num=MAX_DATA_NUM)
neg_size,pos_size = len(neg_data),len(pos_data)
print("【最终筛选】 总数据量:{}  消极观点数据量：{}  积极观点数据量：{}".format(neg_size+pos_size,neg_size,pos_size))


print("开始整合数据...")
all_data = neg_data+pos_data
all_label = [[1, 0]]*len(neg_data)+[[0, 1]]*len(pos_data)

max_text_length = max([len(each.split(" ")) for each in all_data])
print("文本最大词数为：{}".format(max_text_length))

print("开始载入词向量...")
processor = learn.preprocessing.VocabularyProcessor(MAX_TEXT_DIM,tokenizer_fn=loader.tokenizer)
processor.fit_transform(all_data)
words_vec = loader.get_words_vec(model_path=W2V_MODEL_PATH,vocab=processor.vocabulary_,vec_dim=VEC_DIM)
print("词向量字典大小：{}".format(len(words_vec)))

print("开始映射词向量...")
all_data = loader.load_x(all_data,words_vec=words_vec,max_text_dim=MAX_TEXT_DIM,vec_dim=VEC_DIM)

print("数据读入完成！")

print("打乱数据...")
temp = list(zip(all_data,all_label))
random.shuffle(temp)
all_data[:], all_label[:] = zip(*temp)

print("划分数据...")
all_size = neg_size+pos_size
part = math.ceil(all_size/10)
train_data,train_label = all_data[:-3*part],all_label[:-3*part]
test_data,test_label = all_data[-3*part:],all_label[-3*part:]
train_num = len(train_label)
test_num = len(test_label)

#统计数据量
train_neg_num = train_pos_num = test_neg_num = test_pos_num = 0
for label in train_label:
    if label == [1,0]:
        train_neg_num += 1
train_pos_num = train_num-train_neg_num
test_neg_num = neg_size-train_neg_num
test_pos_num = test_num-test_neg_num
print("训练集大小:{}   pos：{}  neg：{}\n"
      "测试集大小：{}   pos：{}  neg：{}"
      .format(train_num,train_pos_num,train_neg_num,test_num,test_pos_num,test_neg_num))


# ------------------------------------训练参数----------------------------------------

epoch_num = 10#训练次数
dev_per_epoch = 2#每多少次大训练验证一次验证集
batch_size = 500#每批次大小
train_keep_prob1 = 0.5
train_keep_prob2 = 0.5
test_keep_prob1 = 1.0
test_keep_prob2 = 1.0
loss_multiples = 1000#loss放大倍数

model_save_dir = os.path.abspath('.')+"/model"#模型保存文件夹路径

# -----------------------------------------------------------------------------------


def train():
    print("初始化训练模型及参数...")
    # 创建session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # 建立CNN模型
    cnn = TextCNN(max_text_dim=MAX_TEXT_DIM,vec_dim=VEC_DIM)

    # 定义优化器
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    gradient = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(gradient)

    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # tensorboard记录
    writer = tf.summary.FileWriter("./board",sess.graph)
    train_acc_summary = tf.Summary()
    train_loss_summary = tf.Summary()
    train_acc_summary.value.add(tag='train_accuracy', simple_value=0)
    train_loss_summary.value.add(tag='train_loss', simple_value=0)
    dev_acc_summary = tf.Summary()
    dev_loss_summary = tf.Summary()
    dev_acc_summary.value.add(tag='dev_accuracy', simple_value=0)
    dev_loss_summary.value.add(tag='dev_loss', simple_value=0)

    # 初始化模型保存
    os.makedirs(model_save_dir, exist_ok=True)
    saver = tf.train.Saver(max_to_keep=1)
    local_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    cur_model_save_dir = os.path.join(model_save_dir,local_time)+"/"
    os.makedirs(cur_model_save_dir, exist_ok=True)
    saver.save(sess,cur_model_save_dir)

    # 词向量保存
    with open(os.path.join(cur_model_save_dir,"vec.pkl"),"wb") as f:
        pickle.dump(words_vec,f,pickle.HIGHEST_PROTOCOL)

    tot_dev_steps = tot_train_steps = 0

    def train_step(x_batch,y_batch,train_NO):
        '''
        训练一个批次
        :param x_batch:
        :param y_batch:
        :param train_NO: 批次编号
        :return:
        :right 正确数
        :loss 损失
        '''
        nonlocal tot_train_steps

        right = loss = 0
        cur_batch_size = len(y_batch)
        for i in range(cur_batch_size):
            x = x_batch[i]
            y = y_batch[i]
            feed_dict = {
                cnn.x: x,
                cnn.y: y,
                cnn.keep_prob1: train_keep_prob1,
                cnn.keep_prob2: train_keep_prob2
            }
            train, step, each_right, each_loss = sess.run([train_op, global_step, cnn.right, cnn.loss],
                                                          feed_dict=feed_dict)
            right += each_right
            loss += each_loss

        acc = right/cur_batch_size
        loss = loss*loss_multiples/cur_batch_size
        info = "\tTRAIN batch#{}  acc={}  loss={}  right/total={}/{}" \
            .format(train_NO, acc, loss, int(right),cur_batch_size)#统计当前批次的信息
        print(info)

        #完成一次满的批次则进行记录
        if cur_batch_size==batch_size:
            train_acc_summary.value[0].simple_value = acc
            train_loss_summary.value[0].simple_value = loss
            writer.add_summary(train_acc_summary,tot_train_steps)
            writer.add_summary(train_loss_summary, tot_train_steps)
            tot_train_steps += 1

        return right,loss


    def dev_step(x_batch,y_batch,dev_NO):
        '''
        验证一个批次
        :param x_batch:
        :param y_batch:
        :param dev_NO: 批次编号
        :return:
        :right 正确数
        :loss 损失
        '''
        nonlocal tot_dev_steps

        right = loss = 0
        cur_batch_size = len(y_batch)
        for i in range(cur_batch_size):
            x = x_batch[i]
            y = y_batch[i]
            feed_dict = {
                cnn.x: x,
                cnn.y: y,
                cnn.keep_prob1: train_keep_prob1,
                cnn.keep_prob2: train_keep_prob2
            }
            step, each_right, each_loss = sess.run([global_step, cnn.right, cnn.loss],
                                                   feed_dict=feed_dict)
            right += each_right
            loss += each_loss

        acc = right / cur_batch_size
        loss = loss * loss_multiples / cur_batch_size
        info = "\tDEV batch#{}  acc={}  loss={}  right/total={}/{}" \
            .format(dev_NO, acc, loss, int(right),cur_batch_size)#统计当前批次的信息
        print(info)

        # 完成一次满的批次则进行记录
        if cur_batch_size == batch_size:
            dev_acc_summary.value[0].simple_value = acc
            dev_loss_summary.value[0].simple_value = loss
            writer.add_summary(dev_acc_summary, tot_dev_steps)
            writer.add_summary(dev_loss_summary, tot_dev_steps)
            tot_dev_steps += 1

        return right, loss


    # -------------------------------------------------------------------------------------------------

    print("开始运行CNN训练...")
    max_dev_acc = 0
    for epoch in range(1,epoch_num+1):#大循环

        #训练集分批次训练
        print("TRAIN EPOCH #{} START".format(epoch))
        train_NO = right = loss = 0
        train_batches = loader.batch_iter(list(zip(train_data, train_label)), batch_size)
        for batch in train_batches:
            x_batch,y_batch = zip(*batch)
            train_NO += 1
            each_right,each_loss = train_step(x_batch,y_batch,train_NO)    #分割出批次进行训练
            right+=each_right
            loss+=each_loss
        acc = right/train_num
        loss = loss/train_NO
        print("TRAIN EPOCH#{} END --- acc={}  loss={}\n".format(epoch,acc,loss))    #训练集总信息

        #每dev_per_epoch次大循环进行验证
        if(epoch%dev_per_epoch==0):
            print("DEV EPOCH #{} START".format(epoch//dev_per_epoch))
            dev_NO = right = loss = 0
            dev_batches = loader.batch_iter(list(zip(test_data,test_label)),batch_size)
            for batch in dev_batches:
                x_batch, y_batch = zip(*batch)
                dev_NO += 1
                each_right, each_loss = dev_step(x_batch, y_batch, dev_NO)   #分割出批次进行验证
                right += each_right
                loss += each_loss
            acc = right / test_num
            loss = loss / dev_NO
            print("DEV EPOCH#{} END --- acc={}  loss={}\n".format(epoch, acc, loss))    #验证集总信息

            if acc>max_dev_acc:#如果准确率有提升则保存模型
                saver.save(sess,cur_model_save_dir)
                max_dev_acc = acc
                print("模型已自动保存！")
    print("模型 最佳acc={}".format(max_dev_acc))

    # -------------------------------------------------------------------------------------------------


# 调用训练方法
train()