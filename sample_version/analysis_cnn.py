import tensorflow as tf
import gensim.models
import jieba
import re
import os
from text_cnn import TextCNN

#载入词向量
model = gensim.models.KeyedVectors.load_word2vec_format('vectors_100.bin',binary=True)
words_vec = model.wv
del model

def check_limit(content,score):
    limit_len = 100 #词最大值
    c = []
    s = []
    for i in range(len(score)):
        if(len(content[i]) <= limit_len):
            c.append(content[i])
            s.append(score[i])
    return c,s

def get_seg_list(sentence):
    '''
    获取分词后生成的列表
    :param sentence: 字符串
    :return: 分词列表
    '''
    sentence = re.sub(r"\s{2,}", " ", sentence)  # 去多余空格
    sentence = re.sub('[，。的了]+', '', sentence)
    return jieba.cut(sentence)


def load_data(file_name):
    content = []
    score = []
    pos = neg = 0
    # count = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    with open(file_name, "r", encoding="gbk") as f:
        for line in f:
            if len(line) == 1:
                continue
            if line.find(':') != -1:
                temp = line.split(":")
                if (temp[0] == "content"):
                    content.append(temp[1])
                elif(temp[0] == "score"):
                    # score.append(int(temp[1][0]))
                    # count[temp[1][0]] += 1
                    if temp[1][0] == '1':
                        neg += 1
                        score.append(0)
                    elif temp[1][0] == '5':
                        pos += 1
                        score.append(1)
                else:
                    content[len(content) - 1] = content[len(content) - 1] + line
            else:
                content[len(content)-1] = content[len(content)-1]+line
    print("num:"+str(len(score))+" pos:"+str(pos)+" neg:"+str(neg))
    return content, score

def load_xy(content, score):
    if(len(content)!=len(score)):#评论和分数个数不匹配
        print("数据有误",len(content),len(score))
        quit(-1)

    list_zero = [[0] * 100][0]

    x = []
    y = []

    for i in range(len(score)):
        sentence = content[i]
        seg_list = get_seg_list(sentence)
        each_x = []
        for word in seg_list:
            try:
                each_x.append(words_vec[word].tolist())
            except:
                pass

        # each_y = [0, 0, 0, 0, 0]
        # each_y[int(score[i] - 1)] = 1
        each_y = [0, 0]
        each_y[int(score[i])] = 1

        for cnt in range(max_length - len(each_x)):
            each_x.append(list_zero)

        x.append(each_x)
        y.append(each_y)
    return x, y



train_content, train_score = load_data("data.txt")
train_content,train_score = check_limit(train_content,train_score)
max_length = max(len(x) for x in train_content)
print("train-max-length:"+ str(max_length))
train_x, train_y = load_xy(train_content, train_score)


test_content, test_score = load_data("test.txt")
test_content, test_score = check_limit(test_content, test_score)
test_x, test_y = load_xy(test_content, test_score)


with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        cnn = TextCNN(max_length=max_length)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cnn.loss)
        # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cnn.loss)
        sess.run(tf.global_variables_initializer())
        times = 21
        for j in range(times):
            right = loss = 0
            if j%10==0:
                for i in range(len(test_score)):
                    x_batch = test_x[i]
                    y_batch = test_y[i]
                    feed_dict = {
                        cnn.x: x_batch,
                        cnn.y: y_batch,
                        cnn.keep_prob: 1.0
                    }
                    each_acc, each_Loss, each_predict = sess.run([cnn.accuracy,cnn.loss,cnn.predictions],feed_dict=feed_dict)
                    right += each_acc
                    loss += each_Loss
                    # print("#" + str(i + 1) + " acc= " + str(each_acc) + " predictions= " + str(each_predict))
                print("DEV acc="+str(right/len(test_score))+" loss="+str(loss)+" right="+str(right))
            right = loss = 0
            for i in range(len(train_score)):
                x_batch = train_x[i]
                y_batch = train_y[i]
                feed_dict = {
                    cnn.x: x_batch,
                    cnn.y: y_batch,
                    cnn.keep_prob: 0.3
                }
                step,each_acc, each_Loss, each_predict = sess.run([train_step,cnn.accuracy, cnn.loss, cnn.predictions],
                                                             feed_dict=feed_dict)
                right += each_acc
                loss += each_Loss
            print("TRAIN acc=" + str(right / len(train_score)) + " loss=" + str(loss) + " right=" + str(right))
        save_dir = os.path.abspath('.')+"/model"
        os.makedirs(save_dir, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, save_dir+"/")

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(21):
#         for batch in range(n_batch):
#             batch_xs,batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
#         acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
#         print("#"+str(epoch)+" acc= "+str(acc))








