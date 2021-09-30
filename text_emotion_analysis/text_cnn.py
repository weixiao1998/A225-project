import tensorflow as tf

class TextCNN(object):
    def __init__(self,max_text_dim,vec_dim):

        # -------------------CNN参数-----------------

        pooled = []
        filter_sizes = [2,3,5,7]
        filters_num = 50

        fc1_num = 300
        fc2_num = 100

        # -------------------------------------------

        def weigh_variable(shape):
            '''
            生成随机变量
            :param shape:维度形状
            :return:该维度形状的随机变量
            '''
            init = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(init)

        def bias_variable(shape):
            '''
            生成偏差变量
            :param shape:维度形状
            :return:该维度形状的偏差变量 默认0.1
            '''
            init = tf.constant(0.1, shape=shape)
            return tf.Variable(init)

        def conv2d(x, W):
            '''
            卷积操作
            :param x: 输入数据
            :param W:过滤器权重
            :return:卷积结果
            '''
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")

        def max_pool(x, max_length,filter_size):
            '''
            池化操作
            :param x:输入数据
            :param max_length:句子最大长度
            :param filter_size:过滤器大小
            :return:池化结果
            '''
            return tf.nn.max_pool(x, ksize=[1, max_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding="VALID")

        #输入x
        self.x = tf.placeholder(tf.float32,[max_text_dim,vec_dim],name="input-x")
        #输入label
        self.y = tf.placeholder(tf.float32,[2],name="input-y")

        x = tf.reshape(self.x, [-1,  max_text_dim, vec_dim, 1])

        # 卷积和池化
        for filter_size in filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 对每个size的filter进行卷积
                W_conv1 = weigh_variable([filter_size, vec_dim, 1, filters_num])  # 2*100 1个平面  10个filters
                b_conv1 = bias_variable([filters_num])  # 每个filter一个bias
                h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
                h_pool1 = max_pool(h_conv1, max_text_dim, filter_size)
                pooled.append(h_pool1)

        # 池化结果拼接
        with tf.name_scope("pooled_concat"):
            num_filters_total = len(filter_sizes)*filters_num
            h_pool = tf.concat(pooled, 3)#pooled在第3个维度拼接
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # 添加全连接层，用于分类
        with tf.name_scope("full-connection-1"):
            W_fc1 = weigh_variable([num_filters_total, fc1_num])
            b_fc1 = bias_variable([fc1_num])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

        with tf.name_scope("full-connection-1-dropout"):
            self.keep_prob1 = tf.placeholder(tf.float32,name="keep_prob1")
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob1)

        with tf.name_scope("full-connection-2"):
            W_fc2 = weigh_variable([fc1_num, fc2_num])
            b_fc2 = bias_variable([fc2_num])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        with tf.name_scope("full-connection-2-dropout"):
            self.keep_prob2 = tf.placeholder(tf.float32,name="keep_prob2")
            h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob2)

        with tf.name_scope("full-connection-3"):
            W_fc3 = weigh_variable([fc2_num, 2])
            b_fc3 = bias_variable([2])

        # 得出输出
        with tf.name_scope("output"):
            self.scores = tf.add(tf.matmul(h_fc2_drop, W_fc3),b_fc3,name="scores")
            self.proportion = tf.nn.softmax(self.scores,name="proportion")

        # 定义损失
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.y)
            self.loss = tf.reduce_mean(losses,name="loss")

        # 定义准确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.scores,-1), tf.argmax(self.y, -1))#比较预测结果是否与标签一致
            self.right = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="right")

