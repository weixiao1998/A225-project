import tensorflow as tf

class TextCNN(object):
    def __init__(self,max_length):
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
        self.x = tf.placeholder(tf.float32,[max_length,100],name="input-x")
        #输入label
        self.y = tf.placeholder(tf.float32,[2],name="input-y")

        x = tf.reshape(self.x, [-1,  max_length, 100, 1])

        pooled = []
        for filter_size in range(2,5):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                W_conv1 = weigh_variable([filter_size, 100, 1, 10])  # 2*100 1个平面  10个fliters
                b_conv1 = bias_variable([10])  # 每个filter一个bias
                h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
                h_pool1 = max_pool(h_conv1, max_length, filter_size)
                pooled.append(h_pool1)
        num_filters_total = 30
        h_pool = tf.concat(pooled, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # 添加全连接层，用于分类
        with tf.name_scope("full-connection"):
            W_fc1 = weigh_variable([num_filters_total, 500])
            b_fc1 = bias_variable([500])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
            self.keep_prob = tf.placeholder(tf.float32,name="keep_prob")
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
            W_fc2 = weigh_variable([500, 2])
            b_fc2 = bias_variable([2])

        with tf.name_scope("output"):
            scores = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            self.predictions = tf.argmax(scores,1,name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=self.y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 0))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

