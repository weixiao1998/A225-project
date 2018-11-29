import tensorflow as tf
import jieba,re,os,sys,threading
from gensim.models.keyedvectors import KeyedVectors
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QAction, qApp, QTextEdit, QGridLayout, \
    QPushButton


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #多线程导入训练好的词向量和模型
        t1 = threading.Thread(target=self.loadData)
        t1.start()
        #初始化ui
        self.initUI()

    def get_seg_list(self,sentence):
        '''
        获取分词后生成的列表
        :param sentence: 字符串
        :return: 分词列表
        '''
        sentence = re.sub(r"\s{2,}", " ", sentence)  # 去多余空格
        sentence = re.sub('[，。的了]+', '', sentence)
        return jieba.cut(sentence)

    #载入词向量及模型
    def loadData(self):
        # 模型文件夹
        save_dir = os.path.abspath('.') + "/model"

        # 获取checkpoint
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        input_checkpoint = checkpoint.model_checkpoint_path

        # 载入词向量
        model = KeyedVectors.load_word2vec_format(os.path.abspath('.') + '/vec/vectors_100.bin', binary=True)
        self.words_vec = model.wv
        del model

        # tf载入模型
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta')
        saver.restore(self.sess, input_checkpoint)
        self.graph = tf.get_default_graph()

        # 状态栏显示
        self.statusBar().showMessage('Ready...分析系统准备就绪')

    # 初始化UI
    def initUI(self):
        self.initMenu()#初始化菜单
        # 界面ui设置
        self.setGeometry(300, 300, 550, 400)
        self.setWindowTitle('文本情感分析系统')
        self.statusBar().showMessage('Ready...')

        # 中心widget
        self.main_ground = QWidget()  # 组件要摆在上面
        self.setCentralWidget(self.main_ground)

        # 开始分析按钮
        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.clicked.connect(self.analyzeText)
        # 用户输入框
        self.textEdit = QTextEdit()
        self.textEdit.setPlaceholderText("请输入待分析文本...")

        # 设置grid layout的组件摆放
        grid = QGridLayout()
        grid.setSpacing(20)
        grid.addWidget(self.textEdit, 1, 1, 1, 3)
        grid.addWidget(self.analyze_btn, 2, 2)

        self.main_ground.setLayout(grid)

        self.show()

    # 关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '提示', "确定要退出吗？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # 初始化菜单
    def initMenu(self):
        # 退出action
        exitAction = QAction('退出', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('退出程序')
        exitAction.triggered.connect(qApp.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('程序')
        fileMenu.addAction(exitAction)

    # 显示分析结果
    def show_analysis(self,y):
        QMessageBox.information(self, "结果", "情绪为："+("积极" if y==1 else "消极") ,QMessageBox.Yes)

    # 分析文本
    def analyzeText(self):
        str = self.textEdit.toPlainText()
        str = re.sub(r"\s{2,}", " ", str)  # 去多余空格
        str = re.sub('[，。的了]+', '', str) # 去停用词
        seg_list = jieba.cut(str) # 分词后生成分词列表
        each_x = [] #一个文本的词向量矩阵
        # 从预训练的词向量中得到每个词的词向量，并将其加入到词向量矩阵中（若找不到则不添加）
        for word in seg_list:
            try:
                each_x.append(self.words_vec[word].tolist())
            except:
                pass

        list_zero = [[0] * 100][0]# 1行100列全为0的向量
        max_length = 100 # 词数最大值
        # 若不足@max_length行，添加@list_zero补全
        for cnt in range(max_length - len(each_x)):
            each_x.append(list_zero)

        # 获取模型中的tensor
        x = self.graph.get_tensor_by_name("input-x:0")#输入
        pre = self.graph.get_tensor_by_name("output/predictions:0")#预测
        keep_prob = self.graph.get_tensor_by_name("full-connection/keep_prob:0")#dropout保留值
        out = self.sess.run(pre, feed_dict={x: each_x, keep_prob: 1.0})#运行TensorFlow模型
        self.show_analysis(out[0])#展示结果


app = QApplication(sys.argv)
ex = MainWindow()
sys.exit(app.exec_())






