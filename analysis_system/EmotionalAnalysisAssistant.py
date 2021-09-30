import jieba,sys,requests,pickle,xlwt,datetime,chardet,os
import matplotlib.pyplot as plt
import tensorflow as tf
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel, QHBoxLayout, QVBoxLayout, QTextEdit
from PyQt5.QtWidgets import QGraphicsOpacityEffect, QTabWidget,QFileDialog,QMessageBox, QLineEdit,QSplashScreen
from PyQt5.QtGui import QIcon, QPixmap, QPalette, QBrush, QMouseEvent,QMovie
from PyQt5.QtCore import QPoint, pyqtSignal,QObject,Qt
from bs4 import BeautifulSoup
from ExcelDialogs import ExcelSaveAlertDialog,ExcelSaveAskDialog

# 用于设置系统任务栏的图标
import ctypes
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("main_window")

# 禁用GPU加速
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Signal(QObject):
    clicked = pyqtSignal()

class CloseLabel(QLabel):
    '''
    窗口关闭按钮
    '''
    def __init__(self):
        super().__init__()
        self.flag = False
        self.s = Signal()

    def mousePressEvent(self, QMouseEvent):
        self.flag = True

    def mouseReleaseEvent(self, QMouseEvent):
        if self.flag is True:
            self.s.clicked.emit()
            self.flag = False

    def mouseMoveEvent(self, QMouseEvent):
        self.flag = False

    def enterEvent(self, *args, **kwargs):
        self.setStyleSheet("QLabel{ border-image: url(./images/close(2).png); background-color: rgb(232, 17, 35); }")

    def leaveEvent(self, *args, **kwargs):
        self.setStyleSheet("QLabel{ border-image: url(./images/close.png) }")


class MinLabel(QLabel):
    '''
    窗口最小化按钮
    '''
    def __init__(self):
        super().__init__()
        self.flag = False
        self.s = Signal()

    def mousePressEvent(self, QMouseEvent):
        self.flag = True

    def mouseReleaseEvent(self, QMouseEvent):
        if self.flag is True:
            self.s.clicked.emit()
            self.flag = False

    def mouseMoveEvent(self, QMouseEvent):
        self.flag = False

    def enterEvent(self, *args, **kwargs):
        self.setStyleSheet("QLabel{ border-image: url(./images/min(2).png); background-color: rgb(225, 225, 225); }")

    def leaveEvent(self, *args, **kwargs):
        self.setStyleSheet("QLabel{ border-image: url(./images/min.png) }")


class MainWindow(QWidget):

    def __init__(self):
        '''
        初始化
        '''
        super().__init__()

        self.init_var()
        self.init_excel_dir()
        self.init_UI()
        self.load_data()

    def init_var(self):
        '''
        初始化系统各种变量、常量
        :return:
        '''
        self.title = "评论分析助手"#标题
        self.version = "V1.0"#版本号
        self.stop_words_path = "./vec/stop_words.txt"#停用词路径
        self.model_dir = "./model"#TensorFlow模型路径
        self.vec_path = './vec/vec.pkl'#词向量路径
        self.result_dir = "./excel_file"#excel储存路径

        self.words_vec = None#词向量
        self.stop_words = None#停用词

        self._startPos = None
        self._endPos = None
        self._isTracking = False

        self.url_edit = None#url输入框
        self.analysis_edit = None#文本输入框
        self.show_txt_edit = None#文件文本展示框
        self.file_path_edit = None#文件路径输入框

    def init_UI(self):
        '''
        初始化系统UI
        :return:
        '''
        self.setMinimumSize(1380, 900)
        self.setMaximumSize(1380, 900)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint)  # 无边框
        self.setContentsMargins(0, 0, 0, 0)
        self.setWindowIcon(QIcon("images/icon.jpg"))                # 设置图标
        self.setWindowTitle(self.title) #设置窗口标题

        # 设置背景图片
        palette1 = QPalette()
        palette1.setBrush(self.backgroundRole(), QBrush(QPixmap("images/bg.jpg")))
        self.setPalette(palette1)

        # 关闭按钮
        btn1 = CloseLabel()
        btn1.setMinimumSize(30, 30)
        btn1.setMaximumSize(30, 30)
        btn1.s.clicked.connect(self.close)
        btn1.setToolTip("关闭窗口")
        btn1.setAttribute(True)
        btn1.setStyleSheet("QLabel{ border-image: url(./images/close.png) }")

        # 顶部的布局
        self.title_box = QHBoxLayout()
        self.title_box.setContentsMargins(0, 0, 0, 0)

        self.title_picture = QLabel()
        self.title_picture.setMaximumSize(25, 25)
        self.title_picture.setMinimumSize(25, 25)
        self.title_picture.setScaledContents(True)
        self.title_picture.setPixmap(QPixmap("./images/icon.jpg"))

        self.title_button = QLabel()
        self.title_button.setText(self.title+" -"+self.version)

        # 最小化按钮
        self.min_button = MinLabel()
        self.min_button.setMaximumSize(30, 30)
        self.min_button.setMinimumSize(30, 30)
        self.min_button.setToolTip("最小化")
        self.min_button.setMouseTracking(True)
        self.min_button.s.clicked.connect(self.showMinimized)
        self.min_button.setStyleSheet("QLabel{ border-image: url(./images/min.png);}")

        # 页面布局
        self.title_box.addStretch(1)
        self.title_box.addWidget(self.title_picture)
        self.title_box.addSpacing(2)
        self.title_box.addWidget(self.title_button)
        self.title_box.addStretch(1)
        self.title_box.addWidget(self.min_button)
        self.title_box.addSpacing(5)
        self.title_box.addWidget(btn1)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 5, 0, 0)
        vbox.addLayout(self.title_box)
        vbox.addStretch(1)

        # 输入的选择框QTabWidget
        self.tabs = QTabWidget()
        self.tabs.setMaximumSize(1000, 600)
        self.tabs.setMinimumSize(1000, 600)

        # 设置QTabWidget的透明度
        op = QGraphicsOpacityEffect()
        op.setOpacity(0.5)
        self.tabs.setGraphicsEffect(op)
        self.tabs.setAutoFillBackground(True)

        # 设置多一个tab0用于居中显示，让tab0的TabBar透明
        self.tab0 = QWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()

        # 样式定义
        self.tabs.setStyleSheet("QTabBar::tab { border:2px groove gray;border-radius:10px;"
                                "padding:7px 7px; height:35px; width:200px; margin:2px;}"
                                "QTabBar::tab:hover{ background-color: Transparent; border:2px groove black; border-radius:10px;}"
                                "QTabBar::tab:selected{ background-color:#b3b3cc; border:2px groove Transparent; border-radius:10px;}")

        # 添加tab页
        self.tabs.addTab(self.tab0, "使用说明")
        self.tabs.addTab(self.tab1, "文本框/网页抓取输入")
        self.tabs.addTab(self.tab2, "文本文件输入")

        # 初始化各个tab页
        self.SetTabUI_0()
        self.SetTabUI_1()
        self.SetTabUI_2()

        # 设置默认的选项卡为简介
        self.tabs.setCurrentIndex(0)

        # 分析按钮
        self.submit = QPushButton(self)
        self.submit.setStyleSheet("QPushButton{ border:2px;border-radius:10px;background-color:white; }"
                                  "QPushButton:hover{ background-color:gray; }")
        self.submit.setMinimumSize(100, 35)
        self.submit.setMaximumSize(100, 35)
        self.submit.setText("分析")
        self.submit.clicked.connect(self.SentMessage)
        op1 = QGraphicsOpacityEffect()
        op1.setOpacity(0.5)
        self.submit.setGraphicsEffect(op1)
        self.submit.setAutoFillBackground(True)

        # 页面布局
        tab_box = QHBoxLayout()
        tab_box.addStretch(1)
        tab_box.addWidget(self.tabs)
        tab_box.addStretch(1)

        submit_box = QHBoxLayout()
        submit_box.addStretch(1)
        submit_box.addWidget(self.submit)
        submit_box.addStretch(1)

        vbox.addLayout(tab_box)
        vbox.addLayout(submit_box)
        vbox.addSpacing(10)
        vbox.addStretch(1)

        self.setLayout(vbox)
        self.setAcceptDrops(True)


    def load_data(self):
        '''
        读入系统所需要的数据
        :return:
        '''

        #载入分析模型
        try:
            # 获取checkpoint
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            input_checkpoint = checkpoint.model_checkpoint_path

            # tf载入模型
            with tf.device('/cpu:0'):
                self.sess = tf.Session()
                path = os.path.join(self.model_dir,".meta")
                saver = tf.train.import_meta_graph(path)
                saver.restore(self.sess, input_checkpoint)
                self.graph = tf.get_default_graph()
        except:
            QMessageBox.warning(self, "错误", "加载分析模型失败！")
            exit(-1)

        # 读入停用词
        self.stop_words = self.get_stopwords()

        # 读入预训练的词向量
        try:
            with open(self.vec_path, 'rb') as f:
                self.words_vec = pickle.load(f)
        except:
            QMessageBox.warning(self, "错误", "词向量读取失败！")
            exit(-1)


    def init_excel_dir(self):
        '''
        初始化excel保存文件夹
        若不存在则新建
        :return:
        '''
        try:
            if os.path.exists(self.result_dir) is False:
                os.mkdir(self.result_dir)
        except:
            QMessageBox.warning(self, "错误", "结果储存文件夹创建失败！")


    def read_file(self,file_path):
        '''
        读入文件内容函数
        能自动识别文件编码，有效读入GBK和UTF-8编码的文件
        :param file_path: 文件路径
        :return: 文件内容
        '''
        try:
            with open(file_path,"rb") as f:
                file_data = f.read()
                # 检测文件编码
                result = chardet.detect(file_data)
                # 指定检测到的编码进行解码
                if result['encoding']!=None: #如果编码可以检测到，则解码
                    file_content = file_data.decode(encoding=result['encoding'])
                    return file_content
                else:#没有检测到编码（一般为空文本情况），返回空字符串
                    return ""
        except:
            # 文件打开出错
            QMessageBox.warning(self, "错误", "文件读取失败！")


    def SetTabUI_0(self):
        '''
        初始化tab0的UI
        :return:
        '''
        # 使用说明
        info = QLabel("评论分析助手<br><br>"
                      "版本号：<br>"
                      + self.version + "<br><br>"
                                       "简介：<br>"
                                       "这是一个用于分析用户评论情感的系统<br><br>"
                                       "使用方法：<br>"
                                       "1. 将文本输入到文本框，若是多条文本则用空行隔开<br>"
                                       "2. 输入URL，系统自动抓取网页内容<br>"
                                       "3. 将文件拖入系统中，系统将自动读取文本内容，多条文本需要用空行隔开")

        # 页面布局
        vbox = QVBoxLayout()
        vbox.addWidget(info)
        self.tab0.setLayout(vbox)

    def SetTabUI_1(self):
        '''
        初始化tab1的UI
        :return:
        '''
        #页面布局
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()

        # url输入框
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("网页URL（目前仅能读取p标签中的内容）")
        # url抓取按钮
        self.catch = QPushButton("抓取URL内容")
        self.catch.clicked.connect(self.deal_url)
        # 分析文本框
        self.analysis_edit = QTextEdit()
        self.analysis_edit.setPlaceholderText("输入需要分析的评论，多条评论用空行隔开")

        # 添加widget
        hbox.addWidget(self.url_edit)
        hbox.addWidget(self.catch)
        vbox.addLayout(hbox,stretch=1)
        vbox.addWidget(self.analysis_edit,stretch=5)

        self.tab1.setLayout(vbox)

    def SetTabUI_2(self):
        '''
        初始化tab2的UI
        :return:
        '''
        # 页面布局
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()

        # 文件路径输入框
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("请选择需要分析的文本文件，仅限txt文件")

        # 选择文件按钮
        self.open_file = QPushButton("选择文件")
        self.open_file.clicked.connect(self.open_txt)

        # 文件文本展示框
        self.show_txt_edit = QTextEdit()
        self.show_txt_edit.setEnabled(False)#不可编辑
        self.show_txt_edit.setStyleSheet("QTextEdit{ background-color:white; }")
        self.show_txt_edit.setPlaceholderText("可拖动文件进入此文本域，仅限txt文件。\n分析多条文本需要用空行隔开！")

        # 添加widget
        hbox.addWidget(self.file_path_edit)
        hbox.addWidget(self.open_file)
        vbox.addLayout(hbox, stretch=1)
        vbox.addWidget(self.show_txt_edit, stretch=5)

        self.tab2.setLayout(vbox)


    def dragEnterEvent(self, evn):
        '''
        重写鼠标拖入事件
        用于处理文件拖入
        :param evn: 事件
        :return:
        '''
        if self.tabs.currentIndex() != 2:#如果不是文件输入页面将不做处理
            evn.ignore()
            return
        else:
            evn.accept()
            # 清空文本展示框
            self.show_txt_edit.clear()
            # 获取文件路径
            file_path = evn.mimeData().text()
            file_path = file_path[8:]
            # 显示文件路径
            self.file_path_edit.setText(file_path)
            # 读入文件内容
            file_content = self.read_file(file_path)
            # 将文件内容展示
            self.show_txt_edit.setText(file_content)

    def mouseMoveEvent(self, e: QMouseEvent):
        '''
        重写鼠标移动事件
        :param e:移动事件
        :return:
        '''
        self._endPos = e.pos() - self._startPos
        self.move(self.pos() + self._endPos)
        e.ignore()

    def mousePressEvent(self, e: QMouseEvent):
        '''
        重写鼠标按压事件
        :param e:
        :return:
        '''
        if e.button() == Qt.LeftButton:
            self._isTracking = True
            self._startPos = QPoint(e.x(), e.y())

    def mouseReleaseEvent(self, e: QMouseEvent):
        '''
        重写鼠标释放事件
        :param e:
        :return:
        '''
        if e.button() == Qt.LeftButton:
            self._isTracking = False
            self._startPos = None
            self._endPos = None

    def SentMessage(self):
        '''
        处理各页面的分析事件
        :return:
        '''
        index = self.tabs.currentIndex()
        if index == 1:#文本框输入方式
            self.deal_analysis(self.analysis_edit)
        elif index == 2:#文件输入方式
            self.deal_analysis(self.show_txt_edit)

    def show_detail(self, proportion):
        '''
        用图形展示单文本结果
        :param proportion: cnn的输出，维度为1x2
        :return:
        '''

        # 字符串定义，由于plt对中文支持不够友善，此处使用英文表达
        stra = "Negative"
        strb = "Positive"
        str = "Analysis result -- "

        # 得出结果，proportion[0]为消极的预测概率
        if proportion[0] > proportion[1]:
            str += stra
        else:
            str += strb

        # 饼图展示
        labels = 'Negative', 'Positive'
        explode = (0, 0.1)
        fig1, ax1 = plt.subplots(num="Analysis Result")
        ax1.pie(proportion, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        ax1.set_title(str, fontsize=12, color="red")

        # 显示绘图
        plt.show()

    def show_details(self, proportion):
        '''
        展示多文本结果
        :param proportion: cnn的输出，维度为1x2
        :return:
        '''
        # 统计消极与积极的数量
        nums = [0,0]
        for output in proportion:
            if output[0]>output[1]:
                nums[0] += 1
            else:
                nums[1] += 1

        # plt展示
        fig, ax = plt.subplots(1,2,num="Analysis Result")#1x2子图

        # 子图1展示饼状图
        ax1 = ax[0]
        labels = "Negative", 'Positive'
        explode = (0, 0.1)
        ax1.pie(nums, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        ax1.set_title("Proportion", fontsize=12, color="red")

        # 子图2展示条状图
        ax2 = ax[1]
        ax2.bar(labels,nums,width=0.8,lw=1)
        ax2.set_title("Statistics", fontsize=12, color="red")

        # 绘图展示
        plt.subplots_adjust(wspace=1)
        plt.show()

    def get_stopwords(self):
        '''
        获取停用词列表
        :return: 停用词列表
        '''
        try:
            stopwords = [] #停用词列表
            # 按行读入停用词
            with open(self.stop_words_path, "r", encoding='gbk') as f:
                lines = f.readlines()
                for line in lines:
                    stopwords.append(line.strip("\n"))#去除回车
            return stopwords
        except:
            QMessageBox.warning(self, "错误", "停用词读取失败！")
            exit(-1)

    def analyze_text(self,sentence):
        '''
        分析一个句子
        :param sentence: 一个句子
        :return: 分析结果 cnn 1*2输出
        '''
        stop_words = self.stop_words
        words_vec = self.words_vec

        def get_seg_list(sentence):
            '''
            获取分词后生成的列表
            :param sentence: 字符串
            :return: 分词列表
            '''
            first_seg_list = list(jieba.cut(sentence))#使用结巴分词
            seg_list = []#分词列表

            # 去除在停用词中的词
            for each in first_seg_list:
                if each not in stop_words:
                    seg_list.append(each)

            return seg_list

        def text_to_vec(sentence):
            '''
            将文本转为词向量矩阵
            :param sentence: 文本
            :return:
            '''
            MAX_TEXT_DIM = 100  # 词数最大值
            VEC_DIM = 100 # 词向量维度
            cnt = 0  # 词向量计数
            seg_list = get_seg_list(sentence)#获取分词列表
            each_x = []  # 一个文本的词向量矩阵

            # 从预训练的词向量中得到每个词的词向量（若超过大小则不添加），并将其加入到词向量矩阵中（若找不到则不添加）。
            for word in seg_list:
                try:
                    if cnt < MAX_TEXT_DIM:
                        each_x.append(words_vec[word].tolist())
                        cnt += 1
                except:
                    pass
            LIST_ZERO = [0]*VEC_DIM # VEC_DIM维零向量
            # 若不足@MAX_TEXT_DIM行，添加@LIST_ZERO补全
            for cnt in range(MAX_TEXT_DIM - len(each_x)):
                each_x.append(LIST_ZERO)

            return each_x

        # 词向量映射
        each_x = text_to_vec(sentence)

        '''
        获取模型中的tensor
        '''
        with tf.device('/cpu:0'):#指定使用cpu进行计算
            # 输入
            x = self.graph.get_tensor_by_name("input-x:0")
            # 预测
            proportion = self.graph.get_tensor_by_name("output/proportion:0")
            # dropout保留值
            keep_prob1 = self.graph.get_tensor_by_name("full-connection-1-dropout/keep_prob1:0")
            # dropout保留值
            keep_prob2 = self.graph.get_tensor_by_name("full-connection-2-dropout/keep_prob2:0")
            # 运行TensorFlow模型
            out = self.sess.run(proportion, feed_dict={x: each_x, keep_prob1: 1.0, keep_prob2: 1.0})

        return out[0]

    def get_edit_texts(self,edit):
        '''
        获取一个edit的文本列表
        :param edit: QTextEdit
        :return:文本列表
        '''

        # 获取文本框输入
        input = edit.toPlainText()
        # 以空行为标准分割输入
        temp_list = input.split("\n\n")

        texts = []#文本列表
        for each in temp_list:
            each = each.strip("\n").strip()#去除回车和多余空格
            if len(each)>0:#如果是有效文本，则添加进文本列表
                texts.append(each)

        return texts

    def deal_url(self):
        '''
        处理url抓取，将信息展示到分析文本框
        :return:
        '''
        try:
            # 获取url
            url = self.url_edit.text()
            # 模拟请求
            r = requests.get(url)
            # 内容解码
            try:
                content = r.content.decode(r.apparent_encoding)
            except:
                content = r.text
                QMessageBox.warning(self, "错误", "默认网页解码失败，内容可能有误！")
            # 使用bs4进行网页数据处理
            soup = BeautifulSoup(content, "html.parser")
            # 找到所有p标签内的内容
            found = soup.find_all("p", text=True)
            # 处理爬取结果
            if len(found) > 0:  # 如果找到p标签
                # 清空输入框
                self.analysis_edit.setText("")
                # 去除外层p标签，并添加内容进文本框
                for each in found:
                    soup = BeautifulSoup(str(each), "html.parser")
                    temp_str = self.analysis_edit.toPlainText()
                    self.analysis_edit.setText(temp_str + "\n\n" + soup.get_text())
            else:
                QMessageBox.information(self, "提示", "找不有效内容！")
        except:
            # 爬取出错
            QMessageBox.warning(self, "错误", "爬取失败！")



    def deal_analysis(self,edit):
        '''
        分析文本框中的内容
        :param edit: 文本框
        :return:
        '''

        # 获取文本框内容，存进文本列表
        texts = self.get_edit_texts(edit)
        # 文本数据条数
        text_num = len(texts)

        try:
            # 按照数据条数进行不同处理
            if text_num == 0:
                QMessageBox.information(self,"提示","没有有效文本！请确认输入内容！")
                return
            elif text_num == 1:

                    output = self.analyze_text(texts[0])
                    self.show_detail(output)  # 直接展示结果
            else:
                outputs = []
                for text in texts:
                    outputs.append(self.analyze_text(text))
                self.show_details(outputs)  # 展示总体结果
                ok = self.ask_excel_dialog() #询问是否保存结果文件
                if ok: # 如果选择保存文件
                    save_ok,file_path = self.create_excel(texts, outputs) # 创建excel
                    if save_ok:
                        self.alert_excel_dialog(excel_path=file_path) # 提示保存成功
        except:
            QMessageBox.warning(self, "错误", "分析失败！")


    def open_txt(self):
        '''
        选择打开一个文件，将文件内容读入文本框
        :return:
        '''
        #获取文件路径
        file_path,file_type = QFileDialog.getOpenFileName(self, '选择文件', '', '(*.txt)')
        #若没有选择直接退出
        if file_path=="":
            return
        #显示文件路径
        self.file_path_edit.setText(file_path)
        #读取文件内容
        file_content = self.read_file(file_path)
        #展示文件文本信息
        self.show_txt_edit.setText(file_content)


    def create_excel(self, texts, outputs):
        '''
        创建一个分析结果excel
        :param texts:文本列表
        :param outputs:分析结果列表
        :return:
        '''
        try:
            #获取系统时间
            now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            #创建excel
            file = xlwt.Workbook(encoding="UTF-8")
            sheet = file.add_sheet('分析结果')
            sheet.col(0).width = 256 * 150
            # 列名
            header = [u'文本', u'消极因子', u'积极因子', u'判断结果']
            for i in range(4):
                sheet.write(0, i, header[i])
            # 按行写入分析结果
            for row in range(1,len(texts)+1):
                now_row = row-1
                sheet.write(row, 0, texts[now_row])
                sheet.write(row, 1, float(outputs[now_row][0]))
                sheet.write(row, 2, float(outputs[now_row][1]))
                if outputs[now_row][0] > outputs[now_row][1]:
                    sheet.write(row, 3, '消极')
                else:
                    sheet.write(row, 3, '积极')
            # 生成文件保存路径
            now_time = now_time.replace(':', '.')
            file_name = "分析结果"+now_time+".xls"
            file_path = os.path.join(self.result_dir,file_name)
            file_path = os.path.abspath(file_path)
            # 保存文件
            file.save(file_path)
            return True,file_path
        except:
            QMessageBox.warning(self, "错误", "创建excel文件失败！")
            return False, ""


    def ask_excel_dialog(self):
        '''
        弹出是否保存分析结果excel对话框
        :return: 返回值为 True 则是 保存， 返回 False 则不保存
        '''
        ask_dialog = ExcelSaveAskDialog()#询问是否保存对话框
        ask_dialog.exec_()
        return ask_dialog.save_flag

    def alert_excel_dialog(self,excel_path):
        '''
        弹出保存成功文本框
        :param excel_path: excel保存的路径
        :return:
        '''
        b = ExcelSaveAlertDialog(excel_path=excel_path)#保存成功对话框
        b.exec_()

class LoadingWindow(QWidget):
    '''
    加载动画窗口
    '''
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # 窗口大小
        self.setMinimumSize(400, 300)
        self.setMaximumSize(400, 300)

        # 无窗口按钮
        self.setWindowFlags(Qt.FramelessWindowHint)

        # 动画
        self.label = QLabel()
        self.label.setScaledContents(True)
        self.label.setVisible(True)
        self.movie = QMovie("./images/loading.gif")
        self.label.setMovie(self.movie)
        self.movie.start()

        # 加载信息
        self.info = QLabel()
        self.info.setText("正在启动评论分析助手.....")

        # 页面布局
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.info)

        self.setLayout(vbox)

        self.show()



# 主程序入口
if __name__ == '__main__':
    # 启动程序
    app = QApplication(sys.argv)
    # 过渡动画
    loading = LoadingWindow()
    app.processEvents()
    # 主窗口
    main_window = MainWindow()
    # 关闭加载动画，显示主窗口
    loading.close()
    main_window.show()
    # 运行直至退出
    sys.exit(app.exec_())
