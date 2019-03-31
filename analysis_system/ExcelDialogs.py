from PyQt5.QtWidgets import QPushButton, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import  QPixmap, QFont
from PyQt5.QtCore import Qt

class ExcelSaveAskDialog(QDialog):

    def __init__(self):
        '''
        询问是否保存excel对话框 初始化
        '''
        super().__init__()
        self.init_ui()
        self.save_flag = False#保存标志，若为False不保存


    def init_ui(self):
        # 设置窗口大小
        self.setMaximumSize(600, 400)
        self.setMinimumSize(600, 400)

        # 设置窗口样式
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框
        self.setWindowOpacity(0.9)
        self.setStyleSheet("QWidget{ background-color:white;border-radius:30px; }")

        #标题
        title = QLabel()
        title.setText("是否保存分析结果excel表格")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("幼圆", 15, 0))

        # excel图标
        picture = QLabel()
        picture.setMinimumSize(100, 100)
        picture.setMaximumSize(100, 100)
        pix = QPixmap("./images/excel.png")  # 按指定路径找到图片
        picture.setPixmap(pix)  # 在label上显示图片
        picture.setScaledContents(True)  # 让图片自适应label大小

        #保存按钮
        save = QPushButton("保存")
        save.setMinimumSize(150, 50)
        save.setMaximumSize(150, 50)
        save.setStyleSheet("QPushButton{ border:2px;border-radius:10px;background-color:rgb(237, 246, 253) }"
                          "QPushButton:hover{ background-color:grey; }")
        save.clicked.connect(self.yes_save)

        #退出按钮
        quit = QPushButton("不保存")
        quit.setMinimumSize(150, 50)
        quit.setMaximumSize(150, 50)
        quit.setStyleSheet("QPushButton{ border:2px;border-radius:10px;background-color:rgb(237, 246, 253) }"
                           "QPushButton:hover{ background-color:grey; }")
        quit.clicked.connect(self.not_save)


        #页面布局
        vbox = QVBoxLayout()

        hbox1 = QHBoxLayout()
        hbox1.addStretch(1)
        hbox1.addWidget(title)
        hbox1.addStretch(1)

        hbox2 = QHBoxLayout()
        hbox2.addStretch(1)
        hbox2.addWidget(picture)
        hbox2.addStretch(1)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(save)
        hbox3.addStretch(1)
        hbox3.addWidget(quit)

        vbox.addLayout(hbox1)
        vbox.addStretch(1)
        vbox.addLayout(hbox2)
        vbox.addStretch(1)
        vbox.addLayout(hbox3)

        self.setLayout(vbox)


    def yes_save(self):
        '''
        保存excel
        :return:
        '''
        self.save_flag = True
        self.close()

    def not_save(self):
        '''
        不保存excel
        :return:
        '''
        self.save_flag = False
        self.close()



class ExcelSaveAlertDialog(QDialog):
    def __init__(self,excel_path):
        '''
        excel保存成功对话框初始化
        :param excel_path: excel路径
        '''
        super().__init__()
        self.excel_path = excel_path
        self.init_ui()

    def init_ui(self):
        # 设置窗口大小
        self.setMaximumSize(600, 400)
        self.setMinimumSize(600, 400)

        #设置窗口样式
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框
        self.setWindowOpacity(0.9)
        self.setStyleSheet("QWidget{ background-color:white;border-radius:30px; }")

        #标题
        title = QLabel()
        title.setText("分析结果excel表格已成功保存")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("幼圆", 15, 0))

        #excel图标
        picture = QLabel()
        picture.setMinimumSize(100, 100)
        picture.setMaximumSize(100, 100)
        pix = QPixmap("./images/excel.png")  # 按指定路径找到图片
        picture.setPixmap(pix)  # 在label上显示图片
        picture.setScaledContents(True)  # 让图片自适应label大小

        #excel路径信息
        path = QLabel()
        path.setText("保存路径： "+self.excel_path)
        path.adjustSize()
        path.setWordWrap(True)
        path.setAlignment(Qt.AlignCenter)
        path.setFont(QFont("幼圆", 10, 0))

        #确认按钮
        quit = QPushButton("我知道了")
        quit.setMinimumSize(150, 50)
        quit.setMaximumSize(150, 50)
        quit.setStyleSheet("QPushButton{ border:2px;border-radius:10px;background-color:rgb(237, 246, 253) }"
                           "QPushButton:hover{ background-color:grey; }")
        quit.clicked.connect(self.close)

        #页面布局
        vbox = QVBoxLayout()

        hbox1 = QHBoxLayout()
        hbox1.addStretch(1)
        hbox1.addWidget(title)
        hbox1.addStretch(1)

        hbox2 = QHBoxLayout()
        hbox2.addStretch(1)
        hbox2.addWidget(picture)
        hbox2.addStretch(1)

        hbox3 = QHBoxLayout()
        hbox3.addStretch(1)
        hbox3.addWidget(path)
        hbox3.addStretch(1)

        hbox4 = QHBoxLayout()
        hbox4.addStretch(1)
        hbox4.addWidget(quit)
        hbox4.addStretch(1)

        vbox.addLayout(hbox1)
        vbox.addStretch(1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addStretch(1)
        vbox.addLayout(hbox4)

        self.setLayout(vbox)

