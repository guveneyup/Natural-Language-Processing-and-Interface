import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import pickle

pickle_df = pd.read_pickle("final_df_pickle.py")
X = pickle_df["reviewText"]
y = pickle_df["sentiment_label"]
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(y)

tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3)).fit(X)
X_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(X)
rf_final_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(911, 673)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 580, 851, 81))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.textEdit = QtWidgets.QTextEdit(Form)
        self.textEdit.setGeometry(QtCore.QRect(30, 60, 421, 501))
        self.textEdit.setObjectName("textEdit")
        self.listView = QtWidgets.QListView(Form)
        self.listView.setGeometry(QtCore.QRect(460, 61, 421, 501))
        self.listView.setObjectName("listView")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(30, 30, 171, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(460, 30, 171, 16))
        self.label_2.setObjectName("label_2")
        self.pushButton.clicked.connect(self.save_file)
        self.pushButton_2.clicked.connect(self.prediction)
        self.pushButton_3.clicked.connect(self.open_file)
        self.items = []
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)


    def open_file(self):
        fill = QtWidgets.QFileDialog.getOpenFileNames(Form, "Dosya Ac", os.getenv("HOME"))
        with open(fill[0][0], "r") as file:
            self.textEdit.setText(file.read())

    def prediction(self):
        self.data = [self.textEdit.toPlainText()][0].split("\n")
        self.new_data = []
        for i in self.data:
            if i != '':
                self.new_data.append(i)
        self.df = pd.DataFrame()
        for i in range(1, len(self.new_data)):
            self.df.loc[i, "customerid"] = self.new_data[i].split(",")[0]
        for i in range(1, len(self.new_data)):
            self.df.loc[i, "text"] = self.new_data[i].split(",")[1]
        ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3)).fit(X)
        self.yorum = ngram_vectorizer.transform(self.df["text"])
        self.pred = rf_final_model.predict(self.yorum)
        self.model = QtGui.QStandardItemModel(self.listView)
        self.listView.setModel(self.model)
        for i in range(0, len(self.pred)):
            if self.pred[i] == 0:
                self.item = QtGui.QStandardItem(str(f"Customerid {self.df.loc[i+1,'customerid']} Negative"))
                self.model.appendRow(self.item)
                self.items.append(self.item)
            else:
                self.item = QtGui.QStandardItem(str(f"Customerid {self.df.loc[i+1,'customerid']} Positive"))
                self.model.appendRow(self.item)
                self.items.append(self.item)

    def save_file(self):
        file = QtWidgets.QFileDialog.getSaveFileName(Form, "Dosya Kaydet", os.getenv("HOME"))
        with open(file[0], "w") as fil:
            for item in self.items:
                fil.write(item.text() + "\n")



    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Natural Language Processing"))
        self.pushButton_3.setText(_translate("Form", "Aç"))
        self.pushButton_2.setText(_translate("Form", "Tahmin Et"))
        self.pushButton.setText(_translate("Form", "Kaydet"))
        self.label.setText(_translate("Form", "Tahmin Edilecek Yorumlar"))
        self.label_2.setText(_translate("Form", "Tahmin Değerleri"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())


