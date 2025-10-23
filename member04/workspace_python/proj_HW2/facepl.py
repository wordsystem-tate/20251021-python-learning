import cv2
from PyQt5.QtWidgets import QLabel,QApplication,QWidget,QVBoxLayout
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import QTimer
import sys
import face_recognition
import numpy as np
import os
from tqdm import tqdm#プログレスバー表示ライブラリ
import pickle#データ保存用
import psycopg2

tolerance=0.4
data_file="faca_data.pkl"#保存するデータファイル名

#顔データをロードする
def load_face_data(file):
    if os.path.exists(file):
            with open(file,"rb")as f:
                return pickle.load(f)
    return None
        
#顔データを保存する
def save_face_data(file,encodings,names):
    with open(file,"wb")as f:
        pickle.dump((encodings,names),f)

class VideoWindow(QWidget):
    def __init__(self):
        
        #画像フォルダのパス
        image_folder='images/'
        #既存データをロード
        face_data=load_face_data(data_file)
        if face_data:
            known_face_encoding,known_face_names=face_data
            print("顔データをロードしました。")
        else:
            print("顔データが見つからないため、新たに作成します。")
            #画像フォルダのパス
            image_folder='images/'
            
            #画像を読み込み、顔の特徴を記録する
            known_face_encodings=[]
            known_face_names=[]
            #フォルダごとに画像を処理
            folder_list=[folder for folder in os.listdir(image_folder)if os.path.isdir(os.path.join(image_folder,folder))]
            #tqbmをつかってフォルダの処理状況を表示
            for foldername in tqdm(folder_list,desc=f"Processing Folders"):
                folder_path=os.path.join(image_folder,foldername)
                
                #フォルダ内の画像ファイルを処理
                image_files=[f for f in os.listdir(folder_path)if f.endswith(".jpg")]
                
                #tqbmを使ってフォルダ内の画像ファイルの処理状況を表示
                for filename in tqdm(image_files,desc=f"Processing Images in {foldername}",leave=False):
                    image_path=os.path.join(folder_path,filename)
                    image=face_recognition.load_image_file(image_path)
                    encoding=face_recognition.face_encodings(image)
                    
                    if encoding:
                        #複数の画像から同じ人物の特徴を追加
                        known_face_encodings.append(encoding[0])
                        known_face_names.append(foldername)
                        #データを保存
                        save_face_data(data_file,known_face_encodings,known_face_names)
                    
        super().__init__()
        self.setWindowTitle("カメラビュー")
        self.resize(640,600)
        
        #カメラ映像表示用ラベル
        self.video_label=QLabel()
        self.video_label.setFixedSize(640,600)
        self.pers_inf=QLabel("")
        self.pers_inf.setStyleSheet("font-size:16px;color:red;")
        
        #下に表示するラベル
        self.text_label=QLabel("初期メッセージ")
        self.text_label.setStyleSheet("font-size: 16px;color:blue;")
        
        #レイアウトに追加
        layout=QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.pers_inf)
        self.setLayout(layout)
        
        #カメラ起動
        self.cap=cv2.VideoCapture(0)
        
        #映像更新用タイマー
        self.video_timer=self.startTimer(30)
        
        #メッセージ更新用タイマー
        self.message_timer=QTimer(self)
        self.message_timer.timeout.connect(self.update_message)
        self.message_timer.start(2000)#2秒ごとにメッセージ更新
        self.message_index=0
        self.messages=["こんにちは","カメラ起動中","映像を確認してください","PyQt5で動的表示中"]
        
    def timerEvent(self,event):
        known_face_encodings,known_face_names=load_face_data(data_file)
        if event.timerId()==self.video_timer:
            ret,frame=self.cap.read()
            if ret:
                
                small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
                rgb_small_frame=np.ascontiguousarray(small_frame[:,:,::-1])
                #顔検出と顔認識
                face_locations=face_recognition.face_locations(rgb_small_frame)
                face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
                #顔の座標を拡大して元のフレームに戻す
                face_locations=[(top*4,right*4,bottom*4,left*4)for(top,right,bottom,left)in face_locations]
                #各顔に対して認識
                
                for(top,right,bottom,left),face_encodings in zip(face_locations,face_encodings):
                    matches=face_recognition.compare_faces(known_face_encodings,face_encodings,tolerance=tolerance)
                    name="Unknown"
                    
                    opened_pages=set()
                    #マッチした顔であれば名前を取得
                    if True in matches:
                        first_match_index=matches.index(True)
                        name=known_face_names[first_match_index]
                        if name not in opened_pages:
                            opened_pages.add(name)
                            connection=psycopg2.connect("host=172.22.4.88 dbname=learn user=learn password=learn")
                            #カードるインスタンス
                            cursor=connection.cursor()
                            cursor.execute("SELECT * FROM 個人情報 WHERE 名前 = %s",(name,))
                            query_result = cursor.fetchall()
                            self.set_person_inf(query_result)
                            
                    #顔の周りにボックスを描画
                    cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
                    font=cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame,name,(left+6,bottom-6),font,0.5,(255,255,255),1)
                    
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                h,w,ch=rgb.shape
                img=QImage(rgb.data,w,h,ch * w,QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(img))
                
    def update_message(self):
        self.text_label.setText(self.messages[self.message_index])
        self.message_index=(self.message_index+1)%len(self.messages)
        
    def set_person_inf(self,query_result):
        #データの存在確認
        if query_result:
            row = query_result[0]
            name = str(row[1])
            age = str(row[2])
            cls = str(row[3])
            scores = str(row[4])
            text = f"名前：{name} {age}歳 クラス：{cls}成績：{scores}"
            self.pers_inf.setText(text)
        else:
            self.pers_inf.setText("個人情報が見つかりませんでした")
            _
            
app=QApplication(sys.argv)
win=VideoWindow()
win.show()
sys.exit(app.exec_())    