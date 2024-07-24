import sys
import easygui
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.Qt import *
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import data.start as start
import data.mainw as mainw
import data.mainw2 as mainw2

class VideoThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(QtGui.QPixmap)
    def __init__(self, recognize_function):
        super().__init__()
        self.running = True
        self.recognize_function = recognize_function
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Ошибка: не удалось открыть камеру.")
            self.running = False
            return
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр")
                break

            try:
                processed_image = self.recognize_function(frame)
                if processed_image:
                    processed_image_qt = self.pil2pixmap(processed_image)
                    self.change_pixmap_signal.emit(processed_image_qt)
            except Exception as e:
                print(f"Ошибка при обработке изображения: {e}")
        cap.release()
    def stop(self):
        self.running = False
        self.wait()
    def pil2pixmap(self, image):
        if image is None:
            return QtGui.QPixmap()
        image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        qim = QtGui.QImage(data, image.size[0], image.size[1], QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(qim)
        return pixmap

class ExampleApp3(QtWidgets.QMainWindow, mainw2.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self._back)
        self.pushButton_2.clicked.connect(self._start)
        self.model = YOLO('data/best.pt')
        self.thread = None
    def _back(self):
        if self.thread:
            self.thread.stop()
        window3.close()
        window.show()
    def _start(self):
        self.thread = VideoThread(self.recognize_image)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
    def update_image(self, pixmap):
        if pixmap:
            self.label.setPixmap(pixmap)
        else:
            print("Получен пустой pixmap")
    def recognize_image(self, frame):
        if frame is None:
            print("Получен пустой кадр.")
            return None
        try:
            img = frame
            self.model.cpu()
            results = self.model(img, imgsz=640, iou=0.1, conf=0.257, verbose=False)
            classes = results[0].boxes.cls.cpu().numpy()
            class_names = results[0].names

            masks = results[0].masks.data if results[0].masks is not None else []
            num_masks = masks.shape[0] 

            colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]

            labeled_image = img.copy()

            for i in range(num_masks):
                color = colors[i]
                mask = masks[i].cpu()

                mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                class_index = int(classes[i])
                class_name = class_names[class_index]

                mask_contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(labeled_image, mask_contours, -1, color, 3)
                cv2.putText(labeled_image, class_name,
                            (int(mask_contours[0][:, 0, 0].mean()), int(mask_contours[0][:, 0, 1].mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
            labeled_image_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
            # Создаем объект Image из массива NumPy
            pil_image = Image.fromarray(labeled_image_rgb)
            return pil_image
        except Exception as e:
            print(f"Ошибка при обработке изображения: {e}")
            return None     

class ExampleApp2(QtWidgets.QMainWindow, mainw.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self._back)
        self.pushButton_2.clicked.connect(self._detect)
        self.pushButton_3.clicked.connect(self._select)
        self.model = YOLO('data/best.pt')
    def _select(self):
        global file_path
        file_path = easygui.fileopenbox(filetypes=["*.docx"])
    def _detect(self):
        image = self.recognize_image(file_path)
        data = image.tobytes("raw", "RGBA")
        qim = QImage(data, image.size[0], image.size[1], QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qim)
        self.label.setPixmap(pixmap)
    def _back(self):
        window2.close()
        window.show()
    def recognize_image(self,path):
        img = cv2.imread(path)
        self.model.cpu()
        results = self.model(img, imgsz=640, iou=0.3, conf=0.257, verbose=False)
        classes = results[0].boxes.cls.cpu().numpy()
        class_names = results[0].names

        masks = results[0].masks.data
        num_masks = masks.shape[0]

        colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]

        labeled_image = img.copy()

        for i in range(num_masks):
            color = colors[i]
            mask = masks[i].cpu()

            mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            class_index = int(classes[i])
            class_name = class_names[class_index]

            mask_contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(labeled_image, mask_contours, -1, color, 3)
            cv2.putText(labeled_image, class_name,
                        (int(mask_contours[0][:, 0, 0].mean()), int(mask_contours[0][:, 0, 1].mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        labeled_image_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        # Создаем объект Image из массива NumPy
        pil_image = Image.fromarray(labeled_image_rgb)
        return pil_image

class ExampleApp(QtWidgets.QMainWindow, start.Ui_Start_Window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pB_Photo.clicked.connect(self.next_Photo)
        self.pB_Camera.clicked.connect(self.next_Camera)
    def next_Photo(self):
        global window2
        window2 = ExampleApp2()  
        window2.show()
        window.close()
    def next_Camera(self):
        global window3
        window3 = ExampleApp3()  
        window3.show()
        window.close()

def main():
    app = QtWidgets.QApplication(sys.argv)
    global window
    window = ExampleApp()  
    window.show()  
    app.exec_()  

if __name__ == '__main__':
    main()  
