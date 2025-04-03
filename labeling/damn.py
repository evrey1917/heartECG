import sys
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from scipy.signal import find_peaks, savgol_filter
import pyqtgraph as pg
import h5py

def ultra_filter(signals, n=None, channel=None):
    if n == None and channel==None:
        y = signals
    else:
        y = signals[n][channel]

    # 1. Найдем пики
    peaks, _ = find_peaks(y, height=(0.1, 0.5), distance=10)
    peaks = peaks[peaks < len(y)]

    # 2. Найдем тренд с помощью фильтра Савицкого-Голея
    y_trend = savgol_filter(y, window_length=51, polyorder=3)

    # 3. Удалим тренд, чтобы получить колебания (пики и волны)
    y_fluctuations = y - y_trend

    # 4. Обработаем волны (сгладим их)
    y_waves_smoothed = savgol_filter(y_fluctuations, window_length=21, polyorder=2)

    # 5. Восстановим данные, добавив колебания к тренду
    y_restored = y_trend + y_waves_smoothed

    return y_restored

class ECGMarkingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ЭКГ Разметка PQRST")
        self.setGeometry(100, 100, 1200, 800)
        
        # Данные ЭКГ
        self.ecg_data = None
        self.sample_rate = 500  # Частота дискретизации (Гц)
        self.current_lead = 0
        self.leads = []
        self.time_axis = None
        self.current_file_index = 0
        self.ecg_files = []
        self.current_folder = ""
        self.all_markers = []  # Массив для хранения разметки всех файлов
        
        # Разметка
        self.markers = {'P': [], 'Q': [], 'R': [], 'S': [], 'T': []}
        self.current_marker = 'P'
        
        # Инициализация UI
        self.init_ui()
        
    def init_ui(self):
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Панель инструментов
        toolbar = QHBoxLayout()
        main_layout.addLayout(toolbar)
        
        # Кнопки навигации
        self.prev_btn = QPushButton("← Назад")
        self.prev_btn.clicked.connect(self.load_prev_file)
        self.prev_btn.setEnabled(False)
        toolbar.addWidget(self.prev_btn)
        
        # Комбобокс для выбора файла
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.file_combo_changed)
        toolbar.addWidget(self.file_combo)
        
        self.next_btn = QPushButton("Вперед →")
        self.next_btn.clicked.connect(self.load_next_file)
        self.next_btn.setEnabled(False)
        toolbar.addWidget(self.next_btn)
        
        # Кнопки загрузки/сохранения
        self.load_btn = QPushButton("Загрузить папку")
        self.load_btn.clicked.connect(self.load_ecg_folder)
        toolbar.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("Сохранить разметку")
        self.save_btn.clicked.connect(self.save_markers)
        self.save_btn.setEnabled(False)
        toolbar.addWidget(self.save_btn)
        
        # Выбор отведения
        self.lead_label = QLabel("Отведение:")
        toolbar.addWidget(self.lead_label)
        
        self.lead_combo = QComboBox()
        self.lead_combo.currentIndexChanged.connect(self.change_lead)
        toolbar.addWidget(self.lead_combo)
        
        # Выбор маркера
        self.marker_label = QLabel("Маркер:")
        toolbar.addWidget(self.marker_label)
        
        self.marker_combo = QComboBox()
        self.marker_combo.addItems(['P', 'Q', 'R', 'S', 'T'])
        self.marker_combo.currentTextChanged.connect(self.change_marker)
        toolbar.addWidget(self.marker_combo)
        
        # График ЭКГ
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Амплитуда')
        self.plot_widget.setLabel('bottom', 'Время', 'сек')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.setLimits(xMin=0)
        
        # Отключаем контекстное меню правой кнопки мыши
        self.plot_widget.setMenuEnabled(False)
        
        # Включаем обработку кликов
        self.plot_widget.scene().sigMouseClicked.connect(self.plot_clicked)
        
        main_layout.addWidget(self.plot_widget)
        
        # Статус бар
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Готово")
        
    def load_ecg_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с файлами ЭКГ")
        if folder:
            self.current_folder = folder
            self.ecg_files = sorted([f for f in os.listdir(folder) if f.endswith('.h5')])
            self.current_file_index = 0
            self.all_markers = []
            
            if self.ecg_files:
                # Заполняем комбобокс
                self.file_combo.clear()
                self.file_combo.addItems(self.ecg_files)
                
                self.load_current_file()
                self.update_nav_buttons()
                self.save_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Предупреждение", "В папке нет файлов .h5")

    def file_combo_changed(self, index):
        """Обработчик изменения выбранного файла в комбобоксе"""
        if index != self.current_file_index and 0 <= index < len(self.ecg_files):
            # Сохраняем текущую разметку
            if self.current_file_index < len(self.all_markers):
                self.all_markers[self.current_file_index] = self.markers.copy()
            
            self.current_file_index = index
            self.load_current_file()
            self.update_nav_buttons()

    def update_nav_buttons(self):
        """Обновляет состояние кнопок навигации"""
        self.prev_btn.setEnabled(self.current_file_index > 0)
        self.next_btn.setEnabled(self.current_file_index < len(self.ecg_files) - 1)
        
        # Убедимся, что комбобокс показывает правильный файл
        if self.file_combo.currentIndex() != self.current_file_index:
            self.file_combo.setCurrentIndex(self.current_file_index)

    def load_current_file(self):
        if not self.ecg_files or self.current_file_index >= len(self.ecg_files):
            return
            
        file_path = os.path.join(self.current_folder, self.ecg_files[self.current_file_index])
        
        try:
            with h5py.File(file_path, 'r') as f:
                self.ecg_data = f['ecg'][()]
            
            self.ecg_data = [ultra_filter(data) for data in self.ecg_data]
            
            # Создаем временную ось
            self.time_axis = np.arange(len(self.ecg_data[0])) / self.sample_rate
            
            # Создаем список отведений
            self.leads = [str(i + 1) for i in range(len(self.ecg_data))]
            
            # Загружаем разметку для этого файла
            if self.current_file_index < len(self.all_markers):
                self.markers = self.all_markers[self.current_file_index]
            else:
                self.markers = {'P': [], 'Q': [], 'R': [], 'S': [], 'T': []}
                self.all_markers.append(self.markers.copy())
            
            self.lead_combo.clear()
            self.lead_combo.addItems(self.leads)
            self.current_lead = 0
            
            self.update_plot()
            self.status_bar.showMessage(f"Файл {self.current_file_index+1}/{len(self.ecg_files)}: {self.ecg_files[self.current_file_index]}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл: {str(e)}")
            self.ecg_data = None

    def load_next_file(self):
        if self.current_file_index < len(self.ecg_files) - 1:
            # Сохраняем текущую разметку
            if self.current_file_index < len(self.all_markers):
                self.all_markers[self.current_file_index] = self.markers.copy()
            
            self.current_file_index += 1
            self.load_current_file()
            self.update_nav_buttons()

    def load_prev_file(self):
        if self.current_file_index > 0:
            # Сохраняем текущую разметку
            if self.current_file_index < len(self.all_markers):
                self.all_markers[self.current_file_index] = self.markers.copy()
            
            self.current_file_index -= 1
            self.load_current_file()
            self.update_nav_buttons()
    
    def save_markers(self):
        if not self.ecg_files:
            return
            
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить разметку", "", 
                                                 "NPZ Files (*.npz);;All Files (*)", 
                                                 options=options)
        if file_name:
            try:
                # Сохраняем индексы точек как массив массивов
                markers_to_save = []
                for markers in self.all_markers:
                    markers_dict = {}
                    for key in markers:
                        # Преобразуем временные метки в индексы точек
                        if self.time_axis is not None and len(self.time_axis) > 0:
                            indices = [np.argmin(np.abs(self.time_axis - x)) for x in markers[key]]
                            markers_dict[key] = indices
                    markers_to_save.append(markers_dict)
                
                # Сохраняем имена файлов и разметку
                np.savez(file_name, 
                        files=np.array(self.ecg_files),
                        markers=markers_to_save)
                
                QMessageBox.information(self, "Сохранено", "Разметка успешно сохранена")
                self.status_bar.showMessage(f"Разметка сохранена в {file_name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {str(e)}")
    
    def change_lead(self, index):
        self.current_lead = index
        self.update_plot()
    
    def change_marker(self, marker):
        self.current_marker = marker
    
    def plot_clicked(self, event):
        if self.ecg_data is None:
            return
        
        pos = event.scenePos()
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        if not self.plot_widget.plotItem.vb.viewRect().contains(mouse_point):
            return
        
        # Находим ближайшую точку данных
        idx = np.argmin(np.abs(self.time_axis - x))
        x = self.time_axis[idx]
        # y = self.ecg_data[self.current_lead][idx]
        
        if event.button() == Qt.LeftButton:
            # Добавляем индекс точки как маркер
            self.markers[self.current_marker].append(x)
            self.update_plot()
        elif event.button() == Qt.RightButton:
            # Удаляем ближайший маркер
            min_dist = float('inf')
            closest_marker = None
            
            for marker_type in self.markers:
                for marker_x in self.markers[marker_type]:
                    dist = abs(marker_x - x)
                    if dist < min_dist:
                        min_dist = dist
                        closest_marker = (marker_type, marker_x)
            
            if closest_marker and min_dist < 0.05:  # Порог ~50 мс
                marker_type, marker_x = closest_marker
                self.markers[marker_type].remove(marker_x)
                self.update_plot()
    
    def update_plot(self):
        if self.ecg_data is None:
            return
        
        self.plot_widget.clear()
        lead_data = self.ecg_data[self.current_lead]
        self.plot_widget.plot(self.time_axis, lead_data, pen='b')
        
        # Цвета для маркеров
        colors = {'P': (255, 0, 255),
                  'Q': (0, 255, 0),
                  'R': (255, 0, 0), 
                  'S': (0, 255, 255),
                  'T': (255, 255, 0)}
        
        # Сортируем все маркеры перед отрисовкой
        for marker_type in self.markers:
            self.markers[marker_type].sort()
        
        for marker_type, color in colors.items():
            for x in self.markers[marker_type]:
                # Вертикальная линия
                line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen(color, width=1))
                self.plot_widget.addItem(line)
                
                # Подпись
                idx = np.argmin(np.abs(self.time_axis - x))
                y = self.ecg_data[self.current_lead][idx]
                text = pg.TextItem(marker_type, color=color, anchor=(0.5, 1.5))
                text.setPos(x, y)
                self.plot_widget.addItem(text)
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Выход',
                                   "Вы уверены, что хотите выйти? Все несохраненные данные будут потеряны.",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ECGMarkingApp()
    window.show()
    sys.exit(app.exec_())