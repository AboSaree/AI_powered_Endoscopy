import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QLabel, QFileDialog, QProgressBar, 
                             QSpinBox, QCheckBox, QGroupBox, QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO

class VideoProcessor(QThread):
    progress_update = pyqtSignal(int)
    frame_processed = pyqtSignal(np.ndarray, bool)
    processing_finished = pyqtSignal(int, int)
    
    def __init__(self, video_path, model, threshold=0.5, skip_frames=1, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.model = model
        self.threshold = threshold
        self.skip_frames = skip_frames
        self.running = True
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        abnormal_frames = 0
        
        frame_count = 0
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % self.skip_frames != 0:
                continue
                
            processed_frames += 1
            
            # Preprocess frame for model
            resized = cv2.resize(frame, (224, 224))
            normalized = resized / 255.0
            input_data = np.expand_dims(normalized, axis=0)
            
            # Predict
            prediction = self.model.predict(input_data, verbose=0)[0][0]
            is_abnormal = prediction < self.threshold
            
            if is_abnormal:
                abnormal_frames += 1
                
            # Emit signals
            self.frame_processed.emit(frame, is_abnormal)
            progress = int((frame_count / total_frames) * 100)
            self.progress_update.emit(progress)
        
        cap.release()
        self.processing_finished.emit(processed_frames, abnormal_frames)
    
    def stop(self):
        self.running = False


class YoloProcessor(QThread):
    progress_update = pyqtSignal(int)
    processing_finished = pyqtSignal(int)
    
    def __init__(self, input_dir, output_dir, conf_threshold=0.3, parent=None):
        super().__init__(parent)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.conf_threshold = conf_threshold
        self.running = True
        
    def run(self):
        try:
            # Load YOLO model
            model = YOLO('best.pt')
            
            # Get list of all images in input directory
            image_files = [f for f in os.listdir(self.input_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                self.processing_finished.emit(0)
                return
                
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            processed_count = 0
            
            # Process each image
            for i, img_file in enumerate(image_files):
                if not self.running:
                    break
                    
                img_path = os.path.join(self.input_dir, img_file)
                
                # Run YOLO prediction
                results = model.predict(
                    img_path, 
                    imgsz=640, 
                    conf=self.conf_threshold, 
                    save=False  # We'll save manually
                )
                
                # Save the results with detections
                for j, result in enumerate(results):
                    # Get image with detections drawn
                    result_img = result.plot()
                    
                    # Create output filename
                    base_name = os.path.splitext(img_file)[0]
                    out_path = os.path.join(self.output_dir, f"{base_name}_detected.jpg")
                    
                    # Save image
                    cv2.imwrite(out_path, result_img)
                
                processed_count += 1
                progress = int((i + 1) / len(image_files) * 100)
                self.progress_update.emit(progress)
            
            self.processing_finished.emit(processed_count)
            
        except Exception as e:
            print(f"Error in YOLO processing: {e}")
            self.processing_finished.emit(0)
            
    def stop(self):
        self.running = False


class DesignViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D Model Design")
        self.setMinimumSize(800, 600)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Create image display label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        # Create close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        
        # Load the design image
        self.load_design_image()
        
    def load_design_image(self):
        # Try to find the design image
        image_path = "design.jpg"  # Default path
        
        # Check common locations if the image doesn't exist at the default path
        if not os.path.exists(image_path):
            # Check in the same directory as the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(script_dir, "design.jpg"),
                os.path.join(script_dir, "design.png"),
                os.path.join(script_dir, "images", "design.jpg"),
                os.path.join(script_dir, "images", "design.png"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break
        
        # Load and display the image if found
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            # Scale to fit the label while maintaining aspect ratio
            pixmap = pixmap.scaled(
                self.image_label.width(), 
                self.image_label.height(),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("Design image not found!\nPlease place 'design.jpg' or 'design.png' in the application directory.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Z-Line vs Esophagitis Video Analyzer")
        self.setMinimumSize(1000, 600)
        
        # Load model
        self.model = None
        try:
            self.model = tf.keras.models.load_model("zline_vs_esophagitis_model.h5")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
        
        self.video_path = ""
        self.output_dir = ""
        self.yolo_output_dir = ""
        self.processor = None
        self.yolo_processor = None
        self.abnormal_frame_count = 0
        self.abnormal_frames_dir = ""
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        
        # Display area
        self.display_label = QLabel("No video loaded")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
        self.display_label.setMinimumHeight(300)
        main_layout.addWidget(self.display_label)
        
        # Status and controls
        control_layout = QHBoxLayout()
        
        # Left side - video controls
        video_group = QGroupBox("Video Selection")
        video_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("Select Video")
        self.load_btn.clicked.connect(self.select_video)
        video_layout.addWidget(self.load_btn)
        
        self.out_dir_btn = QPushButton("Select Output Directory")
        self.out_dir_btn.clicked.connect(self.select_output_dir)
        video_layout.addWidget(self.out_dir_btn)
        
        self.video_status = QLabel("No video selected")
        video_layout.addWidget(self.video_status)
        
        self.out_dir_status = QLabel("No output directory selected")
        video_layout.addWidget(self.out_dir_status)
        
        # Add the 2D model button
        self.design_btn = QPushButton("2D Model")
        self.design_btn.clicked.connect(self.show_design)
        video_layout.addWidget(self.design_btn)
        
        video_group.setLayout(video_layout)
        control_layout.addWidget(video_group)
        
        # Middle - processing controls
        process_group = QGroupBox("Processing Settings")
        process_layout = QVBoxLayout()
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Abnormal Threshold (below is abnormal):"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 99)
        self.threshold_spin.setValue(50)  # 0.5 as percentage - below this is abnormal
        self.threshold_spin.setSuffix("%")
        threshold_layout.addWidget(self.threshold_spin)
        process_layout.addLayout(threshold_layout)
        
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Process every N frames:"))
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(1, 30)
        self.skip_spin.setValue(5)
        skip_layout.addWidget(self.skip_spin)
        process_layout.addLayout(skip_layout)
        
        self.show_all_frames = QCheckBox("Show all processed frames")
        self.show_all_frames.setChecked(True)
        process_layout.addWidget(self.show_all_frames)
        
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        process_layout.addWidget(self.process_btn)
        
        process_group.setLayout(process_layout)
        control_layout.addWidget(process_group)
        
        # Right side - YOLO controls
        yolo_group = QGroupBox("YOLO Detection")
        yolo_layout = QVBoxLayout()
        
        yolo_conf_layout = QHBoxLayout()
        yolo_conf_layout.addWidget(QLabel("YOLO Confidence:"))
        self.yolo_conf_spin = QSpinBox()
        self.yolo_conf_spin.setRange(10, 90)
        self.yolo_conf_spin.setValue(30)  # 0.3 as percentage
        self.yolo_conf_spin.setSuffix("%")
        yolo_conf_layout.addWidget(self.yolo_conf_spin)
        yolo_layout.addLayout(yolo_conf_layout)
        
        self.yolo_dir_btn = QPushButton("Set YOLO Output Directory")
        self.yolo_dir_btn.clicked.connect(self.select_yolo_dir)
        yolo_layout.addWidget(self.yolo_dir_btn)
        
        self.yolo_dir_status = QLabel("No YOLO output directory selected")
        yolo_layout.addWidget(self.yolo_dir_status)
        
        self.yolo_btn = QPushButton("Run YOLO on Abnormal Frames")
        self.yolo_btn.clicked.connect(self.run_yolo_detection)
        self.yolo_btn.setEnabled(False)
        yolo_layout.addWidget(self.yolo_btn)
        
        yolo_group.setLayout(yolo_layout)
        control_layout.addWidget(yolo_group)
        
        main_layout.addLayout(control_layout)
        
        # Progress area
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        main_layout.addLayout(progress_layout)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def show_design(self):
        """Show the 2D model design image in a new dialog"""
        viewer = DesignViewer(self)
        viewer.exec_()

    def select_video(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if filepath:
            self.video_path = filepath
            self.video_status.setText(f"Selected: {os.path.basename(filepath)}")
            self.check_ready()
    
    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        
        if directory:
            self.output_dir = directory
            self.out_dir_status.setText(f"Output: {os.path.basename(directory)}")
            self.check_ready()
    
    def select_yolo_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select YOLO Output Directory"
        )
        
        if directory:
            self.yolo_output_dir = directory
            self.yolo_dir_status.setText(f"YOLO Output: {os.path.basename(directory)}")
            self.check_yolo_ready()
    
    def check_ready(self):
        # Convert string paths to boolean conditions
        has_video = bool(self.video_path)
        has_output = bool(self.output_dir)
        has_model = self.model is not None
        
        self.process_btn.setEnabled(has_video and has_output and has_model)
    
    def check_yolo_ready(self):
        # Enable YOLO button if we have abnormal frames and output directory
        has_abnormal_frames = bool(self.abnormal_frames_dir) and os.path.exists(self.abnormal_frames_dir)
        has_frames = has_abnormal_frames and len(os.listdir(self.abnormal_frames_dir)) > 0
        has_output = bool(self.yolo_output_dir)
        
        self.yolo_btn.setEnabled(bool(has_frames and has_output))
    
    def start_processing(self):
        if not self.video_path or not self.output_dir:
            return
            
        # Disable controls during processing
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.out_dir_btn.setEnabled(False)
        
        # Reset counter
        self.abnormal_frame_count = 0
        
        # Create output directory for abnormal frames if it doesn't exist
        self.abnormal_frames_dir = os.path.join(self.output_dir, "abnormal_frames")
        os.makedirs(self.abnormal_frames_dir, exist_ok=True)
        
        # Get processing parameters
        threshold = self.threshold_spin.value() / 100.0  # Convert percentage to decimal
        # Default to 0.5 if not specified
        if threshold == 0:
            threshold = 0.5
        skip_frames = self.skip_spin.value()
        
        # Start processing thread
        self.status_label.setText("Processing video...")
        self.processor = VideoProcessor(
            self.video_path, self.model, threshold, skip_frames
        )
        self.processor.progress_update.connect(self.update_progress)
        self.processor.frame_processed.connect(self.handle_processed_frame)
        self.processor.processing_finished.connect(self.processing_done)
        self.processor.start()
    
    def run_yolo_detection(self):
        if not self.abnormal_frames_dir or not self.yolo_output_dir:
            QMessageBox.warning(
                self, 
                "Missing Information", 
                "Please ensure you have both abnormal frames detected and a YOLO output directory selected."
            )
            return
        
        # Check if abnormal frames exist
        if not os.path.exists(self.abnormal_frames_dir) or len(os.listdir(self.abnormal_frames_dir)) == 0:
            QMessageBox.warning(
                self, 
                "No Abnormal Frames", 
                "No abnormal frames were detected to process with YOLO."
            )
            return
        
        # Disable buttons during processing
        self.yolo_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        
        # Get YOLO confidence
        conf_threshold = self.yolo_conf_spin.value() / 100.0
        
        # Start YOLO processing
        self.status_label.setText("Processing abnormal frames with YOLO...")
        self.progress_bar.setValue(0)
        
        self.yolo_processor = YoloProcessor(
            self.abnormal_frames_dir, 
            self.yolo_output_dir, 
            conf_threshold
        )
        self.yolo_processor.progress_update.connect(self.update_progress)
        self.yolo_processor.processing_finished.connect(self.yolo_processing_done)
        self.yolo_processor.start()
    
    def handle_processed_frame(self, frame, is_abnormal):
        # Display frame
        if is_abnormal or self.show_all_frames.isChecked():
            self.display_frame(frame, is_abnormal)
        
        # Save abnormal frames
        if is_abnormal:
            self.abnormal_frame_count += 1
            filename = f"abnormal_frame_{self.abnormal_frame_count:04d}.jpg"
            filepath = os.path.join(self.abnormal_frames_dir, filename)
            cv2.imwrite(filepath, frame)
    
    def display_frame(self, frame, is_abnormal):
        # Convert frame for display
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        # Add indicator for abnormal frames
        if is_abnormal:
            # Add red border
            border_thickness = max(3, min(width, height) // 100)
            cv2.rectangle(
                frame, 
                (border_thickness, border_thickness), 
                (width - border_thickness, height - border_thickness), 
                (0, 0, 255), 
                border_thickness
            )
            
            # Add text indicator
            font_scale = max(0.5, min(width, height) / 500)
            cv2.putText(
                frame, 
                "ABNORMAL", 
                (border_thickness * 3, border_thickness * 6), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 0, 255), 
                2
            )
        
        # Convert to QImage and display
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.display_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            width, 
            height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
        # Process events to update display
        QApplication.processEvents()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def processing_done(self, processed_frames, abnormal_frames):
        self.status_label.setText(
            f"Finished! Processed {processed_frames} frames, "
            f"found {abnormal_frames} abnormal frames."
        )
        
        # Re-enable controls
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.out_dir_btn.setEnabled(True)
        
        # Check if YOLO processing can be enabled
        self.check_yolo_ready()
        
        # Show completion message
        QMessageBox.information(
            self,
            "Processing Complete",
            f"Video analysis complete!\n\n"
            f"• Processed {processed_frames} frames\n"
            f"• Detected {abnormal_frames} abnormal frames\n"
            f"• Saved to: {self.abnormal_frames_dir}"
        )
    
    def yolo_processing_done(self, processed_count):
        self.status_label.setText(
            f"YOLO processing complete! Processed {processed_count} images."
        )
        
        # Re-enable buttons
        self.yolo_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        
        # Show completion message
        QMessageBox.information(
            self,
            "YOLO Processing Complete",
            f"YOLO object detection complete!\n\n"
            f"• Processed {processed_count} abnormal frames\n"
            f"• Results saved to: {self.yolo_output_dir}"
        )
        
        # Show the last processed image if available
        if processed_count > 0:
            try:
                # Get the last processed image
                yolo_files = [f for f in os.listdir(self.yolo_output_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if yolo_files:
                    latest_file = os.path.join(self.yolo_output_dir, yolo_files[-1])
                    img = cv2.imread(latest_file)
                    if img is not None:
                        self.display_frame(img, False)  # Display without abnormal marking
            except Exception as e:
                print(f"Error displaying YOLO result: {e}")
    
    def closeEvent(self, event):
        # Stop processing threads if running
        if self.processor and self.processor.isRunning():
            self.processor.stop()
            self.processor.wait()
            
        if self.yolo_processor and self.yolo_processor.isRunning():
            self.yolo_processor.stop()
            self.yolo_processor.wait()
            
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())