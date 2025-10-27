### Football Analysis System Using Machine Learning, Computer Vision & Deep Learning.

This project leverages Machine Learning,Computer Vision,and Deep Learning to build a comprehensive Football Analysis System capable of detecting players, referees, and footballs; tracking their movements; and generating real-world performance metrics like speed and distance covered.

#### Overview

The system uses YOLO (You Only Look Once) — a state-of-the-art object detection model — to identify key entities (players, referees, footballs) in video frames. It integrates tracking algorithms to follow these objects across frames, ensuring continuity and accurate performance analytics.

I fine-tuned my own YOLO model on a custom football dataset to enhance detection accuracy and performance under varying lighting and motion conditions.

Beyond detection and tracking, the system uses KMeans clustering to segment and assign players to teams based on t-shirt colors, Optical Flow to measure camera motion, and Perspective Transformation to convert measurements from pixels to real-world meters, enabling precise computation of player speed and total distance covered.

This project demonstrates end-to-end sports analytics using Computer Vision, Machine Learning and Deep learning.
---

#### Key Things in This Project:

1. Ultralytics & YOLOv8 for object detection on images and videos.  
2. Fine-tuned & trained my own YOLO model on a custom dataset.  
3. Applied KMeans Clustering for pixel segmentation to determine player t-shirt colors.  
4. Used Optical Flow to estimate and correct for camera movement between frames.  
5. Applied Perspective Transformation using OpenCV to model scene depth and convert pixels to meters.  
6. Measured Player Speed & Distance Covered across the pitch in real-world units.

---

#### Datasets Used:

- Kaggle Dataset: [https://www.kaggle.com/competitions/d...](https://www.kaggle.com/competitions/d...)  
- Video Link (since Kaggle removed videos): [https://drive.google.com/file/d/1t6ag...](https://drive.google.com/file/d/1t6ag...)  
- Roboflow Football Dataset: [https://universe.roboflow.com/roboflo...](https://universe.roboflow.com/roboflo...)

---

#### Key Technologies

- YOLOv8 (Ultralytics)
- OpenCV (CV2)
- Scikit-learn (KMeans Clustering)
- Optical Flow
- Python

---

#### Conclusion

This Football Analysis System integrates AI,Computer Vision, and Deep Learning to solve real-world problems in sports analytics. It combines multiple CV techniques—from detection and tracking to motion analysis—showcasing how machine learning can deliver actionable insights in sports performance tracking and game analysis.
