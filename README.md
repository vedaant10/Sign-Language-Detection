# Real-Time Sign Language Recognition



A real-time sign language recognition system built using computer vision and deep learning, designed to translate hand gestures into text.

This project was created to assist a family member with speech limitations, with a focus on practicality, simplicity, and real world usability rather than generic datasets.

 

#### 

#### **Features**

Real-time sign recognition using a webcam



* Custom-trained gesture classifier



* Hand landmark extraction using MediaPipe



* Supports motion-based and static signs



* Modular pipeline: recording → training → live prediction



* Designed for easy expansion (more signs, voice output, multilingual support)





#### **Motivation**

This project was built for a family member who was temporarily unable to speak due to medical reasons.

Rather than creating a generic sign-language model, the goal was to build a personalized system that recognizes a small but meaningful vocabulary with high accuracy and low latency.



The focus was on:



* Real-time performance



* Ease of use



* Ethical data collection (no scraping personal sign data)



#### **How It Works**



Record custom sign videos using a webcam



Extract hand landmarks from each frame using MediaPipe



Train a neural network on landmark sequences



Predict signs in real time using a live camera feed



Display recognized signs as text on screen





**Project structure**

---

Sign-Language-Detection/

│

├── models/                # Trained model (.h5)

├── record\_signs.py        # Record sign videos

├── extract\_landmarks.py   # Convert videos to landmark data

├── train\_model.py         # Train the gesture classifier

├── realtime\_predict.py    # Real-time sign recognition

├── requirements.txt       # Dependencies

├── README.md              # Project documentation

└── .gitignore



#### **Steps to Install(code is in bold)**



1)Clone the repository

---

**git clone https://github.com/vedaant10/Sign-Language-Detection.git**

**cd Sign-Language-Detection**



###### 2)Create and activate a virtual environment



**python -m venv sign\_env**

**sign\_env\\Scripts\\activate   # Windows**



###### 3)Install dependencies



**pip install -r requirements.txt**



#### **Usage**

###### 

###### 1)Record signs



**python record\_signs.py**



###### 2)Extract landmarks



**python extract\_landmarks.py**



###### 3)Train the model



**python train\_model.py**



###### 4)Run real-time prediction



**python realtime\_predict.py**





##### **Notes \& Limitations**





* The model is trained on custom data, not large public datasets



* Accuracy improves significantly with more samples per sign



* Lighting, camera angle, and background affect performance



* This is not a full ASL/ISL translator (by design)

##### 

##### **Future improvements** 



* Voice output (text-to-speech)



* More gesture classes



* Improved temporal modeling (LSTM/Transformer)



* Multilingual output (Hindi/English)



* Mobile camera integration



* User calibration mode



##### **Ethical consideration**



* No scraped or private datasets were used



* All training data was recorded with consent



* The project prioritizes accessibility and user dignity





##### **Contact**



If you have suggestions, feedback, or ideas for collaboration, feel free to open an issue or reach out.

