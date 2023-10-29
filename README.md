# pythonProject_fdm
🫀 Heart Disease Prediction with Machine Learning 🤖

Welcome to our Heart Disease Prediction repository! This project harnesses the power of Machine Learning to predict heart disease, aiding in early diagnosis and proactive healthcare. 🏥

Key Features:

📊 Data Analysis: Thorough exploration and analysis of heart disease datasets.
🧠 Machine Learning Models: Implementation of various ML algorithms for accurate predictions.
📈 Performance Metrics: Evaluation using precision, recall, and F1-score for robust model assessment.
🎨 Data Visualization: Clear visualizations for insights and model interpretation.
🚀 Deployment Ready: Codebase prepared for seamless deployment in real-world scenarios.
Tech Stack:
Python, scikit-learn, pandas, matplotlib, Jupyter Notebook.


Features that are considered in the Model
Demographic:
• Sex: male or female(Nominal)
• Age: Age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
Behavioral
• Current Smoker: whether or not the patient is a current smoker (Nominal)
• Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarettes, even half a cigarette.)
Medical( history)
• BP Meds: whether or not the patient was on blood pressure medication (Nominal)
• Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
• Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
• Diabetes: whether or not the patient had diabetes (Nominal)
Medical(current)
• Tot Chol: total cholesterol level (Continuous)
• Sys BP: systolic blood pressure (Continuous)
• Dia BP: diastolic blood pressure (Continuous)
• BMI: Body Mass Index (Continuous)
• Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)
• Glucose: glucose level (Continuous)
Predict variable (desired target)
• 10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”)

