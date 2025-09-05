Introduction
This project analyzes a heart disease dataset to predict and understand patterns associated with heart disease. The analysis employs three distinct data mining techniques: machine learning classification, K-Means clustering, and frequent pattern mining using the Apriori algorithm.


Dataset Overview
The dataset consists of 918 patient records with 12 attributes.

Attributes:

Attribute	Data Type
Age	int64
Sex	object
ChestPainType	object
RestingBP	int64
Cholesterol	int64
FastingBS	int64
RestingECG	object
MaxHR	int64
ExerciseAngina	object
Oldpeak	float64
ST_Slope	object
HeartDisease	int64

Methodology & Analysis
Three different data mining techniques were applied to the dataset to gain unique insights.

1. Classification Analysis
A Random Forest classifier was trained to predict the presence of heart disease. The model achieved strong performance, with key features identified including age, cholesterol levels, chest pain type, and ST slope.


Performance:

Classification Report

precision	recall	f1-score	support
0 (No HD)	0.86	0.86	0.86	77
1 (HD)	0.90	0.90	0.90	107
Accuracy			0.88	184
Macro Avg	0.88	0.88	0.88	184
Weighted Avg	0.88	0.88	0.88	184


Confusion Matrix


True Negatives (No HD): 66 

False Positives: 11 

False Negatives: 11 

True Positives (HD): 96 

2. Clustering Analysis

K-Means clustering (k=2) was used to group the data into distinct patient profiles. The clusters were visualized using Principal Component Analysis (PCA). This demonstrated that patients can be naturally grouped based on health indicators even without diagnostic labels, which is valuable for segmenting patients into different risk categories.


3. Frequent Pattern Mining
The Apriori algorithm was employed to extract frequent patterns and strong association rules from the data. These rules help identify combinations of features that frequently co-occur with heart disease.


Discovered Strong Rules:

Antecedents	Consequents	Support	Confidence	Lift
{Old, NormalBP}	{HeartDisease}	0.22	0.74	1.26
{Male, ChestPainType_ASY}	{HeartDisease}	0.28	0.68	1.22
{Old, HighChol}	{HeartDisease}	0.25	0.70	1.19
{HighHR, ExerciseAngina_No}	{HeartDisease}	0.31	0.66	1.25
{ST_Slope_Up, Young}	{HeartDisease}	0.26	0.73	1.23


Overall Implications
Combining predictive modeling with unsupervised learning and pattern-based analysis provides a comprehensive view of the data. This multi-faceted approach supports both accurate prediction of disease and a deeper understanding of risk factors, which is valuable for medical professionals and public health initiatives.


Conclusion
The Random Forest model demonstrated good performance in classifying heart disease. K-Means clustering successfully identified distinct patient groups based on health indicators. Association rule mining uncovered common combinations of risk factors that may indicate a higher probability of heart disease.
