# Multisensor-systems-Pattern-Classification--KNN-Algorithm-

# [Brief Overview] 
## [Section One - Data Processing / Fault Classification]
Sensor measurements/vibration data from five test rigs were used (shown below) to design a health monitoring system of a rotating machine through exracting informative features from the raw (sensor) data along with evaluating the data in frequency and time-domain.

![image](https://user-images.githubusercontent.com/42310216/146125282-1d32796c-b806-4697-b06f-1ee69d04c6ac.png)

### [Rotating Machine Faults Considered]
- Fault 1: Bearing Fault
- Fault 2: Gear Mesh 
- Fault 3: Imbalance 
- Fault 4: Misalignment 
- Fault 5: Resonance 

### [Energy levels in six frequency bands]
The raw data was initially normalized and then processed further to deduce the power spectral density for each individual features with appropriate filter (Butterworth) implemented


![image](https://user-images.githubusercontent.com/42310216/146125941-c90a56c8-3583-4bc9-83ba-4b1b01fb07d7.png)

### [Principal Component Analysis]
Principal Component Analysis, PCA, method was implemented to reduce the number of dimensions/features and for visualization purposes. 


![image](https://user-images.githubusercontent.com/42310216/146126161-26d6affc-0b35-4ca5-ad91-546a25cae6a7.png)

## [Section Two - Health Monitoring System]
1-Nearest Neighbor Algorithm was implemented using the *Euclidean* distance measure. KNN algorithm clasifies the new/upcoming measurement based on the class of one of its nearest neighbours. In other words, the algorithm implemented finds the similarity between the *training* measurements and the *new/test* measurements. 

**Following assumptions were made:**


![image](https://user-images.githubusercontent.com/42310216/146126953-d1cd908d-c112-4574-a56b-e481acb1350d.png)

*The algorithm had an accuracy of 98.66% in determining and classifying faults via frequency data*

### [General process to determine the health condition of a system]
![image](https://user-images.githubusercontent.com/42310216/146127184-1e7b19b3-8c33-488f-aa09-707bc10b4787.png)

## [Secton Three - Health Monitoring System (Wind Turbine Co.)] 
Multisensor signal estimation and health monitoring system was designed and implemented for a wind turbine manufacturing company with prior knowledge of the pitch angle being uniformly distributed in the range 0° <= w <= 30° and the sensor noise being normally distributed V ~ N(0,9). 

MMSE Estimator was implemented by taking Bayes' Theorem into consideration to predict the actual measured data (without noise) of wind turbine blades. 
Additionally, CUSUM two-sided test was also implemented to identifiy if the measured data exceeds the desired threshold and generate an alert. 


## [Report]
A report was generated highlighting important findings and critically evaluating the results. 
