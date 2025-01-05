# Decision Tree Classification for Play Tennis Dataset  
**With Jupyter Notebook**

This Jupyter Notebook demonstrates how to build a Decision Tree classifier to predict whether to play tennis based on weather conditions. The dataset used is `Play_tennis.csv`, which contains 14 entries with features like `outlook`, `temp`, `humidity`, and `wind`.

---

## **Table of Contents**
1. [Prerequisites](#prerequisites)
2. [Getting Started](#getting-started)
3. [Running the Code](#running-the-code)
4. [Code Explanation](#code-explanation)
5. [Results](#results)
6. [License](#license)

---

## **Prerequisites**
Before running the code, ensure you have the following installed:
- Python 3.x
- Required Python libraries:
  ```bash
  pip install numpy pandas scikit-learn matplotlib graphviz jupyter
  ```
- Jupyter Notebook (to run the `.ipynb` file).

---

## **Getting Started**
1. **Download the Dataset**  
   Ensure the dataset `Play_tennis.csv` is in the same directory as the notebook.

2. **Launch Jupyter Notebook**  
   Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Open the `.ipynb` file from the Jupyter Notebook interface.

---

## **Running the Code**
1. Open the `.ipynb` file in Jupyter Notebook.
2. Run each cell sequentially to execute the code.

---

## **Code Explanation**
### **1. Import Libraries**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
- Libraries used for data manipulation, visualization, and modeling.

### **2. Load and Explore Data**
```python
data = pd.read_csv('Play_tennis.csv')
data.head()
data.describe()
data.info()
```
- Load the dataset and explore its structure, summary statistics, and data types.

### **3. Data Preprocessing**
```python
data = data.drop(columns='day')
```
- Drop the `day` column as it is not relevant for classification.

### **4. Data Visualization**
```python
data['play'].value_counts().plot(kind='bar')
data['outlook'].value_counts().plot(kind='bar')
```
- Visualize the distribution of the target variable (`play`) and features.

### **5. Entropy and Information Gain Calculation**
```python
def calculate_entropy(df, target, feature='None', value='None'):
    # Function to calculate entropy
    pass

IG_Outlook = calculate_entropy(df, 'play') - (
    (len(df[df['outlook'] == 'Sunny']) / len(df)) * calculate_entropy(df, 'play', 'outlook', 'Sunny') +
    (len(df[df['outlook'] == 'Overcast']) / len(df)) * calculate_entropy(df, 'play', 'outlook', 'Overcast') +
    (len(df[df['outlook'] == 'Rain']) / len(df)) * calculate_entropy(df, 'play', 'outlook', 'Rain')
)
```
- Calculate entropy and information gain for features.

### **6. Data Preparation**
```python
x = data.drop(columns='play')
y = data['play']

from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
data['outlook'] = Le.fit_transform(data['outlook'])
data['temp'] = Le.fit_transform(data['temp'])
data['humidity'] = Le.fit_transform(data['humidity'])
data['wind'] = Le.fit_transform(data['wind'])
data['play'] = Le.fit_transform(data['play'])
```
- Encode categorical variables into numerical values.

### **7. Train-Test Split**
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
```
- Split the data into training and testing sets.

### **8. Build and Train Model**
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)
```
- Train a Decision Tree classifier using entropy as the criterion.

### **9. Evaluate Model**
```python
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
- Evaluate the model using accuracy and classification report.

### **10. Visualize the Decision Tree**
```python
from sklearn.tree import plot_tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=['outlook', 'temp', 'humidity', 'wind'], class_names=['No', 'Yes'], filled=True)
plt.show()
```
- Visualize the Decision Tree structure.

---

## **Results**
- **Accuracy**: The model's accuracy on the test set.
- **Decision Tree Visualization**: A graphical representation of the Decision Tree.
- **Classification Report**: Precision, recall, and F1-score for each class.

---

## **License**
This project is open-source and available under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as needed.

---

## **Support**
If you encounter any issues or have questions, feel free to open an issue in this repository or contact me at [minthukywe2020.com](mailto:minthukywe2020.com).

---

Enjoy exploring the Decision Tree model in Jupyter Notebook! 🚀
