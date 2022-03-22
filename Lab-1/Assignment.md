# Assignment on Lab 1

## Assignments

1. **Assignment 0:** Installation of the necessary softwares.
2. **Assignment 1:** Find the best `K value` for the given `diabetes dataset`.
3. **Assignment 2:**
   - Implement the `KNN` algorithm.
   - Implement the `KNN algorithm` for `2D data`.
   - Implemntation of `KD-Tree` will get higher marks.

---

### Assignmnent Hints for `Assignment 1`

#### Loading Dataset from Google Drive

```python
from google.colab import drive


# Mount Drive
drive.mount('/content/gdrive')

# Load the dataset
data = pd.read_csv(
    '/content/gdrive/MyDrive/diabetes.csv',
    encoding='utf-8',
    engine='python'
)
data.head()
```

#### Feature Selection

```python
feature_columns = [
    'Pregnancies',
    'Insulin',
    'BMI',
    'Age',
    'Glucose',
    'BloodPressure',
    'DiabetesPedigreeFunction',
]

x = data[feature_columns]
y = data['Outcome']
```
