# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

To build a neural network regression model for predicting a continuous target variable, we will follow a systematic approach. The process includes loading and pre-processing the dataset by addressing missing data, scaling the features, and ensuring proper input-output mapping. Then, a neural network model will be constructed, incorporating multiple layers designed to capture intricate patterns within the dataset. We will train this model, monitoring performance using metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE), to ensure accurate predictions. After training, the model will be validated and tested on unseen data to confirm its generalization ability. The final objective is to derive actionable insights from the data, helping to improve decision-making and better understand the dynamics of the target variable. Additionally, the model will be fine-tuned to enhance performance, and hyperparameter optimization will be carried out to further improve predictive accuracy. The resulting model is expected to provide a robust framework for making precise predictions and facilitating in-depth data analysis.

## Neural Network Model

![Screenshot 2024-09-01 174737](https://github.com/user-attachments/assets/4090218d-9a17-4186-af2a-6aae2d72f4c6)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: POOJA S
### Register Number: 212223040146
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


worksheet = gc.open('DEEP LEARNING').sheet1


data = worksheet.get_all_values()


df = pd.DataFrame(data[1:], columns=data[0])
df = df.astype({'INPUT':'float'})
df= df.astype({'OUTPUT':'float'})
df.head()

X=df[['INPUT']].values
y=df[['OUTPUT']].values

X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=33)

Scaler=MinMaxScaler()

Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)

ai_brain = Sequential({
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
})

ai_brain.compile(optimizer = 'rmsprop', loss='mse')
 ai_brain.fit(X_train1,y_train,epochs =70)

loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()

X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)

X_n1 = [[1]]

X_n1_5 = Scaler.transform(X_n1)

ai_brain.predict(X_n1_5)



```
## Dataset Information

![Screenshot 2024-09-01 172235](https://github.com/user-attachments/assets/520efc8b-ca3b-4393-96a3-275d6c5126e7)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-09-01 173145](https://github.com/user-attachments/assets/e6282dab-6d8b-4328-911c-c7e163db4845)


### Test Data Root Mean Squared Error

![Screenshot 2024-09-01 173644](https://github.com/user-attachments/assets/5ccc6ac8-db22-4684-8863-37c51acf4813)


### New Sample Data Prediction

![Screenshot 2024-09-01 173549](https://github.com/user-attachments/assets/330c811f-160c-44f6-8447-84be6b1ca71a)


## RESULT

To develop a neural network regression model for the given dataset is created sucessfully.
