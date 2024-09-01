# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

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

X_n1 = [[8]]

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
