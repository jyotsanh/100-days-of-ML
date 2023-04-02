import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


datsets = load_diabetes()

x = datsets.data
y = datsets.target


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

class Multiple_Regression:
    def __init__(self):
        self.bias = 0
        self.weights = 0
        self.betas = 0
    def fit(self,x_train,y_train):

        x_train = np.insert(x_train,0,1,axis=1)

        self.betas = np.linalg.inv(np.dot(x_train.T,x_train)).dot(x_train.T).dot(y_train)
        self.bias = self.betas[0]
        self.weights = self.betas[1:]

    def predict(self,x_test):

        x_test = np.insert(x_test,0,1,axis=1)

        y_pred = np.dot(x_test,self.betas)
        return y_pred
    


model = LinearRegression()
lr = Multiple_Regression()

model.fit(x_train,y_train)
lr.fit(x_train,y_train)

model_y_pred = model.predict(x_test)
lr_y_pred = lr.predict(x_test)

print("sklearn accuracy :",r2_score(y_test,model_y_pred))
print("My model accuracy :",r2_score(y_test,lr_y_pred))
