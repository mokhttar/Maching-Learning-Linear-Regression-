import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#open csv file and extract hp (hourse pwoer) and mpg(mile par galone)
file=pd.read_csv('mtcars(mpg_hp).csv')
#extract hp and mpg columns from the data set
hp=file['hp']
mpg=file['mpg']
#how many point do we have (columns)
N=len(file)
def CalculatError(a,b,x,y,N):
    """function to calculate the error in the lecture is : sum[y(i)-a*x+b]**2 /N"""
    Error=0
    for  i in range(N) :
      Error += (y[i] - (a * x[i] + b))**2 / N
    print(f'The total square error is {Error}')
def leanearRegression(x,y):
    """this function will give us the a and b for the best  fiting line"""
    #first Step Calculating a
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    sloup_result = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x ** 2) - np.mean(x) ** 2)
    b_result=y_mean-sloup_result*x_mean
    return sloup_result,b_result
best_sloup,best_b=leanearRegression(hp,mpg)
CalculatError(best_sloup,best_b,hp,mpg,N)
plt.scatter(hp, mpg, color='blue', label='Data Points')
plt.plot(hp, best_sloup * hp + best_b, color='red', label='Best Fit Line')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Gallon (mpg)')
plt.title('Best Fit Line for Horsepower vs Miles per Gallon')
plt.legend()
plt.show()

