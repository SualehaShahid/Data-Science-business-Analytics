import pandas as pd
import requests
from io import StringIO
url = 'https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'

# Fetch data from the URL
response = requests.get(url)

if response.status_code == 200:
    data = response.text
else:
    print("Failed to fetch data from the URL.")
    exit()
    
# Create a pandas DataFrame
df = pd.read_csv(StringIO(data))

# Save data as a CSV file
df.to_csv('data.csv', index=False)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('data.csv')
data.head(12)
data.isnull==True

import seaborn as sns
sns.set_style('darkgrid')
sns.scatterplot(y=data['Scores'], x= data['Hours'])
plt.title('Percentage of Marks vs Study Hours',size= 15)
plt.ylabel('Percentage of Marks', size= 12)
plt.xlabel('Hours Studied', size=12)
plt.show()

sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression plot', size= 20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()

X = data[['Hours']].values  # Select the 'Hours' column as input
y = data['Scores'].values  # Select the 'Scores' column as output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

hours = [9.25]
answer = regressor.predict([hours])
print("Predicted Score= {}".format(round(answer[0],2)))
