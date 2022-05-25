--linear Regression--

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Hours.csv")
print(data)

x=data.iloc[:,:-1].values.reshape(-1, 1) 
y=data.iloc[:,1].values.reshape(-1, 1) 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
a=reg.fit(x,y)
b=reg.score(x,y)*100
c=reg.intercept_
gragh=plt.scatter(x,y,color='red')

plt.plot(x,reg.predict(x),color='blue')
plt.show



-----KNN---

import numpy as np
import pandas as pd
data=pd.read_csv("kdata.csv")
print(data)

x=data.iloc[:,:-1].values
y=data.iloc[:,2].values
from sklearn.neighbors import KNeighborsClassifier
kn= KNeighborsClassifier(n_neighbors=3)
kn.fit(x,y)
x_test=np.array([6,6])
y_test=kn.predict([x_test])
print(y_test)

kn= KNeighborsClassifier(n_neighbors=3,weights='distance')
kn.fit(x,y)
x_test=np.array([6,2])
y_test=kn.predict([x_test])
print(y_test)


---sales--

import numpy as np
import pandas as pd
data=pd.read_csv("sales.csv")
print(data)

x=data.iloc[:,:-1]
y=data.iloc[:,5].values
print(x)
print(y)

from sklearn.preprocessing import LabelEncoder
lable = LabelEncoder
x=x.apply(LabelEncoder().fit_transform)
print(x)

from sklearn.tree import DecisionTreeClassifier
reg = DecisionTreeClassifier()
reg.fit(x.iloc[:,1:5],y)
x_in = np.array([1,1,0,0])
y_pred=reg.predict([x_in])
print(y_pred)

#from sklearn.externals.six import StringIO
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data=StringIO()
export_graphviz(reg,out_file=dot_data,filled=True,rounded=True,special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')

from PIL import Image

image = Image.open('tree.png')
image.show()



-K means Clustering--

#import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.DataFrame({'X':[0.1,0.15,0.08,0.16,0.2,0.25,0.24,0.3],
                 'y':[0.6,0.71,0.9,0.85,0.3,0.5,0.1,0.2]})
f1 = df['X'].values
f2 = df['y'].values
X = np.array(list(zip(f1, f2)))
print(X)

#centroid points
C_x=np.array([0.1,0.3])
C_y=np.array([0.6,0.2])
centroids=C_x,C_y

#plot the given points
colmap = {1: 'r', 2: 'b'}
plt.scatter(f1, f2, color='k')
plt.show()

plt.scatter(C_x[0],C_y[0], color=colmap[1])
plt.scatter(C_x[1],C_y[1], color=colmap[2])
plt.show()

C = np.array(list((C_x, C_y)), dtype=np.float32)
print (C)

plt.scatter(f1, f2, c='#050505')
plt.scatter(C_x[0], C_y[0], marker='*', s=200, c='r')
plt.scatter(C_x[1], C_y[1], marker='*', s=200, c='b')
plt.show()
from sklearn.cluster import KMeans
model=KMeans(n_clusters=2,random_state=0)
model.fit(X)
labels=model.labels_
print(labels)

#using labels find population around centroid
count=0
for i in range(len(labels)):
    if (labels[i]==1):
        count=count+1

print('No of population around cluster 2:',count-1)

#Find new centroids
new_centroids = model.cluster_centers_

print('Previous value of m1 and m2 is:')
print('M1==',centroids[0])
print('M1==',centroids[1])

print('updated value of m1 and m2 is:')
print('M1==',new_centroids[0])
print('M1==',new_centroids[1])
