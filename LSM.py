import matplotlib.pyplot as plt
import numpy as np

# 1차함수
a = 1.5
b = 0.5 

x = np.arange(0.0,5.0,0.1);
y = a*x+b;

dataset = np.concatenate([x.reshape((-1,1)),y.reshape((-1,1))],axis=-1);
dataset = dataset + np.random.uniform(-0.3,0.3,size=dataset.shape)

plt.plot(x,y,'r')
plt.scatter(dataset[:,0],dataset[:,1], marker='o')
plt.show();


An_list = [];
Bn_list = [];

for (x,y) in dataset:    
    An = [x, 1];
    Bn = [y];
    An_list.append(An);
    Bn_list.append(Bn);

A = np.array(An_list);
B = np.array(Bn_list);


A_inv = np.dot(np.linalg.pinv(np.dot(A.T,A)),A.T);
print(np.dot(A_inv,B));

a,b = np.dot(A_inv,B).T[0];

x = np.arange(0.0,5.0,0.1);
y = a*x+b;

plt.plot(x,y,'r')
plt.scatter(dataset[:,0],dataset[:,1], marker='o')
plt.show();

    
    
# 2차함수

a = 1.5
b = 0.5 
c = 0.1

x = np.arange(0.0,5.0,0.1);
y = a*(x*x)+b*x +c;

dataset = np.concatenate([x.reshape((-1,1)),y.reshape((-1,1))],axis=-1);
dataset = dataset + np.random.uniform(-0.3,0.3,size=dataset.shape)

plt.plot(x,y,'r')
plt.scatter(dataset[:,0],dataset[:,1], marker='o')
plt.show();


An_list = [];
Bn_list = [];

for (x,y) in dataset:    
    An = [x*x, x , 1];
    Bn = [y];
    An_list.append(An);
    Bn_list.append(Bn);

A = np.array(An_list);
B = np.array(Bn_list);


A_inv = np.dot(np.linalg.pinv(np.dot(A.T,A)),A.T);
print(np.dot(A_inv,B));

a,b,c = np.dot(A_inv,B).T[0];


x = np.arange(0.0,5.0,0.1);
y = a*(x*x)+b*x +c;


plt.plot(x,y,'r')
plt.scatter(dataset[:,0],dataset[:,1], marker='o')
plt.show();






