import numpy as np
from scipy import linalg as la
v = np.array([2,-1,4])
u = np.array([-2,1,5])
a = -2
b =1
A = np.array([[0,3,-1],[-1,4,-2],[1,3,1]])
B = np.array([[2,-1,2],[-1,0,1],[-1,2,2]])
#Question 1
print('L1 Norm = ',la.norm(v,1))
print('Question 1 = ',la.norm(v,1)*v+a*u)
#Question 2
Cosine_Theta = (np.dot(u,v)) / (la.norm(u,2)*la.norm(v,2))
print('Dot Product = ',np.dot(u,v))
print('Cosine Theta = ',Cosine_Theta)
#Question3
a_A_dot_v = a*np.matmul(A,v)
print('a*(A dot v) = ',a_A_dot_v)
#Question4
Q4 = np.dot(A, B.transpose())+ B.trace()*la.tril(B)
print('Question 4 =', Q4)