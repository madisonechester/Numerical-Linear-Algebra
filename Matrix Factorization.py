# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:00:14 2023

@author: mecheste
"""

import numpy as np 

def lunopiv(A,n,ptol):
    for k in range(0,n-1):
        pivot = A[k,k]
        if abs(pivot) < ptol:
            print('zero pivot encountered')
            break 
        for i in range (k+1,n):
            A[i,j] = A[i,k]/pivot
            for j in range(k+1,n):
                A[i,j] = A[i,j] - A[i,k]*A[j,k]
    L = np.eye(n) + np.tril(A,-1)
    U = np.triu(A)
    return L,U



n = 5;
A = np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        A[i,j] = np.random.nromal(0,10)
        
LU = np.zeros((n,n))

r = 3;
nblocks = int(n/r)
Aij = np.zeros((nblocks,nblocks),np.matrix)
S = np.copy(A)

for i in range(0,nblocks - 1):
    Aij[0,0] = np.copy(S[0:r,0:r]);
    Aij[0,1] = np.copy(S[0:r,r:n]);
    Aij[1,0] = np.copy(S[r:n,0:r]);
    Aij[1,1] = np.copy(S[r:n,r:n]);
    L11,U11 = lunopiv(Aij[0,0],r,1.e-16)
    
    print('%d %24.16e\n' % (i,np.linalg.norm(np.dotL11,U11)-S[0:r,0:r].np.inf))
    U12 = np.linalg.solve(L11,Aij[0,1])
    L21t = np.linalg.solve(U11.T,Aij[1,0].T)
    
    LU[i*r:(i+1)*r,i*r:(i+1)*r] = np.copy(np.tril(L11,-1)+U11);
    LU[i*r:(i+1)*r,(i+1)*r] = np.copy(U12);
    LU[(i+1)*r:n,i*r:(i+1)*r] = np.copy(L21t.T);
    
    S = Aij[1,1] - np.dot(L21t.T,U12)
    
Aij[0,0] = S[0:r,0:r]
L11,U11 = lunopiv(Aij[0,0],r,1.e-16)
LU[n-r:n,n-r:n] = np.copy(np.tril(L11,-1) + U11);
 
print('LU factorization %d\n' % (r))
  