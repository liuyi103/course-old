from FuncDesigner import *
from openopt import MILP
import numpy as np
import math
import copy
n=10
k=10
m=10
beta=1.0/k<1.0/n and 1.0/k or 1.0/n
b=np.random.rand(n)*beta+1
v=np.random.rand(n,m)
cov=np.random.rand(n,m,m)
p=[0.3]*m
q=np.random.randint(3,10,size=m)
xs=[[0 for j in range(m)]for i in range(n)]
alpha=lambda:math.sqrt(sum([(\
                             np.max([sum([xs[i][j] for i in range(n)])-q[j],(p[j]==0 and 0 or -1e6)])\
                             )**2 for j in range(m)]))
best=alpha()
bestnode=copy.deepcopy(p)
def reqgen(m):
    ans=[]
    for i in range(3):
        req=np.random.randint(low=0,high=2,size=m)
        if sum(req)<=1:
            continue
        ans+=[(req,np.random.randint(low=1,high=sum(req)))]
    return ans
req=[reqgen(m) for i in range(n)]
s=np.random.randint(low=3,high=8,size=m)
def input():
    pass
input()
def opt(i):
    x,xx=oovars('x xx',domain=bin)
    startpoint={x:[0]*m,xx:[0]*(m*m)}
    constraints=[sum([x[j]*p[j]for j in range(m)])<=b[i]]+[\
                                                             (sum(x[k]for k in range(m) if req[i][j][0][k])<=req[i][j][1])\
                                                             for j in range(len(req[i]))]\
                                                             +[xx[j*m+k]*2<=x[j]+x[k] for j in range(m) for k in range(m)]\
                                                             +[xx[j*m+k]*2>=x[j]+x[k]-1 for j in range(m) for k in range(m)]
    obj=sum([v[i,j]*x[j] for j in range(m)])+sum(xx[j*m+k]*cov[i,j,k] for j in range(m) for k in range(m))
    prob=MILP(objective=obj, startPoint=startpoint, constraints=constraints)
    r=prob.solve('ralg')
    return [r(x[j])for j in range(m)]
round=0
tabu=[]
curnode=copy.deepcopy(p)
def score(node):
    p=node
    xs=[opt(i) for i in range(n)]
    return alpha()
def getnei(node):
    score(node)
    ans=[]
    grad=np.array([np.max([sum([xs[i][j] for i in range(n)])-q[j],(p[j]==0 and 0 or -1e6)]) for j in range(m)])
    node=np.array(node)
    k=0.01
    while min(node+k*grad)>=0:
        ans+=[node+k*grad]
        k*=2
    for i in range(m):
        if sum([xs[i][j] for i in range(n)])-q[j]<0:
            tmp=copy.deepcopy(node)
            tmp[i]=0
            ans+=[tmp]
        if sum([xs[i][j] for i in range(n)])-q[j]>0:
            tmp=copy.deepcopy(node)
            
while score(bestnode)>1:
    tabu+=[curnode]
    nei=getnei(curnode)
    while nei[-1] in tabu:
        nei.pop()
    curnode=nei[-1]
    if score(nei[-1])<score(bestnode):
        bestnode=copy.deepcopy(nei[-1])
