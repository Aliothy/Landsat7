import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def data2s(data):
    format_data = '%H:%M:%S.%f'
    data = datetime.strptime(data,format_data)
    datasec = 3600*(data.hour+24*(data.hour<14))+60*data.minute+data.second
    return datasec

def tetaphi2cartesian(teta,phi,teta0):
    x = np.cos(phi)*np.cos(teta+teta0)
    y = np.cos(phi)*np.sin(teta+teta0)
    z = np.sin(phi)
    return np.array([x,y,z])

def rotation_m(alpha,beta,gamma,n):
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cc = np.cos(gamma)
    sc = np.sin(gamma)
    if n==0:
        R_0s = np.array([[ca*cc-sa*cb*sc,-ca*sc-sa*cb*cc,sa*sb],
                         [sa*cc+ca*cb*sc,-sa*sc+ca*cb*cc,-ca*sb],[sb*sc,sb*cc,cb]])
    elif n==1:
        R_0s = np.array([[-sa*cc-ca*cb*sc,sa*sc-ca*cb*cc,ca*sb],
                         [ca*cc-sa*cb*sc,-ca*sc-sa*cb*cc,sa*sb],[0,0,0]])
    elif n==2:
        R_0s = np.array([[sa*sb*sc,sa*sb*cc,sa*cb],[-ca*sb*sc,-ca*sb*cc,-ca*cb],
                         [cb*sc,cb*cc,-sb]])
    elif n==3:
        R_0s = np.array([[-ca*sc-sa*cb*cc,-ca*cc+sa*cb*sc,0],
                         [-sa*sc+ca*cb*cc,-sa*cc-ca*cb*sc,0],[sb*cc,-sb*sc,0]])
    return R_0s

def alphabetagamma2cartesian(alpha,beta,gamma,delta):
    R_0s = rotation_m(alpha,beta,gamma,0)
    cart = np.dot(R_0s,np.array([np.cos(delta),np.sin(delta),0]))
    return cart[0],cart[1],cart[2]

def cartesian2tetaphi(x,y,z):
    teta = np.arctan2(y,x)
    phi = np.arctan(z*np.sin(teta)/y)
    return teta,phi

class ExtKalman(object):

    def __init__(self,Rqq,Rvv,data):
        
        self.x = []
        self.P = []
        self.Rqq = Rqq
        self.Rvv = Rvv
        self.t = 0
        self.data = data
        self.output = []
        self.delta_r = []

    def initialization(self):
        
        pos1 = self.data[0,:3]
        pos2 = self.data[1,:3]
        pos3 = self.data[2,:3]
        i_s = pos1/np.linalg.norm(pos1)
        i1 = pos2/np.linalg.norm(pos2)
        i2 = pos3/np.linalg.norm(pos3)
        k_s = np.cross(i_s,i1)
        k_s = k_s/np.linalg.norm(k_s)
        k_s2 = np.cross(i_s,i2)
        k_s2 = k_s2/np.linalg.norm(k_s2) 
        j_s = np.cross(k_s,i_s)
        j_s = j_s/np.linalg.norm(j_s)
        j_s2 = np.cross(k_s2,i_s)
        j_s2 = j_s2/np.linalg.norm(j_s2)
        R_s0 = np.array([i_s,j_s,k_s])
        R_s02 = np.array([i_s,j_s2,k_s2])
        R_0s = np.transpose(R_s0)
        R_0s2 = np.transpose(R_s02)

        alpha = np.arctan2(R_0s[0,2],-R_0s[1,2])
        beta = np.arctan2(np.sin(alpha)*R_0s[0,2]-np.cos(alpha)*R_0s[1,2],R_0s[2,2])
        gamma = np.arctan2(-np.cos(alpha)*R_0s[0,1]-np.sin(alpha)*R_0s[1,1],np.cos(alpha)*R_0s[0,0]+np.sin(alpha)*R_0s[1,0])
        i1_s = np.dot(R_s0,i1)
        delta = np.arccos(i1_s[0])
        delta_dot = delta/self.data[1,4]

        alpha2 = np.arctan2(R_0s2[0,2],-R_0s2[1,2])
        beta2 = np.arctan2(np.sin(alpha2)*R_0s2[0,2]-np.cos(alpha2)*R_0s2[1,2],R_0s2[2,2])
        gamma2 = np.arctan2(-np.cos(alpha2)*R_0s2[0,1]-np.sin(alpha2)*R_0s2[1,1],np.cos(alpha2)*R_0s2[0,0]+np.sin(alpha2)*R_0s2[1,0])
        i1_s = np.dot(R_s02,i2)
        delta2 = np.arccos(i1_s[0])
        delta_dot2 = delta2/self.data[1,4]
        self.x = np.array([alpha,beta,gamma,delta,delta_dot])
        err = np.array([alpha-alpha2,beta-beta2,gamma-gamma2,delta-delta2,delta_dot-delta_dot2])
        self.P = np.outer(err,err) 

    def normalize(self):
        self.x = self.x%(2*np.pi)   

    def F_mat(self,dt):

        F = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,dt],[0,0,0,0,1]])
        return F
    
    def H_mat(self):
        
        alpha = self.x[0]
        beta = self.x[1]
        gamma = self.x[2]
        delta = self.x[3]
        R_0s = rotation_m(alpha,beta,gamma,0)
        R_a = rotation_m(alpha,beta,gamma,1)
        R_b = rotation_m(alpha,beta,gamma,2)
        R_c = rotation_m(alpha,beta,gamma,3)
        y_s = np.array([np.cos(delta),np.sin(delta),0])
        y_dots = np.array([-np.sin(delta),np.cos(delta),0])
        H = np.transpose(np.array([np.dot(R_a,y_s),np.dot(R_b,y_s),np.dot(R_c,y_s),np.dot(R_0s,y_dots),[0,0,0]]))
        
        return H


    def out_model(self):
        alpha = self.x[0]
        beta = self.x[1]
        gamma = self.x[2]
        delta = self.x[3]
        x,y,z = alphabetagamma2cartesian(alpha,beta,gamma,delta)
        y_est = np.array([x,y,z])
        return y_est

    def update(self):
        
        self.initialization()
        self.data = self.data[1:,:]
        for row in self.data:
            
            self.t = row[3]
            H = self.H_mat()
            F = self.F_mat(row[4])
            error = row[:3]-self.out_model()
            R_ee = self.Rvv+np.dot(np.dot(H,self.P),np.transpose(H))
            Kp = np.transpose(np.linalg.solve(np.transpose(R_ee),np.transpose(np.dot(np.dot(F,self.P),np.transpose(H)))))
            #update x and P
            self.x = np.dot(F,self.x)+np.dot(Kp,error)
            self.P = np.dot(np.dot(F,self.P),np.transpose(F))+self.Rqq-np.dot(np.dot(Kp,R_ee),np.transpose(Kp))
            self.normalize()
            self.output.append(self.x)
            R_s0 = rotation_m(self.x[0],self.x[1],self.x[2],0).transpose()
            pos_delta = np.dot(R_s0,row[:3])
            self.delta_r.append(np.arctan2(pos_delta[1],pos_delta[0]))
            

# orbital parameters
T_1day = 23*3600+56*60+4.09
wt = 2*np.pi/T_1day

# data extraction
doc = np.array(pd.read_csv('ls7_data_5_10_2015.csv'))
for row in doc:
    row[2]=data2s(str(row[2][:15]))
doc[:,2] = doc[:,2]-doc[0,2]
deltat = np.array(doc[1:,2]-doc[:-1,2])
deltat = np.append(np.array([0]),deltat).reshape(len(doc),1)
doc[:,:2]=doc[:,:2]*np.pi/180
doc = np.append(doc,deltat,axis=1)
y_real = np.zeros((len(doc),5))
for i in range(len(doc)):
    y_real[i,:3] = tetaphi2cartesian(doc[i,0],doc[i,1],wt*doc[i,2])
y_real[:,3:] = doc[:,2:]
print(y_real.shape)


#Kalman parameters
Rvv = 1e-4*np.eye(3)
Rqq = np.diag([1e-10,1e-10,1e-10,1e-5,5e-1])
Kalman = ExtKalman(Rqq,Rvv,y_real)
Kalman.update()

out = np.array(Kalman.output)
print(len(out))
app = np.zeros((len(out),3))
tetaphi = np.zeros((len(out),2))
for i in np.arange(len(app)):
    x,y,z = alphabetagamma2cartesian(out[i,0],out[i,1],out[i,2],out[i,3])
    app[i,:] = [x,y,z]
    tetaphi[i,:] = cartesian2tetaphi(x,y,z)

tetaphi[:,0] = ((tetaphi[:,0]-wt*doc[1:,2])%(2*np.pi))
tetaphi[:,0] = np.where(tetaphi[:,0]>np.pi,tetaphi[:,0]-2*np.pi,tetaphi[:,0])
tetaphi = tetaphi*180/np.pi
misrep = (out[:,3]-np.array(Kalman.delta_r))%(np.pi)


#plot
#fig1 correctness of the orbit
fig1 = plt.figure(1)
ax = plt.axes(projection='3d')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(y_real[:,0],y_real[:,1],y_real[:,2],color='red',s=3)
ax.scatter(app[:,0],app[:,1],app[:,2],color='green')
ax.legend(["real orbit data","estimated data"])

#fig2 errors
fig2 = plt.figure(2)
plt.subplot(2,2,1)
plt.scatter(doc[1:,2],misrep,s=5)
plt.title('variations on $\delta$')
plt.xlabel('time')
plt.ylabel('$\delta-\hat{\delta}$')
plt.subplot(2,2,2)
plt.scatter(doc[1:,2],y_real[1:,0]-app[:,0],s=5)
plt.title('variations on $x$')
plt.xlabel('time')
plt.ylabel('$x-\hat{x}$')
plt.subplot(2,2,3)
plt.scatter(doc[1:,2],y_real[1:,1]-app[:,1],s=5)
plt.title('variations on $y$')
plt.xlabel('time')
plt.ylabel('$y-\hat{y}$')
plt.subplot(2,2,4)
plt.scatter(doc[1:,2],y_real[1:,2]-app[:,2],s=5)
plt.title('variations on $z$')
plt.xlabel('time')
plt.ylabel('$z-\hat{z}$')
plt.tight_layout()

#fig3 Spherical coordinates
fig3 = plt.figure(3)
ax = plt.axes()
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.set_xlabel(' longitude $\theta$ ')
ax.set_ylabel(' latitude  $ \phi $')
ax.scatter(doc[:,0]*180/np.pi,doc[:,1]*180/np.pi,color='red',s=5)
ax.scatter(tetaphi[:,0],tetaphi[:,1],color='green',s=5)
ax.legend(["real orbit data","estimated data"])

plt.show()