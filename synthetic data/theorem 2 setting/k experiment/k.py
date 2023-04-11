import numpy as np
from numpy.random import multivariate_normal, normal, choice, shuffle
import time
from scipy.optimize import leastsq
from numpy.random import rand 

def ReLU(x):
  return x * (x > 0)

def dReLU(x):
  return 1. * (x > 0)

def forward(x,w,beta):
  y = []
  for i in range(x.shape[0]):
    x_i = x[i].reshape(m,k)
    kernel = ReLU(x_i @ w)
    y_i = (kernel @ beta).sum()/np.sqrt(p)
    y += [y_i]
  return np.array(y)


def loss(x,w,beta,y_gt):
  y_pred = forward(x,w,beta)
  loss = (y_gt - y_pred)**2
  return loss.sum()/2


def dLdb(x,w,beta,y_gt,wj):                     # beta(t-1),w(t-1)               
  y_pred = forward(x,w,beta)
  first_term = y_gt - y_pred
  second_term = []
  for i in range(x.shape[0]):
    x_i = x[i].reshape(m,k)
    kernel = ReLU(x_i @ wj)
    second_term += [kernel.sum()]
  second_term = np.array(second_term)
  dldb = -np.multiply(first_term,second_term).sum()/np.sqrt(p)
  return dldb

def dLdW(x,w,beta,y_gt,wj,bj):                   # beta(t),w(t-1) 
  y_pred = forward(x,w,beta)
  first_term = np.expand_dims(y_gt - y_pred,axis=1)
  second_term = []  
  for i in range(x.shape[0]):
    x_i = x[i].reshape(m,k)
    dkernel = np.expand_dims(dReLU(x_i @ wj),axis=1)
    second_term += [np.sum(np.multiply(dkernel,x_i),axis=0)]
  second_term = np.array(second_term)
  dldw = -bj * np.sum(np.multiply(first_term,second_term),axis=0)/np.sqrt(p)
  return dldw

def gradient_descent(x,y_gt,maxiter):
  gamma = 0.01
  threshold = 1e-6
  w = multivariate_normal(w_mean, w_cov, p).transpose()
  beta = normal(beta_mean,beta_cov,p) 

  w_0 = np.copy(w)
  beta_0 = np.copy(beta)
  #loss_record = []

  for t in range(maxiter):
    #loss_record += [cur_loss]
    #if (t > 2) and (loss_record[-4] - loss_record[-3] < 0) and (loss_record[-3] - loss_record[-2] < 0) and (loss_record[-2] - loss_record[-1] < 0):     
      #break
    new_w = np.zeros(w.shape)
    new_beta = np.zeros(beta.shape)
    for j in range(p):
      new_beta[j] = beta[j] - gamma * dLdb(x,w,beta,y_gt,w[:,j])
    for j in range(p):
      new_w[:,j] = w[:,j] - gamma * dLdW(x,w,new_beta,y_gt,w[:,j],new_beta[j])/k
    if np.all(np.abs(beta - new_beta) <= threshold) and np.all(np.abs(w - new_w) <= threshold):
      break
    w = new_w
    beta = new_beta
  return w,beta,w_0,beta_0

def retrain_beta(x,y_gt,maxiter,w_tmax):
  gamma = 0.01
  threshold = 1e-6
  beta = np.zeros((p,))

  beta_0 = np.copy(beta)

  for t in range(maxiter):
    new_beta = np.zeros(beta.shape)
    for j in range(p):
      new_beta[j] = beta[j] - gamma * dLdb(x,w_tmax,beta,y_gt,w_tmax[:,j])
    if np.all(np.abs(beta - new_beta) <= threshold):
      break
    beta = new_beta
  return beta,beta_0

def compute_averageL2(para_rpa,para_tmax):
  error = (para_rpa - para_tmax)**2
  return error.sum()/para_rpa.size

def construct_hidden_designmat(x):
  design_mat = []
  for i in range(m):
    if len(design_mat) == 0:  
      design_mat = x[:,k*i:k*(i+1)] 
    else:
      design_mat = np.concatenate((design_mat,x[:,k*i:k*(i+1)]),axis=0)
  return design_mat

def output_designmat(x,w_rpa):
  design_matrix = []
  for i in range(x.shape[0]):
    x_i = x[i].reshape(m,k)
    kernel = ReLU(x_i @ w_rpa)
    if len(design_matrix) == 0:
      design_matrix = jnp.sum(kernel,axis = 0)
    else:
      design_matrix = jnp.column_stack((design_matrix,np.sum(kernel,axis = 0)))
  return design_matrix

def construct_output_designmat(x,w_rpa):
  design_matrix = []
  for i in range(x.shape[0]):
    x_i = x[i].reshape(m,k)
    kernel = ReLU(x_i @ w_rpa)
    if len(design_matrix) == 0:
      design_matrix = np.sum(kernel,axis = 0)
    else:
      design_matrix = np.column_stack((design_matrix,np.sum(kernel,axis = 0)))
  return design_matrix

def objective_w_norm1(vj,thetaj,wj0,design_mat):
  first_term = thetaj - wj0
  second_term = design_mat.transpose() @ vj
  return np.sqrt(np.absolute(first_term - second_term))

def objective_b_norm1(u,eta,b0,w_rpa,x):
  first_term = eta - b0
  second_term = construct_output_designmat(x,w_rpa) @ u
  return np.sqrt(np.absolute(first_term - second_term))

def add_noise(w_cln,b_cln,epislon):
  noise_mean = 1
  noise_var = 1
  w_noise = np.zeros(w_cln.shape)
  b_noise = np.zeros(b_cln.shape)

  for j in range(p):
    w_noise[:,j] = choice([0,1],k,p=[1 - epislon,epislon])
  
  b_noise = choice([0,1],p,p=[1 - epislon,epislon])

  w_noise = np.multiply(w_noise,normal(noise_mean,noise_var,(k,p)))
  b_noise = np.multiply(b_noise,normal(noise_mean,noise_var,p))

  w_ctm = w_cln + w_noise
  b_ctm = b_cln + b_noise

  return b_ctm,w_ctm

def model_repair(x,w_ctm,b_ctm,design_mat,w_0,beta_0):
  w_rpa = np.zeros(w_ctm.shape)
  b_rpa = np.zeros(b_ctm.shape)

  for j in range(p):
    vj0 = np.zeros((m*n,))
    res_x, success = leastsq(objective_w_norm1,vj0,args=(w_ctm[:,j],w_0[:,j],design_mat))
    w_rpa[:,j] = w_0[:,j] + design_mat.transpose() @ res_x 
    
  u0 = np.zeros((n,))
  res_x, success = leastsq(objective_b_norm1,u0,args=(b_ctm,beta_0,w_rpa,x))
  b_rpa = beta_0 + construct_output_designmat(x,w_rpa) @ res_x
  
  return b_rpa,w_rpa

def experiment_rcv_wb_gen():           
  avg_werror = np.zeros((num_trials,len(epislon_list)))
  avg_berror = np.zeros((num_trials,len(epislon_list)))

  for trial in range(num_trials):
    print('trial ',trial)
    x = multivariate_normal(input_mean, input_cov, n)         # n * d
    y_gt = rand(n)
    w_tmax,beta_tmax,w_0,beta_0 = gradient_descent(x,y_gt,maxiter)

    werror_onetrail = []
    berror_onetrail = []
    for epislon in epislon_list:
      b_ctm,w_ctm = add_noise(w_tmax,beta_tmax,epislon)
      design_mat = construct_hidden_designmat(x) 
      b_rpa,w_rpa = model_repair(x,w_ctm,b_ctm,design_mat,w_0,beta_0)       
      werror = compute_averageL2(w_rpa,w_tmax)
      berror = compute_averageL2(b_rpa,beta_tmax)
      werror_onetrail += [werror]
      berror_onetrail += [berror]
    avg_werror[trial] = werror_onetrail
    avg_berror[trial] = berror_onetrail

  return np.average(avg_werror,axis=0),np.average(avg_berror,axis=0)


n =  5                       # number of input data 
m =  5                       # number of partitions
p_list = [100,200,300,400,500]       # number of neurons at first layer       p bigger, error smaller
k_list = [50,100,200]
epislon_list = np.linspace(0,1,51)   # percentage of contamination
maxiter = 1000                # max iteration for gradient descent
num_trials = 100               # number of trials 

for p in p_list:
  for k in k_list:                # P_ix dimension || w dimension
    d =  int(m * k)              # x dimension 
    input_mean = np.zeros((d,))
    input_cov = np.eye(d)
    w_mean = np.zeros((k,))
    w_cov = np.eye(k)/k
    beta_mean = 0
    beta_cov = 1
  
    start_time = time.time()
    avg_werror,avg_berror = experiment_rcv_wb_gen()
    processing_time = time.time()-start_time
  
    print("current p")
    print(p)
    print("current k")
    print(k)
    print("total time")
    print(processing_time)
    print("avg_werror")
    print(avg_werror)
    print("avg_berror")
    print(avg_berror)