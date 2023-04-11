import numpy as np
from numpy.random import multivariate_normal, normal, choice, shuffle
import time
from scipy.optimize import leastsq
from numpy.random import rand, randint 
import mnist
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from jax.numpy.linalg import norm
from jax import jit,value_and_grad
import jax.numpy as jnp

def ReLU(x):
  return x * (x > 0)

def dReLU(x):
  return 1. * (x > 0)

def forward(x,w,beta):
  y = []
  for i in range(x.shape[0]):
    x_i = x[i].reshape(m,k)
    #print('x_i ',x_i)
    kernel = ReLU(x_i @ w)
    #print('w ',w)
    #print('kernel ',kernel)
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
  gamma = 0.001
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
    if t == maxiter-1:
      print('gradient_descent max iteration reached')      
    w = new_w
    beta = new_beta
  return w,beta,w_0,beta_0

def retrain_beta(x,y_gt,maxiter,w_tmax):
  gamma = 0.001
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
'''
def objective_w(vj,thetaj,wj0,design_mat):
  first_term = thetaj - wj0
  second_term = design_mat.transpose() @ vj
  return jnp.sum(jnp.abs(first_term - second_term))

def objective_b(u,eta,b0,w_rpa,x):
  first_term = eta - b0
  second_term = output_designmat(x,w_rpa) @ u
  return jnp.sum(jnp.abs(first_term - second_term))
'''
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
    '''
    obj_and_grad = jit(value_and_grad(objective_w))
    res = minimize(obj_and_grad,vj0,args=(w_ctm[:,j],w_0[:,j],design_mat),jac=True)
    w_rpa[:,j] = w_0[:,j] + design_mat.transpose() @ res.x
    '''
  u0 = np.zeros((n,))
  res_x, success = leastsq(objective_b_norm1,u0,args=(b_ctm,beta_0,w_rpa,x))
  b_rpa = beta_0 + construct_output_designmat(x,w_rpa) @ res_x
  '''
  obj_and_grad = jit(value_and_grad(objective_b))
  res = minimize(obj_and_grad,u0,args=(b_ctm,beta_0,w_rpa,x),jac=True)
  b_rpa = beta_0 + construct_output_designmat(x,w_rpa) @ res.x
  '''
  return b_rpa,w_rpa

def experiment_rcv_wb_gen():           
  avg_werror = np.zeros((num_trials,len(epislon_list)))
  avg_berror = np.zeros((num_trials,len(epislon_list)))
  avg_acc_score = np.zeros((num_trials,))
  dataset_images = mnist.train_images()
  dataset_labels = mnist.train_labels()
  test_images = mnist.test_images()
  test_labels = mnist.test_labels()

  for trial in range(num_trials):
    print(trial)
    zero_index = np.where(dataset_labels == 0)[0].astype(int)
    one_index = np.where(dataset_labels == 1)[0].astype(int)
    two_index = np.where(dataset_labels == 2)[0].astype(int)
    zero_index_idx = choice(len(zero_index),66,replace=False)
    one_index_idx = choice(len(one_index),66,replace=False)
    two_index_idx = choice(len(two_index),66,replace=False)
    chosen_zero = dataset_images[zero_index[zero_index_idx[:33]]]
    chosen_one = dataset_images[one_index[one_index_idx[:33]]]
    chosen_two = dataset_images[two_index[two_index_idx[:33]]]
    chosen_input = np.concatenate((chosen_zero,chosen_one,chosen_two)).reshape(99,-1)
    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(chosen_input)                   # training_examples (=99) * d      
    #w_gt = multivariate_normal(w_mean, w_cov, p).transpose()      # k * p
    #beta_gt = normal(beta_mean,beta_cov,p)                # p
    zeros = np.zeros(33)
    ones = np.ones(33)
    twos = np.ones(33) * 2
    y_gt = np.concatenate((zeros,ones,twos))
    w_tmax,beta_tmax,w_0,beta_0 = gradient_descent(x,y_gt,maxiter)

    test_zero_index = np.where(test_labels == 0)[0].astype(int)
    test_one_index = np.where(test_labels == 1)[0].astype(int)
    test_two_index = np.where(test_labels == 2)[0].astype(int)   
    test_chosen_zero = test_images[test_zero_index]
    test_chosen_one = test_images[test_one_index]
    test_chosen_two = test_images[test_two_index]
    test_input = np.concatenate((test_chosen_zero,test_chosen_one,test_chosen_two)).reshape(-1,28*28)
    x_test = scaler.transform(test_input)

    test_zeros = np.zeros(len(test_chosen_zero))
    test_ones = np.ones(len(test_chosen_one))
    test_twos = np.ones(len(test_chosen_two)) * 2    
    test_y_gt = np.concatenate((test_zeros,test_ones,test_twos))
    
    y_predict = forward(x_test,w_tmax,beta_tmax)
    acc_score = accuracy_score(test_y_gt,np.rint(y_predict))
    print('accuracy_score : ',acc_score)
    avg_acc_score[trial] = acc_score

    werror_onetrail = []
    berror_onetrail = []
    chosen_zero = dataset_images[zero_index[zero_index_idx[33:]]]
    chosen_one = dataset_images[one_index[one_index_idx[33:]]]
    chosen_two = dataset_images[two_index[two_index_idx[33:]]]
    chosen_input = np.concatenate((chosen_zero,chosen_one,chosen_two)).reshape(99,-1)
    x = scaler.transform(chosen_input)
    shuffle(x)
    x_part = x[:n,:]
    for epislon in epislon_list:
      print(trial,' ',epislon)
      b_ctm,w_ctm = add_noise(w_tmax,beta_tmax,epislon)
      design_mat = construct_hidden_designmat(x_part) 
      b_rpa,w_rpa = model_repair(x_part,w_ctm,b_ctm,design_mat,w_0,beta_0)       
      werror = compute_averageL2(w_rpa,w_tmax)
      berror = compute_averageL2(b_rpa,beta_tmax)
      werror_onetrail += [werror]
      berror_onetrail += [berror]
    avg_werror[trial] = werror_onetrail
    avg_berror[trial] = berror_onetrail

  return np.average(avg_werror,axis=0),np.average(avg_berror,axis=0),np.average(avg_acc_score,axis=0)


n_list =  [20,40,60,80,99]                                  # design matrix first dimension
p = 300       # number of neurons at first layer       p bigger, error smaller
epislon_list = np.linspace(0,1,21)   # percentage of contamination
maxiter = 2000                # max iteration for gradient descent
num_trials = 10               # number of trials 

for n in n_list:
  m = 2
  k = 392
  d =  int(m * k)              # x dimension 
  w_mean = np.zeros((k,))
  w_cov = np.eye(k)/k
  beta_mean = 0
  beta_cov = 1
  
  start_time = time.time()
  avg_werror,avg_berror,avg_acc_score = experiment_rcv_wb_gen()
  processing_time = time.time()-start_time

  print("current n")
  print(n)
  print("total time")
  print(processing_time)
  print("avg_werror")
  print(avg_werror)
  print("avg_berror")
  print(avg_berror)
  print("avg_acc_score")
  print(avg_acc_score)