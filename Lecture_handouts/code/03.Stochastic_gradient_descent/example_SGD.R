rm(list=ls())
#
library(mvtnorm)
#library(rootSolve)
#
set.seed(2023)
#
fntsz <- 1.5
#
T <- 500
#
n_obs <- 10^4
#
rho <- 0.9
mu_y <- 0.0
mu_x <- 0.0
sig2_y<-1.0
sig2_x<-1.0
#
mu <-c(mu_y,mu_x)
sig2 <- rbind(c(sig2_y,rho),c(rho,sig2_x))
#
z_obs <- rmvnorm(n=n_obs, mean=mu, sigma=sig2) ;
y_obs <- z_obs[,1]
x_obs <- z_obs[,2]
#
# define the empirical risk function
#
comp_loss_fun <-function(w, z) {
  y <- z[1]
  x <- z[2]
  lf <- ( y -w[1] -w[2]*x )^2
  return ( lf )
}
#
comp_grad_loss <- function(w, z) {
  y <- z[1]
  x <- z[2]
  grd <- c(0,0)
  grd[1] <-  -2 * ( y -w[1] -w[2]*x )
  grd[2] <-  -2 * ( y -w[1] -w[2]*x )*x
  return (grd)
}
#
#erf <- function(w,n=length(z_obs),z=z_obs) {
#  y_obs <- z_obs[,1]
#  x_obs <- z_obs[,2]
#  erf <- mean( (y_obs-w[1]-w[2]*x_obs)^2 ) 
#}
#
w_true <- c(0.0, 1.0)


################################################################################
################################################################################

# batch SGD 

w_seed <- c(2,-2)

eta <- c(0.015)

batch_size_vec <- c(1, 100)

set.seed(2023)

pdf(file = "./SGD_w_1.pdf")

for (m in 1:length(batch_size_vec)) {
  #
  # Set the seed
  #
  w <- w_seed
  w_chain <- c(w)
  #
  # iterate
  #
  t <- 1
  Qterm <- 0
  while ( (t < T) &&  (Qterm != 1) ) {
    #step 1: GD update
    #eta <- learn_rate(t)
    #grad <- gradient(erf,w)
    #J <- sample.int( n = n_obs, size = m, replace = TRUE)
    batch_size <- batch_size_vec[m]
    J <- sample.int( n = n_obs, size = batch_size, replace = FALSE)
    grad <- c(0,0)
    for (j in J) {
      grad <- grad + comp_grad_loss( w, z_obs[j,] )
    }
    grad <- grad / batch_size 
    w <- w -eta*grad
    # step 2; termination crioterion
    t <- t+1
    if  ( t >= T ) {
      Qterm <- 1
    }
    # record the produced chain
    w_chain <- rbind(w_chain,w)
  } 
  if (m==1) {
    plot(w_chain[,1], type="l", ylim = c(-1.5,2.0), ylab ='w_{1}^{(t)}', xlab ='t', col=m, 
         cex.lab=fntsz, 
         cex.axis=fntsz, 
         cex.main=fntsz, 
         cex.sub=fntsz )
    abline(h = w_true[1], col=length(batch_size_vec)+1)
  }
  else {
    lines(w_chain[,1], type="l", ylim = c(-1.5,2.0), ylab ='w_{1}^{(t)}', xlab ='t', col=m , 
          cex.lab=fntsz, 
          cex.axis=fntsz, 
          cex.main=fntsz, 
          cex.sub=fntsz)
  }
  
}
legend('topright', title="batch size", legend=batch_size_vec,  lty=1, col=1:length(batch_size_vec), 
       cex=fntsz)
dev.off()

################################################################################
################################################################################

# batch SGD 

w_seed <- c(2,-2)

eta <- c(0.015)

batch_size_vec <- c(1, 100)

set.seed(2023)

pdf(file = "./SGD_w_2.pdf")

for (m in 1:length(batch_size_vec)) {
  #
  # Set the seed
  #
  w <- w_seed
  w_chain <- c(w)
  #
  # iterate
  #
  t <- 1
  Qterm <- 0
  while ( (t < T) &&  (Qterm != 1) ) {
    #step 1: GD update
    #eta <- learn_rate(t)
    #grad <- gradient(erf,w)
    #J <- sample.int( n = n_obs, size = m, replace = TRUE)
    batch_size <- batch_size_vec[m]
    J <- sample.int( n = n_obs, size = batch_size, replace = FALSE)
    grad <- c(0,0)
    for (j in J) {
      grad <- grad + comp_grad_loss( w, z_obs[j,] )
    }
    grad <- grad / batch_size 
    w <- w -eta*grad
    # step 2; termination crioterion
    t <- t+1
    if  ( t >= T ) {
      Qterm <- 1
    }
    # record the produced chain
    w_chain <- rbind(w_chain,w)
  } 
  if (m==1) {
    plot(w_chain[,2], type="l", ylim = c(-2.1,1.2), ylab ='w_{2}^{(t)}', xlab ='t', col=m, 
         cex.lab=fntsz, 
         cex.axis=fntsz, 
         cex.main=fntsz, 
         cex.sub=fntsz )
    abline(h = w_true[2], col=length(batch_size_vec)+1)
  }
  else {
    lines(w_chain[,2], type="l", ylim = c(-2.1,1.2), ylab ='w_{2}^{(t)}', xlab ='t', col=m , 
          cex.lab=fntsz, 
          cex.axis=fntsz, 
          cex.main=fntsz, 
          cex.sub=fntsz)
  }
  
}
legend('bottomright', title="batch size", legend=batch_size_vec,  lty=1, col=1:length(batch_size_vec), 
       cex=fntsz)
dev.off()

