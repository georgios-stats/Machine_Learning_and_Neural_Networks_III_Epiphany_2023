rm(list=ls())
#
library(nloptr)
library(mvtnorm)
library(truncnorm)
#
set.seed(2023)
#
fntsz <- 1.5
#
T <- 200
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
x_obs <- rtruncnorm(n=n_obs, a=-1, b=1, mean = mu_x, sd = sqrt(sig2_x) )
y_obs <- 0.0 + rho*x_obs +0.05*rnorm(n=n_obs)
z_obs <- cbind(y_obs,x_obs)
#x11();plot(z_obs)
#
# define the empirical risk function
#
comp_loss_fun <-function(w, z) {
  y <- z[1]
  x <- z[2]
  lf <- -cos( (y -w[1] -w[2]*x)/2 )
  return ( lf )
}
#
comp_grad_loss <- function(w, z) {
  y <- z[1]
  x <- z[2]
  grd <- c(0,0)
  grd[1] <-  -sin( (y -w[1] -w[2]*x)/1 )
  grd[2] <-  -x*sin( (y -w[1] -w[2]*x)/1 )
  return (grd)
}
#
w_true <- c(0.0, rho)
#
comp_eta <- function(t) {
  eta <- 50/t
  return(eta)
}
#
################################################################################
################################################################################
#
# batch SGD 
#
w_seed <- c(1.5,1.5)
w_seed <- c(1,-1)
#
batch_size_vec <- c(1)
#
set.seed(2023)
#
pdf(file = "./projSGD_w_1.pdf")
#
# SGD
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
  batch_size <- batch_size_vec
  J <- sample.int( n = n_obs, size = batch_size, replace = FALSE)
  grad <- c(0,0)
  for (j in J) {
    grad <- grad + comp_grad_loss( w, z_obs[j,] )
  }
  eta <- comp_eta(t)
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
plot(w_chain[,1], type="l",  ylab ='w_{1}^{(t)}', xlab ='t', col=1, 
     cex.lab=fntsz, 
     cex.axis=fntsz, 
     cex.main=fntsz, 
     cex.sub=fntsz )
#
# projSGD
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
  batch_size <- batch_size_vec
  J <- sample.int( n = n_obs, size = batch_size, replace = FALSE)
  grad <- c(0,0)
  for (j in J) {
    grad <- grad + comp_grad_loss( w, z_obs[j,] )
  }
  grad <- grad / batch_size 
  eta <- comp_eta(t)
  w <- w -eta*grad
  # step 1.5 projection
  eval_f0 <- function( w_proj, w_now ){ 
    return( sqrt(sum((w_proj-w_now)^2)) )
  }
  eval_grad_f0 <- function( w, w_now ){ 
    return( c( 2*(w[1]-w_now[1]), 2*(w[2]-w_now[2]) ) )
  }
  eval_g0 <- function( w_proj, w_now) {
    return( sum(w_proj^2) -(1.5)^2 )
  }
  eval_jac_g0 <- function( x, w_now ) {
    return(   c(2*w[1],2*w[2] )  )
  }
  out <- nloptr(x0=c(0.0,0.0),
                eval_f=eval_f0,
                eval_grad_f=eval_grad_f0,
                eval_g_ineq = eval_g0,
                eval_jac_g_ineq = eval_jac_g0, 
                w_now=w,
                opts = list("algorithm" = "NLOPT_LD_MMA",
                            "xtol_rel"=1.0e-8),
                  )
  w <- out$solution
  # step 2; termination crioterion
  t <- t+1
  if  ( t >= T ) {
    Qterm <- 1
  }
  # record the produced chain
  w_chain <- rbind(w_chain,w)
} 
lines(w_chain[,1], type="l",  ylab ='w_{1}^{(t)}', xlab ='t', col=2, 
      cex.lab=fntsz, 
      cex.axis=fntsz, 
      cex.main=fntsz, 
      cex.sub=fntsz )
#
abline(h = w_true[1], col=3)
#
legend('right', title="Algorithm", legend=c('SGD', 'projSGD','true'),  lty=1, col=c(1,2,3), 
       cex=fntsz)
dev.off()
#
################################################################################
################################################################################
#
# batch SGD 
#
w_seed <- c(1.5,1.5)
w_seed <- c(1,-1)

#
batch_size_vec <- c(1)
#
set.seed(2023)
#
pdf(file = "./projSGD_w_2.pdf")
#
# SGD
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
  batch_size <- batch_size_vec
  J <- sample.int( n = n_obs, size = batch_size, replace = FALSE)
  grad <- c(0,0)
  for (j in J) {
    grad <- grad + comp_grad_loss( w, z_obs[j,] )
  }
  eta <- comp_eta(t)
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
plot(w_chain[,2], type="l",  ylab ='w_{2}^{(t)}', xlab ='t', col=1, 
     ylim=c(-33,1.5), 
     cex.lab=fntsz, 
     cex.axis=fntsz, 
     cex.main=fntsz, 
     cex.sub=fntsz )
#
# projSGD
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
  batch_size <- batch_size_vec
  J <- sample.int( n = n_obs, size = batch_size, replace = FALSE)
  grad <- c(0,0)
  for (j in J) {
    grad <- grad + comp_grad_loss( w, z_obs[j,] )
  }
  grad <- grad / batch_size 
  eta <- comp_eta(t)
  w <- w -eta*grad
  # step 1.5 projection
  eval_f0 <- function( w_proj, w_now ){ 
    return( sqrt(sum((w_proj-w_now)^2)) )
  }
  eval_grad_f0 <- function( w, w_now ){ 
    return( c( 2*(w[1]-w_now[1]), 2*(w[2]-w_now[2]) ) )
  }
  eval_g0 <- function( w_proj, w_now) {
    return( sum(w_proj^2) -(1.5)^2 )
  }
  eval_jac_g0 <- function( x, w_now ) {
    return(   c(2*w[1],2*w[2] )  )
  }
  out <- nloptr(x0=c(0.0,0.0),
                eval_f=eval_f0,
                eval_grad_f=eval_grad_f0,
                eval_g_ineq = eval_g0,
                eval_jac_g_ineq = eval_jac_g0, 
                w_now=w,
                opts = list("algorithm" = "NLOPT_LD_MMA",
                            "xtol_rel"=1.0e-8),
  )
  w <- out$solution
  # step 2; termination crioterion
  t <- t+1
  if  ( t >= T ) {
    Qterm <- 1
  }
  # record the produced chain
  w_chain <- rbind(w_chain,w)
} 
lines(w_chain[,2], type="l",  ylab ='w_{2}^{(t)}', xlab ='t', col=2, 
      cex.lab=fntsz, 
      cex.axis=fntsz, 
      cex.main=fntsz, 
      cex.sub=fntsz )
#
abline(h = w_true[2], col=3)
#
legend('right', title="Algorithm", legend=c('SGD', 'projSGD','true'),  lty=1, col=c(1,2,3), 
       cex=fntsz)
dev.off()

