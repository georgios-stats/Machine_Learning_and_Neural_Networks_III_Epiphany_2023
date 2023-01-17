rm(list=ls())
#
library(mvtnorm)
#library(rootSolve)
#
set.seed(2023)
#
fntsz <- 1.5
#
T <- 1000
#
n_obs <- 500
#
rho <- 1.0
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
erf <- function(w,n=length(z_obs),z=z_obs) {
  y_obs <- z_obs[,1]
  x_obs <- z_obs[,2]
  erf <- mean( (y_obs-w[1]-w[2]*x_obs)^2 ) 
}
#
w_true <- c(0.0, 1.0)
#
# define the learning rate
#
learn_rate <- function(t) {
  learn_rate <- 0.01
}


################################################################################
################################################################################

w_seed <- c(2,-2)

eta_vec <- c(0.01, 0.02, 0.05, 0.99)

pdf(file = "./w_1.pdf")

set.seed(2023)

for (j in 1:length(eta_vec)) {
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
    grad <- c( 0.5*(w[1]+w[2]*mean(x_obs)-mean(y_obs) ) , mean(0.5*(w[1]+w[2]*x_obs-y_obs )*x_obs) )
    eta <- eta_vec[j]
    w <- w -eta*grad
    # step 2; termination crioterion
    #grad <- gradient(erf,w)
    #if  ( ( norm(grad, type="2") <= 0.0001 ) ) {
    #  Qterm <- 1
    #}
    t <- t+1
    if  ( t >= T ) {
      Qterm <- 1
    }
    # record the produced chain
    w_chain <- rbind(w_chain,w)
  }
  if (j==1) {
    plot(w_chain[,1], type="l", ylim = c(-1.5,2.0), ylab ='w_{1}^{(t)}', xlab ='t', col=j, 
         cex.lab=fntsz, 
         cex.axis=fntsz, 
         cex.main=fntsz, 
         cex.sub=fntsz )
    abline(h = w_true[1], col='red')
  }
  else {
    lines(w_chain[,1], type="l", ylim = c(-1.5,2.0), ylab ='w_{1}^{(t)}', xlab ='t', col=j , 
          cex.lab=fntsz, 
          cex.axis=fntsz, 
          cex.main=fntsz, 
          cex.sub=fntsz)
  }
  
}
legend('topright', title="learning rate", legend=eta_vec,  lty=1, col=1:length(eta_vec), 
       cex=fntsz)
dev.off()


################################################################################
################################################################################

w_seed <- c(2,-2)

eta_vec <- c(0.01, 0.02, 0.05, 0.99)

pdf(file = "./w_2.pdf")

set.seed(2023)

for (j in 1:length(eta_vec)) {
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
    grad <- c( 0.5*(w[1]+w[2]*mean(x_obs)-mean(y_obs) ) , mean(0.5*(w[1]+w[2]*x_obs-y_obs )*x_obs) )
    eta <- eta_vec[j]
    w <- w -eta*grad
    # step 2; termination crioterion
    #grad <- gradient(erf,w)
    #if  ( ( norm(grad, type="2") <= 0.0001 ) ) {
    #  Qterm <- 1
    #}
    t <- t+1
    if  ( t >= T ) {
      Qterm <- 1
    }
    # record the produced chain
    w_chain <- rbind(w_chain,w)
  }
  if (j==1) {
    plot(w_chain[,2], type="l", ylim =c(-2.1,1.2), ylab ='w_{2}^{(t)}', xlab ='t', col=j , 
         cex.lab=fntsz, 
         cex.axis=fntsz, 
         cex.main=fntsz, 
         cex.sub=fntsz)
    abline(h = w_true[2], col='red')
  }
  else {
    lines(w_chain[,2], type="l", ylim = c(-2.1,1.2), ylab ='w_{2}^{(t)}', xlab ='t', col=j, 
          cex.lab=fntsz, 
          cex.axis=fntsz, 
          cex.main=fntsz, 
          cex.sub=fntsz)
  }
  
}
legend('bottomright', title="learning rate", legend=eta_vec,  lty=1, col=1:length(eta_vec), 
       cex=fntsz)
dev.off()




################################################################################
################################################################################

w_seed <- c(2,-2)

eta_vec <- c(0.01, 0.02, 0.05, 0.99)

pdf(file = "./f_error.pdf")

set.seed(2023)

for (j in 1:length(eta_vec)) {
  #
  # Set the seed
  #
  w <- w_seed
  fw_chain <- c(erf(w)-erf(w_true))
  #
  # iterate
  #
  t <- 1
  Qterm <- 0
  while ( (t < T) &&  (Qterm != 1) ) {
    #step 1: GD update
    #eta <- learn_rate(t)
    #grad <- gradient(erf,w)
    grad <- c( 0.5*(w[1]+w[2]*mean(x_obs)-mean(y_obs) ) , mean(0.5*(w[1]+w[2]*x_obs-y_obs )*x_obs) )
    eta <- eta_vec[j]
    w <- w -eta*grad
    # step 2; termination crioterion
    #grad <- gradient(erf,w)
    #if  ( ( norm(grad, type="2") <= 0.0001 ) ) {
    #  Qterm <- 1
    #}
    t <- t+1
    if  ( t >= T ) {
      Qterm <- 1
    }
    # record the produced chain
    fw_chain <- rbind(fw_chain,erf(w)-erf(w_true))
  }
  if (j==1) {
    plot(fw_chain, type="l", ylim = c(-0.2,2.0), ylab ='f^{(t)}-f^{*}', xlab ='t', col=j , 
         cex.lab=fntsz, 
         cex.axis=fntsz, 
         cex.main=fntsz, 
         cex.sub=fntsz)
  }
  else {
    lines(fw_chain, type="l", ylim = c(-0.2,2.0), ylab ='f^{(t)}-f^{*}', xlab ='t', col=j , 
          cex.lab=fntsz, 
          cex.axis=fntsz, 
          cex.main=fntsz, 
          cex.sub=fntsz)
  }
  
}
legend('topright', title="learning rate", legend=eta_vec,  lty=1, col=1:length(eta_vec), 
       cex=fntsz)
dev.off()

################################################################################
################################################################################

w_seed <- c(2,-2)

eta_vec <- c(1, 3)

T <- 10

pdf(file = "./w_1_zoom.pdf")

set.seed(2023)

for (j in 1:length(eta_vec)) {
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
    grad <- c( 0.5*(w[1]+w[2]*mean(x_obs)-mean(y_obs) ) , mean(0.5*(w[1]+w[2]*x_obs-y_obs )*x_obs) )
    eta <- eta_vec[j]
    w <- w -eta*grad
    # step 2; termination crioterion
    #grad <- gradient(erf,w)
    #if  ( ( norm(grad, type="2") <= 0.0001 ) ) {
    #  Qterm <- 1
    #}
    t <- t+1
    if  ( t >= T ) {
      Qterm <- 1
    }
    # record the produced chain
    w_chain <- rbind(w_chain,w)
  }
  if (j==1) {
    plot(w_chain[,1], type="l", ylim = c(-1.5,2.0), ylab ='w_{1}^{(t)}', xlab ='t', col=j , 
         cex.lab=fntsz, 
         cex.axis=fntsz, 
         cex.main=fntsz, 
         cex.sub=fntsz)
    abline(h = w_true[1], col='red')
  }
  else {
    lines(w_chain[,1], type="l", ylim = c(-1.5,2.0), ylab ='w_{1}^{(t)}', xlab ='t', col=j , 
          cex.lab=fntsz, 
          cex.axis=fntsz, 
          cex.main=fntsz, 
          cex.sub=fntsz)
  }
  
}
legend('topright', title="learning rate", legend=eta_vec,  lty=1, col=1:length(eta_vec), 
       cex=fntsz)
dev.off()


################################################################################
################################################################################

w_seed <- c(2,-2)

eta_vec <- c(1, 3)

T <- 10

pdf(file = "./w_2_zoom.pdf")

set.seed(2023)

for (j in 1:length(eta_vec)) {
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
    grad <- c( 0.5*(w[1]+w[2]*mean(x_obs)-mean(y_obs) ) , mean(0.5*(w[1]+w[2]*x_obs-y_obs )*x_obs) )
    eta <- eta_vec[j]
    w <- w -eta*grad
    # step 2; termination crioterion
    #grad <- gradient(erf,w)
    #if  ( ( norm(grad, type="2") <= 0.0001 ) ) {
    #  Qterm <- 1
    #}
    t <- t+1
    if  ( t >= T ) {
      Qterm <- 1
    }
    # record the produced chain
    w_chain <- rbind(w_chain,w)
  }
  if (j==1) {
    plot(w_chain[,2], type="l", ylim = c(-1.5,2.0), ylab ='w_{2}^{(t)}', xlab ='t', col=j , 
         cex.lab=fntsz, 
         cex.axis=fntsz, 
         cex.main=fntsz, 
         cex.sub=fntsz)
    abline(h = w_true[2], col='red')
  }
  else {
    lines(w_chain[,2], type="l", ylim = c(-1.5,2.0), ylab ='w_{2}}^{(t)}', xlab ='t', col=j , 
          cex.lab=fntsz, 
          cex.axis=fntsz, 
          cex.main=fntsz, 
          cex.sub=fntsz)
  }
  
}
legend('topright', title="learning rate", legend=eta_vec,  lty=1, col=1:length(eta_vec), 
       cex=fntsz)
dev.off()


################################################################################
################################################################################

w_seed <- c(2,-2)

eta_vec <- c(1, 3)

T <- 10

pdf(file = "./f_error_zoom.pdf")

set.seed(2023)

for (j in 1:length(eta_vec)) {
  #
  # Set the seed
  #
  w <- w_seed
  fw_chain <- c(erf(w)-erf(w_true))
  #
  # iterate
  #
  t <- 1
  Qterm <- 0
  while ( (t < T) &&  (Qterm != 1) ) {
    #step 1: GD update
    #eta <- learn_rate(t)
    #grad <- gradient(erf,w)
    grad <- c( 0.5*(w[1]+w[2]*mean(x_obs)-mean(y_obs) ) , mean(0.5*(w[1]+w[2]*x_obs-y_obs )*x_obs) )
    eta <- eta_vec[j]
    w <- w -eta*grad
    # step 2; termination crioterion
    #grad <- gradient(erf,w)
    #if  ( ( norm(grad, type="2") <= 0.0001 ) ) {
    #  Qterm <- 1
    #}
    t <- t+1
    if  ( t >= T ) {
      Qterm <- 1
    }
    # record the produced chain
    fw_chain <- rbind(fw_chain,erf(w)-erf(w_true))
  }
  if (j==1) {
    plot(fw_chain, type="l", ylim = c(0.1,4.0), ylab ='f^{(t)}-f^{*}', xlab ='t', col=j , 
         cex.lab=fntsz, 
         cex.axis=fntsz, 
         cex.main=fntsz, 
         cex.sub=fntsz)
    abline(h = 0.0, col='red')
  }
  else {
    lines(fw_chain, type="l", ylim = c(0.1,4.0), ylab ='f^{(t)}-f^{*}', xlab ='t', col=j , 
          cex.lab=fntsz, 
          cex.axis=fntsz, 
          cex.main=fntsz, 
          cex.sub=fntsz)
  }
  
}
legend('topright', title="learning rate", legend=eta_vec,  lty=1, col=1:length(eta_vec), 
       cex=fntsz)
dev.off()






