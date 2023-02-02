rm(list=ls())
graphics.off()
#
library(mvtnorm)
library(invgamma)
#
set.seed(2023)
#
fntsz <- 1.5
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
d <- dim(z_obs)[2]-1
x_obs <- z_obs[ , 1:d ]
y_obs <- z_obs[ , d+1 ]
#
mle.out <- lm(y_obs ~ 1 + x_obs )
b_mle <- mle.out$coefficients
sig2_mle <- var(mle.out$residuals)
#
# define the log sampling pdf 
#
log_sampling_pdf <-function(z, w) {
  d = length(z)-1 # number of regressors
  x <- c(1.0,z[1:d]) 
  y <- z[d+1]
  b <- w[1:(d+1)]
  y_mean <- sum(c(x) * c(b)) ;
  y_sig2 <- exp( w[ d+2 ] )
  log_pdf <- log( dnorm(y, 
                        mean = y_mean, 
                        sd = sqrt(y_sig2), 
                        log = FALSE) )
  return ( log_pdf )
}
#
comp_learning_rate <- function(t, C = 1.0, s = 1.0) { 
  eta <- C / (t)^s  
  return ( eta )
}
#
# log_likelihood <-function(n, d, z, w) {
#   # n is the # of examples
#   # d is the # of regressors
#   log_lik <- 0.0 
#   for (i in 1:n) {
#     log_lik <- log_lik + log_sampling_pdf(z[i,], w)
#   }
#   return ( log_lik )
# }
#
beta_prior_mean <- rep(0.0,times=d+1)
#
beta_prior_var <- diag(rep(100.0,times=d+1))
#
phi_prior <- 1.0
#
psi_prior <- 1.0
#
log_prior_pdf <- function(w,
                          beta_prior_mean = 0.0,
                          beta_prior_var = 100.0,
                          phi = phi_prior,
                          psi = psi_prior) {
  d <- length(w)
  b <- w[1:(d-1)]
  log_prior <- 0.0
  for (j in 1:(d-1)) {
    log_prior <- log_prior +log( dnorm( b[j], 
                                        mean = beta_prior_mean, 
                                        sd = sqrt(beta_prior_var), 
                                        log = FALSE) )
  }
  Jac <- exp( w[d] )
  log_prior <- log_prior +log( dinvgamma( exp(w[d]), 
                                          shape = phi, 
                                          rate = psi, 
                                          log = FALSE) 
                               )  +log(Jac)
  return(log_prior)
}
#
#
################################################################################
################################################################################
#
# COMPUTE THE PARAMETERS OF THE EXACT MARGINAL POSTERIOR DISTRIBUTIONS 
# (ASSUMED AS A BLACK BOX HERE)
#
source('./example_SGLD_auxiliary.R')
#
out <- marginal_posterior_parameters_Normal_Linear_Regression(z_obs[,d+1], 
                                                       cbind(1.0,z_obs[,1:d]), 
                                                       beta_prior_mean, 
                                                       beta_prior_var, 
                                                       phi_prior, 
                                                       psi_prior) ;
beta_mu_post <- out$mu_post  
beta_Sig_post <- out$Sig2_diag_post 
phi_post<- out$phi_post 
psi_post <- out$psi_post 

#
################################################################################
################################################################################
#
# SGLD 
#
Tmax <- 1000
#
w_seed <- c(1.1, -2.0, log(1.0))
#
eta_vec <- c(10^(-4), 10^(-5), 10^(-6))
eta <- 10^(-5)
eta_C <- 10^(-4)
eta_s <- 1.0
#
batch_size_vec <- c(10, 100, 1000)
batch_size <- 100
#
tau_vec <- c(0.2, 1.0, 2.0)
tau <- 1.0
#
#set.seed(2023)
#
for (batch_size in batch_size_vec ) {
  # Set the seed
  w <- w_seed
  w_chain <- c(w)
  # iterate
  t <- 1
  Qterm <- 0
  while ( (Qterm != 1) ) {
    # counter 
    t <- t+1
    # sub-sample
    J <- sample.int( n = n_obs, size = batch_size, replace = FALSE)
    # update
    grad_est_lik <- rep( 0.0, times=length(w) )
    for (j in J) {
      fun <- function(w, z=z_obs[j,]){
        gr <- log_sampling_pdf(z, w)
        return(gr)
      }
      grad_est_lik <- grad_est_lik + numDeriv::grad(fun, w)
    }
    grad_est_lik <- ( n_obs / batch_size) * grad_est_lik 
    #eta <- comp_learning_rate(t, eta_C, eta_s)
    w <- w +eta*grad_est_lik +eta*numDeriv::grad(log_prior_pdf, w) +sqrt(eta)*sqrt(tau)*rnorm(n = length(w), mean = 0, sd = 1)
    #    w <- w +eta*grad_est_lik +eta*numDeriv::grad(log_prior_pdf, w) 
    # termination criterion
    if  ( t >= Tmax ) {
      Qterm <- 1
    }
    # record the produced chain
    w_chain <- rbind(w_chain,w)
  }
  #
  for (j in 1:2) {
    pdf(file = paste("SGLD_w",as.character(j),".batch_size=",as.character(batch_size),".pdf", sep = "") )
    h=hist(w_chain[(Tmax/2):Tmax,j], 
         freq = FALSE,
         type="l",
         xlab = paste("w(",as.character(j),")",sep=""),
         ylab = "",
         main = paste("SGD"," batch size=", as.character(batch_size) , sep = ""),
         cex.lab=fntsz,
         cex.axis=fntsz,
         cex.main=fntsz,
         cex.sub=fntsz );
    xx <- seq(from=beta_mu_post[j]-4.0*sqrt(beta_Sig_post[j]), 
              to=beta_mu_post[j]+4.0*sqrt(beta_Sig_post[j]), 
              length = 100)
    fxx <- dt_gen(xx, 
                  2*phi_post,
                  beta_mu_post[j], 
                  beta_Sig_post[j]*(psi_post/phi_post))
    lines(xx,
          fxx   ,
          type="l",col="red")
    dev.off()
  }
  #
  j = 3
  pdf(file = paste("SGLD_w",as.character(j),".batch_size=",as.character(batch_size),".pdf", sep = "") )
  hist(exp(w_chain[(Tmax/2):Tmax,j]), 
       freq = FALSE,
       type="l",
       xlab = paste("exp(w(",as.character(j),"))",sep=""),
       ylab = "",
       main = paste("SGD"," batch size=", as.character(batch_size) , sep = ""),
       cex.lab=fntsz,
       cex.axis=fntsz,
       cex.main=fntsz,
       cex.sub=fntsz ); 
  xx <- seq(from=0.17, to=0.2, length = 100)
  fxx <- invgamma::dinvgamma(xx, phi_post, psi_post)
  lines(xx,fxx,type="l",col="red")
  dev.off()
}
#
#
#
################################################################################
################################################################################
#
# SGLD 
#
Tmax <- 1000
#
w_seed <- c(1.1, -2.0, log(1.0))
#
eta_vec <- c(10^(-4), 10^(-5), 10^(-6))
eta <- 10^(-5)
eta_C <- 10^(-4)
eta_s <- 1.0
#
batch_size_vec <- c(10, 100, 1000)
batch_size <- 100
#
tau_vec <- c(0.01, 0.1, 1.0, 2.0, 10.0)
tau <- 1.0
#
#set.seed(2023)
#
for (tau in tau_vec ) {
  # Set the seed
  w <- w_seed
  w_chain <- c(w)
  # iterate
  t <- 1
  Qterm <- 0
  while ( (Qterm != 1) ) {
    # counter 
    t <- t+1
    # sub-sample
    J <- sample.int( n = n_obs, size = batch_size, replace = FALSE)
    # update
    grad_est_lik <- rep( 0.0, times=length(w) )
    for (j in J) {
      fun <- function(w, z=z_obs[j,]){
        gr <- log_sampling_pdf(z, w)
        return(gr)
      }
      grad_est_lik <- grad_est_lik + numDeriv::grad(fun, w)
    }
    grad_est_lik <- ( n_obs / batch_size) * grad_est_lik 
    #eta <- comp_learning_rate(t, eta_C, eta_s)
    w <- w +eta*grad_est_lik +eta*numDeriv::grad(log_prior_pdf, w) +sqrt(eta)*sqrt(tau)*rnorm(n = length(w), mean = 0, sd = 1)
    #    w <- w +eta*grad_est_lik +eta*numDeriv::grad(log_prior_pdf, w) 
    # termination criterion
    if  ( t >= Tmax ) {
      Qterm <- 1
    }
    # record the produced chain
    w_chain <- rbind(w_chain,w)
  }
  #
  for (j in 1:2) {
    pdf(file = paste("SGLD_w",as.character(j),".tau=",as.character(tau),".pdf", sep = "") )
    h=hist(w_chain[(Tmax/2):Tmax,j], 
           freq = FALSE,
           xlab = paste("w(",as.character(j),")",sep=""),
           ylab = "",
           main = paste("SGD"," tau=", as.character(tau) , sep = ""),
           cex.lab=fntsz,
           cex.axis=fntsz,
           cex.main=fntsz,
           cex.sub=fntsz );
    xx <- seq(from=beta_mu_post[j]-4.0*sqrt(beta_Sig_post[j]), 
              to=beta_mu_post[j]+4.0*sqrt(beta_Sig_post[j]), 
              length = 100)
    fxx <- dt_gen(xx, 
                  2*phi_post,
                  beta_mu_post[j], 
                  beta_Sig_post[j]*(psi_post/phi_post))
    lines(xx,
          fxx   ,
          type="l",col="red")
    dev.off()
  }
  #
  j = 3
  pdf(file = paste("SGLD_w",as.character(j),".tau=",as.character(tau),".pdf", sep = "") )
  hist(exp(w_chain[(Tmax/2):Tmax,j]), 
       freq = FALSE,
       type="l",
       xlab = paste("exp(w(",as.character(j),"))",sep=""),
       ylab = "",
       main = paste("SGD"," tau=", as.character(tau) , sep = ""),
       cex.lab=fntsz,
       cex.axis=fntsz,
       cex.main=fntsz,
       cex.sub=fntsz ); 
  xx <- seq(from=0.17, to=0.2, length = 100)
  fxx <- invgamma::dinvgamma(xx, phi_post, psi_post)
  lines(xx,fxx,type="l",col="red")
  dev.off()
}



