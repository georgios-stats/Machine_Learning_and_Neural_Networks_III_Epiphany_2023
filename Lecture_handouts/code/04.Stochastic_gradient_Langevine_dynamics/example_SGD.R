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
rho <- 0.9
mu_y <- 0.0
mu_x <- 0.0
sig2_y<-1.0
sig2_x<-1.0
mu <-c(mu_y,mu_x)
sig2 <- rbind(c(sig2_y,rho),c(rho,sig2_x))
z_obs <- rmvnorm(n=n_obs, mean=mu, sigma=sig2) ;
d <- dim(z_obs)[2]-1
x_obs <- z_obs[ , 1:d ]
y_obs <- z_obs[ , d+1 ]
#
# MLE (exact computation)
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
log_prior_pdf <- function(w,
                          beta_prior_mean = 0.0,
                          beta_prior_var = 100.0,
                          phi = 1.0,
                          psi = 1.0) {
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
################################################################################
################################################################################
#
# batch SGD 
#
Tmax <- 1000
#
w_seed <- c(1.1, -2.0, log(1.0))
#
eta_vec <-c(10^(-4), 10^(-5), 10^(-6))
eta <- 10^(-5)
#
batch_size_vec <- c(10, 100, 1000)
batch_size <- 100
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
    w <- w +eta*grad_est_lik +eta*numDeriv::grad(log_prior_pdf, w)
    # termination crioterion
    if  ( t >= Tmax ) {
      Qterm <- 1
    }
    # record the produced chain
    w_chain <- rbind(w_chain,w)
  }
  #
  for (j in 1:2) {
    pdf(file = paste("SGD_w",as.character(j),".batch_size=",as.character(batch_size),".pdf", sep = "") )
    plot(w_chain[,j], 
         type="l",
         xlab ="t",
         ylab = paste("w(",as.character(j),")",sep=""),
         main = paste("SGD"," batch size=", as.character(batch_size) , sep = ""),
         cex.lab=fntsz,
         cex.axis=fntsz,
         cex.main=fntsz,
         cex.sub=fntsz );
    abline(h=b_mle[j], col="red")
    legend('bottomright', title="algorithms", legend=c(paste("SGD(m=",as.character(batch_size),")",sep=""),"MLE"),  lty=1, col=1:length(batch_size_vec), 
           cex=fntsz)
    dev.off()
  }
  #
  j = 3
  pdf(file = paste("SGD_w",as.character(j),".batch_size=",as.character(batch_size),".pdf", sep = "") )
  plot(exp(w_chain[,j]), 
       type="l",
       xlab ="t",
       ylab = paste("w(",as.character(j),")",sep=""),
       main = paste("SGD"," batch size=", as.character(batch_size) , sep = ""),
       cex.lab=fntsz,
       cex.axis=fntsz,
       cex.main=fntsz,
       cex.sub=fntsz ); 
  abline(h=b_mle[j], col="red")
  legend('bottomright', title="algorithms", legend=c(paste("SGD(m=",as.character(batch_size),")",sep=""),"MLE"),  lty=1, col=1:length(batch_size_vec), 
         cex=fntsz)
  dev.off()
}
#