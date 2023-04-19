rm(list=ls())
#
data_generating_model <- function(n,w) {
  z <- rep( NaN, times=n*3 )
  z <- matrix(z, nrow = n, ncol = 3)
  z[,1] <- rep(1,times=n)
  z[,2] <- runif(n, min = -10, max = 10)
  p <- w[1]*z[,1] + w[2]*z[,2] 
  p <- exp(p) / (1+exp(p))
  z[,3] <- rbinom(n, size = 1, prob = p)
  ind <- (z[,3]==0)
  z[ind,3] <- -1
  x <- z[,1:2]
  y <- z[,3]
  return(list(z=z, x=x, y=y))
}
n_obs <- 1000000
w_true <- c(-3,4)  
set.seed(2023)
out <- data_generating_model(n = n_obs, w = w_true) 
set.seed(0)
z_obs <- out$z #z=(x,y)
x <- out$x
y <- out$y
#z_obs2=z_obs
#z_obs2[z_obs[,3]==-1,3]=0
#w_true <- as.numeric(glm(z_obs2[,3]~ 1+ z_obs2[,2],family = "binomial" )$coefficients)

comp_grad_loss <- function(w, z, lam ) {
  d <- length(w)
  x <- z[ 1:d ]
  y <- z[ d+1 ]
  Q <- 1.0 - y* sum(w*x)
  if ( Q < 0.0 ) {
    gr <- lam*2.0*w
  } else if ( Q > 0.0 ) {
    gr <- -y*x +lam*2.0*w
  } else {
    gr <- lam*2.0*w
  }
  return (gr)
}
#




set.seed(0)

m <- 3
eta <- 20
Tmax <- 50000
def_lam <- 0.0
w_seed <- c(-0.0, 0.0)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
G <- rep(0.0,times=length(w))
eps <- 10^(-6)
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  # step 1: update  
  J <- sample.int(n = n_obs, size = m, replace = TRUE)
  if (m==1) {
    zbatch <- matrix(z_obs[J,],nrow=m)
  } else {
    zbatch <- z_obs[J,]
  }
  g <- 0.0 
  for (i in 1:m) {
    g <- g + comp_grad_loss(w, c(zbatch[i,]), lam=def_lam )
  }
  g <- g/m
  G <- G + g^2
  w <- w - eta * (1.0/sqrt(G+eps)) * g
  #w <- w - eta * grad_risk_fun( w, zbatch, m )
  w_chain <- rbind(w_chain, w)
  # step 2: check for rtermination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
plot(w_chain[,1], type='l') +
  abline(h=w_true[1], col='red')
plot(w_chain[,2], type='l') +
  abline(h=w_true[2], col='red')


comp_predictive_rule <- function(w,x) {
  h <- sign( sum(w*x) )
  return(h)
}

w_sgd <- colMeans(w_chain[(Tmax-500):Tmax,])
w_sgd
comp_predictive_rule(w_sgd,c(1,0))
