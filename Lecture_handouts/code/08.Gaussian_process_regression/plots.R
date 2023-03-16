# LOAD THE R PACKAGES REQUIRED
library('lhs')
library('DiceKriging')
library('DiceView')
library('Cairo')
library('latex2exp')



# 
# Plot the exact function we do not know ==========================
#

# THE TEST FUNCTION
fun_santetal <- function(x){ return( exp(-1.4*x[1])*cos(3.5*pi*x[1]) ) }

# THE TRAINING DATASET
n <- 100 ;
X <- seq(0,1,length=n) ;
Y <- rep(0,n) ;
for (i in 1:n) {
  Y[i] <- fun_santetal(X[i]) ;
}


# PLOTS

#TeX(r'($\alpha  x^\alpha$, where $\alpha \in \{1 \ldots 5\}$)')
par(mar = c(6, 6, 6, 6))
plot(X,Y, type="l",
     col="red",
     main=TeX(r'($x\rightarrow\eta(x)$)'), xlab="x", ylab=TeX(r'($\eta$)'),
     xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
     cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)

dev.copy(pdf,'./Ex_real_fun.pdf')
dev.off()

# 
# Plot the training ponts ==========================
#

# THE TEST FUNCTION
fun_santetal <- function(x){ return( exp(-1.4*x[1])*cos(3.5*pi*x[1]) ) }

# THE TRAINING DATASET
n <- 10 ;
X <- lhs::optimumLHS(n,1) ;
Y <- rep(0,n) ;
for (i in 1:n) {
  Y[i] <- fun_santetal(X[i]) ;
}

# PLOTS
par(mar = c(6, 6, 6, 6))
plot(X,Y,
     col="black",
     main=TeX(r'($y_i=\eta(x_i)+\epsilon_i$)'), xlab=TeX(r'($x_i$)'), ylab=TeX(r'($y_i$)'),
     xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
     cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)

# # THE SHOW DATASET
# n <- 100 ;
# X <- seq(0,1,length=n) ;
# Y <- rep(0,n) ;
# for (i in 1:n) {
# 	Y[i] <- fun_santetal(X[i]) ;
# }

# PLOTS

#lines(X,Y,
#		col="gray",
#		main="y = f(x)", xlab="x", ylab="y",
#		xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
#		cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)

dev.copy(pdf,'./Ex_real_fun_data.pdf')
dev.off()

# 
# Plot the covariance function for different families ==========================
#

# Plot the covariance function Path

rm(list=ls())
covtype <- c("exp", "matern3_2",  "gauss")
covtype_lab <- c(TeX(r'(Exponential ($\nu=0.5$))'), 
                 TeX(r'(Matenrn ($\nu=3/2$))'), 
                 TeX(r'(Gaussian ($\nu\rightarrow\infty$))'))
d <- 1
n <- 500
x <- seq(from=0, to=10, length=n)
param <- 1  ; 
sigma2 <- 1
# Plot the simulated paths
par(mar = c(6, 6, 6, 6))
plot(x, rep(0,n), type="l", ylim=c(-2.2, 4.7), 
     xlab="input, x", ylab=TeX(r'(output, $\eta(x)$)'), 
     main=TeX(r'(sampling path $\eta(x)\sim GP(0,K(.,.))$)'),
     cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)
for (i in 1:length(covtype)) {
  model <- km(~1, 
              design=data.frame(x=x), 
              response=rep(0,n), 
              covtype=covtype[i],
              coef.trend=0, 
              coef.cov=param, 
              coef.var=sigma2, 
              nugget=1e-4)
  y <- simulate(model)
  lines(x, y, col=i, lty=i+3, cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)
}
legend("topleft", title="Types:", legend=covtype_lab, col=1:length(covtype_lab), lty=(1:length(covtype_lab))+3, cex=2)

dev.copy(pdf,'./Ex_compare_CovFum_path.pdf')
dev.off()

# Plot the covariance function path

rm(list=ls())

covtype <- c("exp", "matern3_2",   "gauss")

covtype_lab <- c(TeX(r'(Exponential ($\nu=0.5$))'), 
                 TeX(r'(Matenrn ($\nu=3/2$))'), 
                 TeX(r'(Gaussian ($\nu\rightarrow\infty$))'))
d <- 1
n <- 500
x <- seq(from=0, to=5, length=n)
param <- 1  ; 
sigma2 <- 1
# Plot the covariance function
par(mar = c(6, 6, 6, 6))
plot(x, rep(0,n), type="l", ylim=c(0,1), 
     xlab=TeX(r'(distance $r=|x-x'|$)'), ylab=TeX(r'(Cov. function $K(r)$)'), main="Covariance functions",
     cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)
for (i in 1:length(covtype)) {
  covStruct <- covStruct.create( covtype=covtype[i], 
                                 d=d, 
                                 known.covparam="All",
                                 var.names="x", 
                                 coef.cov=param, 
                                 coef.var=sigma2 )
  y <- covMat1Mat2( covStruct, 
                    X1=as.matrix(x), 
                    X2=as.matrix(0) )
  lines(x, y, col=i, lty=i)
}
legend(x=2, y=0.8, title="Types:", legend=covtype_lab, col=1:length(covtype_lab), lty=1:length(covtype_lab), cex=2)

dev.copy(pdf,'./Ex_compare_CovFum_distance.pdf')
dev.off()


# 
# Plot the Gaussian covariance function against the distance for different scalling ==========================
#

rm(list=ls())

# Gaussian covariance function
covtype <- "gauss"
d <- 1
n <- 500
x <- seq(from=0, to=10, length=n)
#
param <- c(0.5, 3)
sigma2 <- 1
#
# Plot the covariance function
par(mar = c(6, 6, 6, 6))
plot(x, rep(0,n), type="l",
     ylim=c(0,1), 
     xlab=TeX(r'(distance $r=|x-x'|$)'), 
     ylab=TeX(r'(Cov. funct $K(r)$)'), 
     main="	Gaussian covariance function", 
     cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)
for (i in 1:length(param)) {
  covStruct <- covStruct.create(covtype=covtype, 
                                d=d, 
                                known.covparam="All",
                                var.names="x", 
                                coef.cov=param[i], 
                                coef.var=sigma2)
  y <- covMat1Mat2( covStruct, 
                    X1=as.matrix(x), 
                    X2=as.matrix(0) )
  lines(x, y, col=i, lty=i, 
        cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)
}
legend('topright', TeX(paste(r'($\phi=$)', param)), col=1:length(param), lty=1:length(param), cex=2)

dev.copy(pdf,'./Ex_CovFun_Gaussian_scale_distance.pdf')
dev.off()

# 
# Plot the Path of the Gaussian covariance function for different scalling ==========================
#

rm(list=ls())

# Gaussian covariance function
covtype <- "gauss"
d <- 1
n <- 500
x <- seq(from=0, to=10, length=n)
#
param <- c(0.5, 3)
sigma2 <- 1
#
# Plot the simulated paths
par(mar = c(6, 6, 6, 6))
plot(x, rep(0,n), type="l", ylim=c(-2.2, 4.7), 
     xlab="input, x", ylab=TeX(r'(output, $\eta(x)$)'),
     main="Sampling path using Gaussian cov. fun.",
     cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)
for (i in 1:length(param)) {
  model <- km(~1, 
              design=data.frame(x=x), 
              response=rep(0,n), 
              covtype=covtype,
              coef.trend=0, 
              coef.cov=param[i], 
              coef.var=sigma2, 
              nugget=1e-4)
  y <- simulate(model)
  lines(x, y, col=i,
        cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)
}
#
legend('topright', TeX(paste(r'($\phi=$)', param)), col=1:length(param), lty=1:length(param), cex=2)

dev.copy(pdf,'./Ex_CovFun_Gaussian_scale_path.pdf')
dev.off()

#
# Posterior the GP regression model
#

rm(list=ls())

fun_santetal <- function(x){ return( exp(-1.4*x[1])*cos(3.5*pi*x[1]) ) }

n_6 <- 6 ;
X_6 <- optimumLHS(n=n_6, k=1) ;
Y_6 <- apply(X_6, 1, fun_santetal) ;

n_10 <- 10 ;
X_10 <- optimumLHS(n=n_10, k=1) ;
Y_10 <- apply(X_10, 1, fun_santetal) ;


beta_6 <- 0.0692 ;
psi_6 <- 0.1 ;
sig_6 <- sqrt(0.1961249) ;

beta_10 <- 0.3846 ;
psi_10 <- 0.3022;
sig_10 <- sqrt(0.774677) ;

beta_6b <- 1.0 ;
psi_6b <- 0.05 ;
sig_6b <- sqrt(0.3) ;

beta_10b <- 2*beta_10 ;
psi_10b <-  sig_10/4;
sig_10b <- 4*sig_10 ;

# good
model_post <- km(formula = ~1,
                 design = data.frame(x=X_10), 
                 response = Y_10,
                 covtype = "matern5_2" ,
                 coef.trend = beta_10,
                 coef.cov = psi_10,
                 coef.var = sig_10^2)
sectionview(model_post, title = paste(" "), 
            xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
            cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)
sectionview(fun_santetal,add=TRUE,col='red', 
            xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
            cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)

dev.copy(pdf,'./GPR-n=10-par=good.pdf') ;
dev.off() ;




# bad
model_post <- km(formula = ~1,
                 design = data.frame(x=X_10), 
                 response = Y_10,
                 covtype = "matern5_2",
                 coef.trend = beta_10b,
                 coef.cov = psi_10b,
                 coef.var = sig_10b^2)
sectionview(model_post, title = paste("  "), 
            xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
            cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)
sectionview(fun_santetal,add=TRUE,col='red', 
            xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
            cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)

dev.copy(pdf,'./GPR-n=10-par=bad.pdf') ;
dev.off() ;




model_post <- km(formula = ~1,
                 design = data.frame(x=X_6), 
                 response = Y_6,
                 covtype = "matern5_2",
                 coef.trend = beta_6b,
                 coef.cov = psi_6b,
                 coef.var = sig_6b^2)
sectionview(model_post, title = paste("  "), 
            xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
            cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)
sectionview(fun_santetal,add=TRUE,col='red', 
            xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
            cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)

dev.copy(pdf,'./GPR-n=6-par=bad.pdf') ;
dev.off() ;



model_post <- km(formula = ~1,
                 design = data.frame(x=X_6), 
                 response = Y_6,
                 covtype = "matern5_2",
                 coef.trend = beta_6,
                 coef.cov = psi_6,
                 coef.var = sig_6^2)
sectionview(model_post, title = paste("  "), 
            xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
            cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2,
)
sectionview(fun_santetal,add=TRUE,col='red', 
            xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
            cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)

dev.copy(pdf,'./GPR-n=6-par=good.pdf') ;
dev.off() ;






#
# Trained Posterior the GP regression model
#

rm(list=ls())

# THE TEST FUNCTION
fun_santetal <- function(x){ return( exp(-1.4*x[1])*cos(3.5*pi*x[1]) ) }

# THE TRAINING DATASET
n <- 10 ;
X <- optimumLHS(n=n, k=1) ;
Y <- apply(X, 1, fun_santetal) ;

# TRAIN THE GP REGRESSION MODEL

model_post_train <- km(formula = ~1,
                       design = data.frame(x=X), 
                       response = Y,
                       covtype = "matern5_2",
                       nugget = 1e-7)

show(model_post_train)

# PLOTS
sectionview( model_post_train , title="Predictive (trained) GP regression", 
             xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
             cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)
sectionview( fun_santetal, add=TRUE, col='red', 
             xlim = c(0.0,1.1), ylim = c(-0.7,0.7), 
             cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2)

dev.copy(pdf,'./trained_GPR.pdf') ;
dev.off() ;
