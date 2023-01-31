
comp_sufficient_stats <- function (y, X) {
  
  n <- dim(X)[1]
  
  d <- dim(X)[2]
  
  XtX <- t(X)%*%X
  
  XtY <- t(X)%*%y
  
  YtY <- sum(y*y)
  
  return( list(XtX=XtX, XtY=XtY, YtY=YtY, n=n, d=d) )
}

comp_V_post <- function (y, X, V_prior) {
  
  sufficient_stats <- comp_sufficient_stats(y, X)
  
  XtX <- sufficient_stats$XtX
  
  V_post <- solve( XtX + solve(V_prior) ) 
  
  return (V_post)
}

comp_mu_post <- function (y, X, mu_prior, V_prior, a_prior, lam_prior) {
  
  sufficient_stats <- comp_sufficient_stats(y, X)
  
  XtX <- sufficient_stats$XtX
  
  XtY <- sufficient_stats$XtY
  
  mu_post <- comp_V_post( y, X, V_prior) %*% ( solve( V_prior , mu_prior ) + XtY )
  
  return (mu_post)
}


comp_a_post <- function ( y, X,  a_prior) {
  
  sufficient_stats <- comp_sufficient_stats(y, X)
  
  n <- sufficient_stats$n
  
  a_post <- 0.5*n + a_prior
  
  return (a_post)
}

comp_lam_post <- function (y, X, mu_prior, V_prior, a_prior, lam_prior) {
  
  sufficient_stats <- comp_sufficient_stats(y, X)
  
  YtY <- sufficient_stats$YtY
  
  V_post <- comp_V_post(y, X, V_prior)
  
  mu_post <- comp_mu_post(y, X, mu_prior, V_prior, a_prior, lam_prior)
  
  S <- sum(mu_prior * solve(V_prior,mu_prior)) 
  
  S <- S -sum(mu_post * solve(V_post, mu_post))
  
  S <- S + YtY
  
  lam_post <- 0.5*S + lam_prior
  
  return (lam_post)
}

marginal_posterior_parameters_Normal_Linear_Regression <- function(y, X, 
                                                                   mu_prior, Sig2_prior, 
                                                                   phi_prior, psi_prior) {

  phi_post <- comp_a_post( y, X, phi_prior) ;
  
  psi_post <- comp_lam_post(y, X, mu_prior, Sig2_prior, phi_prior, psi_prior) ;
  
  mu_post <- comp_mu_post(y, X, mu_prior, Sig2_prior, phi_prior, psi_prior) ;
  
  Sig2_diag_post <- diag(comp_V_post(y, X, Sig2_prior)) ;
  
  return( list(mu_post=mu_post, 
               Sig2_diag_post=Sig2_diag_post, 
               phi_post=phi_post, 
               psi_post=psi_post)
          ) ;
}

dt_gen <- function(x, df, mu = 0.0, sig2 = 1.0, Qlog = FALSE) {
  # pdf <- dt((x-mu)/sqrt(sig2), df,  log = Qlog)
  # if (Qlog) {
  #   pdf <- pdf + 0.5*log(sig2)
  # } else {
  #   pdf <- pdf * sqrt(sig2)
  # }
  pdf <- dnorm(x, mu, sqrt(sig2),  log = Qlog)
  return( pdf )
}




