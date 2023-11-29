#############################################################################
######                        Smoothed circulas                        ######
######                                                                 ######
######       Jose Ameijeiras-Alonso            and    Irène Gijbels    ###### 
###### Universidade de Santiago de Compostela           KU Leuven      ######
######												                   ######
#############################################################################

## Supplementary R code for the paper "Smoothed circulas: nonparametric 
## estimation of circular cumulative distribution functions and circulas" 

# This script contains the following information:

# 1) Description of the implemented functions 
# 2) Description of their arguments 
# 3) Description of the values they return 
# 4) Examples showing how to use these functions
# 5) Dependences of these functions (these libraries need to be loaded, before
#        using this code)
# 6) Code for the functions described above
# 7) Auxiliary functions needed to execute the described main functions



#############################################################################
#############################################################################


##### 1 ####

## The function bw.kcde.torus computes the plug-in concentration parameter
#    for kernel cumulative distribution estimator (KCDE)
#    for different kernels (see Theorem 5 and Table 2 of the manuscript)

## The function bw.circula.torus computes the plug-in concentration parameter
#    for kernel circula distribution estimator
#    for different kernels (when Corollary 7 applies)

## The function kcde.torus computes the kernel cumulative distribution 
#    estimator. This is valid for both, estimating the KCDE and the circula
#    distribution estimator.

## The function plot.kcde.torus plots the kernel cumulative distribution 
#    estimator. This is valid for both, estimating the KCDE and the circula
#    distribution estimator.

#############################################################################

##### 2 ####

## The common arguments of bw.kcde.torus, bw.circula.torus, and kcde.torus are:

# x: Data from which the smoothing parameter is to be computed. A matrix 
#     of size (n times p), where n is the sample size and p si the number of
#     dimensions. The number of dimensions must be two. If bw.kcde.torus is
#     employed, p can be also equal to one.

# kernel: a character string giving the smoothing kernel to be used. 
#          This must be one of "vonmises", "wrappednormal", or "wrappedt2". 
#          Default kernel="vonmises".

## The common argument of the functions bw.kcde.torus and bw.circula.torus is:

# approximate: logical, if TRUE, the explicit expressions (relying on
#               asymptotics) for the optimal smoothing parameters are employed.
#               See "Optimal smoothing parameter" in Table 2.  If FALSE,
#               an optimization routine is employed, searching for the smoothing
#               parameter minimizing the asymptotic mean squared error of the
#               cumulative distribution function (Equations (2.8) and (2.10) 
#               of Theorem 2 are employed). Default, approximate=T.

## The following arguments are available for bw.kcde.torus

# M: integer indicating the number of components in the mixture. The true
#     unknown toroidal distribution that appears in the formula of the optimal
#     smoothing parameter is replaced by a product of mixtures of M von Mises
#     distributions. If M is a vector, the Akaike Information Criterion will be
#     used, to select the number of components between the indicated values of
#     M. Default, M=1:5.

# undersmooth: logical, if TRUE, a smoothing parameter that undersmooths the
#               cumulative distribution estimation is provided. This reduces 
#               bias of the estimator. This option may be useful if the
#               marginals and the circula are estimated at the same time (see
#               Section 3.1). Default, undersmooth=F.


## The following arguments are available for bw.circula.torus

# binding: a character string giving the binding density in the parametric 
#           circular that is employed to compute the plug-in smoothing 
#           parameter. The true unknown circula distribution that appears in 
#           the formula of the optimal smoothing parameter is replaced by the  
#           Jones et al. (2015) circula. That circula depends on a binding 
#           circular density that can be imposed with the argument binding.
#           This must be one of "vonmises", "wrappednormal", or "wrappedcauchy".
#           Default binding="wrappedcauchy".

# tolint: conververgence tolerance for adaptIntegrate. bw.circula.torus needs to
#          compute a double integral in the expression of the optimal smoothing 
#          parameter. This double integral is computed numerically with the 
#          function adaptIntegrate. 


## The following arguments are available for kcde.torus

# bw: the vector of the smoothing parameters to be used. bw can also be a
#      character string giving the rule to choose the smoothing parameter. 
#      If bw="PI", the plug-in concentration parameter for the KCDE is computed
#      If bw="PI.us", a smoothing parameter that undersmooths the cumulative
#      distribution estimation is computed. If bw="PI.c", the plug-in
#      concentration parameter for the circula distribution estimator is
#      computed. Default, bw="PI".  

# grid: either a matrix or a positive integer. If it is a matrix, the estimator 
#        is computed in the values of grid. In that case, the number of columns
#        should coincide with the number of variables. If it is a single value, 
#        it coincides with the number of equally spaced points at which the 
#        estimator is computed. Default, grid=100.

# zero: if grid is an integer, it gives the leftmost point of the grid at which
#        the estimator is computed. In each dimension, the grid of values is
#        between zero and (zero+2*pi).

# simple.grid: logical, if TRUE, each row of grid is interpreted as a single 
#               vector where the estimator needs to be computed. If FALSE, the
#               estimator is computed for all combinations of the vectors
#               supplied in the columns of grid. Default, simple.grid=F. 

## The following arguments are available for plot.kcde.torus


## x: a "kcde.circular" object, such as the one returned by kcde.torus

## type: in the unidimensional case, the line type. In the bidimensional case,
#         either type="contour", for a contour plot, or type="image", for a 
#         gridof colored rectangles with colors corresponding to the values of 
#         the estimated distribution. The plot.kcde.torus is not yet available 
#         for a dimension greater than 2. 

# data.points: logical, if TRUE, the data values are included on the plot.

# main, xlab, ylab, xaxt, yaxt:	plotting parameters with useful defaults.
# ...:	 further plotting parameters can be included following the arguments of 
#         the function plot.default (for the unidimensional case), contour, or 
#         image.

#############################################################################


##### 3 ####


##  The functions bw.kcde.torus and bw.circula.torus return the vector of the 
#     smoothing parameters. The same value is returned for each dimension 
#     When the kernel is vonmises the returned value is equal to the
#      concentration parameter kappa.
#     When the kernel is wrapped normal the returned value is equal to the
#      concentration parameter rho (mean resultant length).
#     When the kernel is wrapped t2 the returned value is equal to the
#      dispersion parameter lambda.

##  The function kcde.torus returns: 

# x: the coordinates of the points where the density is estimated.
# y: the values of the estimated distribution function.
# bw: the smoothing parameter used.
# n: the sample size.
# ndim: the number of dimensions.
# call:	the call which produced the result.
# data: the original data from which the density is computed.
# data.name: the deparsed name of the x argument.


#############################################################################



##### 4 ####


## Examples

# For computing the plug-in smoothing parameter for the KCDE

# set.seed(2022)
# x <- cbind(rvonmises(100,circular(0),2),rvonmises(100,circular(pi),5))
# bw.kcde.torus(x) # Plug-in vector kappa when employing the von Mises kernel


# For computing the plug-in smoothing parameter for the circula distribution estimation

# We generate data from the Jones et al. (2015) circula (q=1, the binding density is the wrapped Cauchy density)
# x1 <- runif(100,-pi,pi)
# x2t <- rwrappedcauchy(100,circular(0),0.5)
# x2 <- x2t+x1
# For having both in the interval [-pi,pi)
# x2b=x2
# x2[x2b>pi]=x2b[x2b>pi]-2*pi
# x2[x2b<(-pi)]=x2b[x2b<(-pi)]+2*pi
# x <- cbind(x1,x2)
# bw.circula.torus(x) # Plug-in vector kappa when employing the von Mises kernel






# For computing the circula-marginals distribution estimator

# Random data generated from a toroidal distribution, with the previous circula and marginals being von Mises (see Section 3.4 of Jones et al., 2015)

# y1 <- qvonmises((x1+pi)/(2*pi),circular(0),2) 
# y2 <- qvonmises((x2+pi)/(2*pi),circular(0),5) 

# We are going to use the von Mises kernel to estimate the circula, but first, we need to estimate the marginals
# Following Section 3.1, we can do two strategies:

# 1. a) Use the same kernel and undersmooth in the estimation of the marginals
# F1us <- kcde.torus(y1, bw="PI.us")
# F2us <- kcde.torus(y2, bw="PI.us")
# We can represent the KCDE in a grid of values between -pi and pi
# par(mfrow=c(1,2))
# plot(F1us, main="First (undersmoothed) marginal estimation")
# We can also add the data values to the plot
# plot(F2us, data.points=T, main="Second (undersmoothed) marginal estimation")

# We will need to obtain the pseudo-observations. The KCDE can be computed in the sample points

# F1a <- kcde.torus(y1, bw="PI.us", grid=matrix(y1,ncol=1))
# F2a <- kcde.torus(y2, bw="PI.us", grid=matrix(y2,ncol=1))


# 1. b) Use a different kernel (the wrapped t with 2 degrees of freedom)

# We evaluate the KCDE in a grid of values between -pi and pi
# F1wt2 <- kcde.torus(y1, kernel="wrappedt2")
# F2wt2 <- kcde.torus(y2, kernel="wrappedt2")

# We evaluate the KCDE in the sample points
# F1b <- kcde.torus(y1, kernel="wrappedt2", grid=matrix(y1,ncol=1))
# F2b <- kcde.torus(y2, kernel="wrappedt2", grid=matrix(y2,ncol=1))

# From the pseudo-observations, we compute the circula distribution estimator

# Ca=kcde.torus(2*pi*cbind(F1a$y,F2a$y)-pi,bw="PI.c")
# Cb=kcde.torus(2*pi*cbind(F1b$y,F2b$y)-pi,bw="PI.c")

# The contour plots of these two estimators can be obtained as follows
# plot(Ca, main="Circula estimation (undersmoothing marginals)")
# plot(Cb, main="Circula estimation (using different kernels)")

# We can compute the CDF estimator using the estimated circula and marginals
# Using the undersmoothing approach
# F1 <- kcde.torus(2*pi*cbind(F1a$y,F2a$y)-pi,Ca$bw,grid=2*pi*cbind(F1us$y,F2us$y)-pi)
# seqgrid <- seq(-pi,pi,len=100)
# contour(seqgrid, seqgrid ,F1$y, main="CDF estimation (undersmoothing marginals)")


# Using a different kernel
# F2 <- kcde.torus(2*pi*cbind(F1b$y,F2b$y)-pi,Cb$bw,grid=2*pi*cbind(F1wt2$y,F2wt2$y)-pi)
#  contour(seqgrid, seqgrid ,F2$y, main="CDF estimation", xaxt="n", yaxt="n")

# Comparing with the direct CDF estimator 
# (we can also add the sample points to the graph)
# F3 <- kcde.torus(cbind(y1,y2))
# plot(F3, data.points=T, col=2, add=T)
# legend("bottomleft",c("circula-marginal estimator (using different kernels)", "direct KCDE"),col=1:2, lty=rep(1,2))


#############################################################################
#############################################################################


##### 5 ####


# Some of the internal functions of these libraries are needed to compute the implemented functions for this supplementary material

library(movMF) # movMF
library(circular) # several functions
library(cubature) # adaptIntegrate



#############################################################################
#############################################################################

##### 6 ####


bw.kcde.torus=function(x, kernel="vonmises", approximate=TRUE, M=NULL, undersmooth=FALSE){
  
  x=as.matrix(x)
  dimdata=dim(x)[2]
  ndata=dim(x)[1]
  
  # For the plug-in rule F is replaced by a mixture of von Mises
  
  # First, the number of components M is determined
  # The maximum likelihood estimates of the von Mises mixture are saved in meanflist (vector of means), kappaflist (vector of concentrations) and pflist (vector of weights)
  
  meanflist=list()
  kappaflist=list()
  pflist=list()
  
  if(is.null(M)){
    M=1:5
  }
  
  for(j in 1:dimdata){
    
    meanf=numeric()
    kappaf=numeric()
    AICf=Inf
    pf=0
    
    xc=circular(x[,j])
    
    for(indM in M){
      
      if(indM==1){
        
        mlevm=mle.vonmises(xc)
        mean=mlevm$mu
        kappa=mlevm$kappa
        AIC <-dvonmises(xc, circular(mean),kappa)
        AIC <- -2 * sum(log(AIC)) + 4*indM + 2*(indM-1)
        
        if(AIC<AICf){
          AICf=AIC
          meanf=mean
          kappaf=kappa
          pf=1
        }
        
      }else{
        
        z <- cbind(cos(x[,j]), sin(x[,j]))
        y2 <- try(movMF(z, indM, start = "S"), TRUE)
        norm <- mean <- kappa <- p <- numeric(indM)
        mu <- matrix(NA, indM, 2)
        AIC <- 0
        
        if(class(y2) != "try-error"){
          for (i in 1:indM) {
            norm[i] <- sqrt(sum(y2$theta[i, ]^2))
            mu[i, ] <- y2$theta[i, ]/norm[i]
            mean[i] <- atan2(mu[i, 2], mu[i, 1])
            kappa[i] <- y2$theta[i, 1]/mu[i, 1]
            p[i] <- y2$alpha[i]
            AIC <- AIC + p[i] * dvonmises(xc, circular(mean[i]),kappa[i])
          }
          AIC <- -2 * sum(log(AIC)) + 4*indM + 2*(indM-1)
          
          # We restrict the value of kappa to avoid errors in the numerical integration
          kappamax=700
          if(sum(kappa>kappamax)>0){AIC=Inf}
          
          if(AIC<AICf){
            AICf=AIC
            meanf=mean
            kappaf=kappa
            pf=p
          }
          
        }
        
      }
    }
    
    meanflist[[j]]=meanf
    kappaflist[[j]]=kappaf
    pflist[[j]]=pf
    
  }
  

  # In the following, the values of F and its partial derivatives are computed in the vector theta and in the vector equal to theta except in one component where it is equal to -pi
  
  fsmpi=numeric()
  fsmpi1=numeric()
  fsmpi2=numeric()
  
  for(j in 1:dimdata){
    fsmpi1[j]=0
    for(i in 1:length(meanflist[[j]])){
      fsmpi1[j]=fsmpi1[j]+pflist[[j]][i]*dvonmises(circular(-pi),circular(meanflist[[j]][i]),kappaflist[[j]][i])
    }
  }
  
  fsmpi1=1+2*pi*fsmpi1
  
  if(dimdata==1){
    fsmpi=fsmpi1
  }else{
    for(j in 1:dimdata){
      fsmpi2[j]=0
      for(i in 1:length(meanflist[[j]])){
        fsmpi2[j]=fsmpi2[j]+pflist[[j]][i]*integrate(function(px) pvonmises3(px,meanflist[[j]][i],kappaflist[[j]][i]),-pi,pi)$v
      }
    }
    
    for(j in 1:dimdata){
      fsmpi[j]=fsmpi1[j]*fsmpi2[-j]
    }
  }
  
  dvmp2=function(x,j){
    res=0
    for(i in 1:length(meanflist[[j]])){
      res=res-pflist[[j]][i]*kappaflist[[j]][i]*sin(x-meanflist[[j]][i])*exp(kappaflist[[j]][i]*cos(x-meanflist[[j]][i]))/(2*pi*besselI(kappaflist[[j]][i],0))
      res=res+pflist[[j]][i]*kappaflist[[j]][i]*sin(-pi-meanflist[[j]][i])*exp(kappaflist[[j]][i]*cos(-pi-meanflist[[j]][i]))/(2*pi*besselI(kappaflist[[j]][i],0))
    }
    return(res^2)
  }
  
  pvm2=function(x,j){
    res=0
    for(i in 1:length(meanflist[[j]])){
      res=res+pflist[[j]][i]*pvonmises3(x,meanflist[[j]][i],kappaflist[[j]][i])
    }
    return(res^2)
  }
  
  proddvmppvm=function(x,j){
    res1=0
    for(i in 1:length(meanflist[[j]])){
      res1=res1-pflist[[j]][i]*kappaflist[[j]][i]*sin(x-meanflist[[j]][i])*exp(kappaflist[[j]][i]*cos(x-meanflist[[j]][i]))/(2*pi*besselI(kappaflist[[j]][i],0))
      res1=res1+pflist[[j]][i]*kappaflist[[j]][i]*sin(-pi-meanflist[[j]][i])*exp(kappaflist[[j]][i]*cos(-pi-meanflist[[j]][i]))/(2*pi*besselI(kappaflist[[j]][i],0))
    }
    res2=0
    for(i in 1:length(meanflist[[j]])){
      res2=res2+pflist[[j]][i]*pvonmises3(x,meanflist[[j]][i],kappaflist[[j]][i])
    }
    return(res1*res2)
  }
  
  if22=numeric()
  if(dimdata==1){
    if22=integrate(function(x)dvmp2(x,1),-pi,pi)$val
  }
  
  
  if(dimdata==2){
    
    if22a=numeric()
    if22b=numeric()
    if22c=numeric()
    
    for(j in 1:2){
      if22a[j]=integrate(function(x)dvmp2(x,j),-pi,pi)$val
      if22b[j]=integrate(function(x)pvm2(x,j),-pi,pi)$val
      if22c[j]=integrate(function(x)proddvmppvm(x,j),-pi,pi)$val
    }
    
    if22[1]=if22a[1]*if22b[2]
    if22[2]=if22a[2]*if22b[1]
    if22[3]=2*prod(if22c)
    
    
  }
  
  # Since the following part is common for the plug-in smoothing parameter of the KCDE and the circula, we provide the next part in the function bw.both.torus
  kamise=bw.both.torus(ndata, dimdata, kernel, approximate, undersmooth,fsmpi,if22)
  return(kamise)
}

  


#############################################################################


bw.circula.torus=function(x,kernel="vonmises",binding="wrappedcauchy", approximate=TRUE,tolint=1e-03){
  
  # For the plug-in rule C is replaced by the Jones et al., 2015 circula
  # Their circula depends on:
  # -The binding density g, specified in the argument binding
  # -The value of q, which is equal to 1 (positive) or -1 (negative dependence)
  
  # This function works if mle.binding is available. The following are available in the circular R package: mle.vonmises, mle.wrappednormal, and mle.wrappedcauchy
  
  # First, it computes the maximum likelihood estimates of the parameters of g, after chooses the value of q from the largest maximized likelihood over the two possible values
  
  gfunc=function(x) eval(parse(text=paste("mle.",binding,"(circular(x))",sep="")))
  q=1
  mleval1=gfunc(x[,2]-q*x[,1])
  if(is.null(mleval1$kappa)){mleval1$kappa=mleval1$rho}
  q=-1
  mlevalm1=gfunc(x[,2]-q*x[,1])
  if(is.null(mlevalm1$kappa)){mlevalm1$kappa=mlevalm1$rho}
  
  if(is.null(mleval1$AIC)){
    ffunc1=function(x) eval(parse(text=paste("d",binding,"(circular(x),mleval1$mu,mleval1$kappa)",sep="")))
    valmll1=sum(log(ffunc1(x[,2]-x[,1])))
    ffunc2=function(x) eval(parse(text=paste("d",binding,"(circular(x),mlevalm1$mu,mlevalm1$kappa)",sep="")))
    valmll2=sum(log(ffunc2(x[,2]+x[,1])))
    
    if(valmll1>valmll2){
      qf=1
      mleval=mleval1
    }else{
      qf=-1
      mleval=mlevalm1
    }
    
  }else{
    
    if(mleval1$AIC<mlevalm1$AIC){
      qf=1
      mleval=mleval1
    }else{
      qf=-1
      mleval=mlevalm1
    }
    
  }
  

  
  
  if22=intC2(binding,mleval$mu,mleval$kappa,qf,tol=tolint)
  fsmpi=csmpi(binding,mleval$mu,mleval$kappa,qf)
  dimdata=2 
  ndata=dim(x)[1]
  undersmooth=FALSE
  
  
  kamise=bw.both.torus(ndata, dimdata, kernel, approximate, undersmooth, fsmpi,if22)
  return(kamise)
  
  
}




#############################################################################

kcde.torus=function(x,bw="PI",kernel="vonmises",grid=100,zero=-pi,simple.grid=F){

  if (is.character(bw)){
    bw <- switch(tolower(bw), pi = bw.kcde.torus(x,kernel=kernel), pi.us=bw.kcde.torus(x,kernel=kernel,undersmooth=T), pi.c=bw.circula.torus(x,kernel=kernel), stop("unknown bandwidth rule"))
  }  


  name=deparse1(substitute(x))
  x=as.matrix(x)
  dimdata=dim(x)[2]
  ndata=dim(x)[1]
  eval.points=grid
  
  if(length(grid)==1){
    eval.points=matrix(seq(zero,zero+2*pi,len=grid),nrow=grid,ncol=dimdata)
  }
  
  mFjp=list()
  
  if(kernel=="wrappedcauchy"){
    for(j in 1:dimdata){
      mseqxd=outer(eval.points[,j],x[,j],"-")
      mseqxd2=outer(rep(-pi,length(eval.points[,j])),x[,j],"-")
      mFjp[[j]]= pwrappedcauchyc(mseqxd,bw[j])-pwrappedcauchyc(mseqxd2,bw[j])
    }
  }
  if(kernel=="vonmises"){
    for(j in 1:dimdata){
      mseqxd=outer(eval.points[,j],x[,j],"-")
      mseqxd2=outer(rep(-pi,length(eval.points[,j])),x[,j],"-")
      mFjp[[j]]=matrix(pvonmisesc(mseqxd,bw[j]),ncol=ndata)-matrix(pvonmisesc(mseqxd2,bw[j]),ncol=ndata)
    }
  }
  if(kernel=="wrappedt2"){
    for(j in 1:dimdata){
      mseqxd=outer(eval.points[,j],x[,j],"-")
      mseqxd2=outer(rep(-pi,length(eval.points[,j])),x[,j],"-")
      mFjp[[j]]=matrix(pwrappedtc(mseqxd,bw[j]),ncol=ndata)-matrix(pwrappedtc(mseqxd2,bw[j]),ncol=ndata)
    }
  }
  
  if(kernel=="wrappednormal"){
    for(j in 1:dimdata){
      mseqxd=outer(eval.points[,j],x[,j],"-")
      mseqxd2=outer(rep(-pi,length(eval.points[,j])),x[,j],"-")
      mFjp[[j]]=matrix(pwrappednormalc(mseqxd,bw[j]),ncol=ndata)-matrix(pwrappednormalc(mseqxd2,bw[j]),ncol=ndata)
    }
  }
  
  if(dimdata==1){
    estimate=rowSums(mFjp[[1]])/ndata
  }
  
  if(dimdata>1){
    
    if(simple.grid==F){
      estimate=array(0,rep(dim(eval.points)[1],dim(eval.points)[2]))
      
      
      for(i in 1:ndata){
        
        estimatet=mFjp[[1]][,i]
        
        for(j in 1:(dimdata-1)){
          estimatet=outer(estimatet,mFjp[[j+1]][,i])
        }
        
        estimate=estimate+estimatet
        
      }
      
    }else{
      
      estimate=mFjp[[1]]
      
      for(j in 1:(dimdata-1)){
        estimate=estimate*mFjp[[j+1]]
      }
      
      estimate=rowSums(estimate)
      
    }
    
    estimate=estimate/ndata
    
  }
  

  structure(list(x=eval.points, y=estimate, bw=bw, n=ndata, ndim=dimdata, call = match.call(),data=x, data.name = name),class="kcde.circular")
  
}


#############################################################################

plot.kcde.circular=function(x, type = NULL,data.points=F, main = NULL, xlab = NULL, ylab = NULL, xaxt = NULL, yaxt = NULL, ...){
  
  if(x$ndim>2){stop("plot not yet implemented for a number of dimensions greater than 2")}
  
  if(x$ndim==1){
    if (is.null(xlab)){ xlab <- paste("N =", x$n, "  Smoothing parameter =",formatC(x$bw))}
    if (is.null(ylab)){ ylab <- "Kernel cumulative distribution estimator"}
    if (is.null(xaxt)){ xaxt <- "n"}
    if (is.null(yaxt)){ yaxt <- "s"}
    if (is.null(main)){ main <- deparse(x$call)}
    if (is.null(type)){ type <- "l"}
    plot.default(x$x,x$y, main = main, xlab = xlab, ylab = ylab, type = type, xaxt=xaxt, yaxt=yaxt, ...)
    if (xaxt == "n"){
      if(sum(abs(range(x$x[,1])-c(-pi,pi)))<10^(-6)){
        axis(1,c(-pi,-pi/2,0,pi/2,pi),c(expression(-pi),expression(-pi/2),0,expression(pi/2),expression(pi)))
      }else if(sum(abs(range(x$x[,1])-c(0,2*pi)))<10^(-6)){
        axis(1,c(0,pi/2,pi,3*pi/2,2*pi),c(0,expression(pi/2),expression(pi),expression(3*pi/2),expression(2*pi)))
      }else{
          axis(1)
      }
    }
    if(data.points==T){rug(x$data)}
    
      
  }
  
  
  if(x$ndim==2){
    if (is.null(main)){ main <- deparse(x$call)}
    xlab2 <- NULL
    if (is.null(xlab)){ 
      xlab <- paste("N =", x$n, "  Smoothing parameter =",formatC(x$bw[1]))
      xlab2 <- "x1"
    }
    if (is.null(ylab)){ ylab <- "x2"}
    if (is.null(type)){ type <- "contour"}
    if (is.null(xaxt)){ xaxt <- "n"}
    if (is.null(yaxt)){ yaxt <- "n"}
    if (type=="contour"){contour(x$x[,1],x$x[,2],x$y, main = main, xlab = xlab, ylab = ylab, xaxt=xaxt, yaxt=yaxt,  ...)}
    if (type=="image"){image(x$x[,1],x$x[,2],x$y, main = main, xlab = xlab, ylab = ylab, xaxt=xaxt, yaxt=yaxt, ...)}
    
    if (!is.null(xlab2)){ mtext(xlab2, side=1, line=2,cex=1)}
    
    if (xaxt == "n"){
      if(sum(abs(range(x$x[,1])-c(-pi,pi)))<10^(-6)){
        axis(1,c(-pi,-pi/2,0,pi/2,pi),c(expression(-pi),expression(-pi/2),0,expression(pi/2),expression(pi)))
      }else if(sum(abs(range(x$x[,1])-c(0,2*pi)))<10^(-6)){
        axis(1,c(0,pi/2,pi,3*pi/2,2*pi),c(0,expression(pi/2),expression(pi),expression(3*pi/2),expression(2*pi)))
      }else{
        axis(1)
      }
    }
    if (yaxt == "n"){
      if(sum(abs(range(x$x[,2])-c(-pi,pi)))<10^(-6)){
        axis(2,c(-pi,-pi/2,0,pi/2,pi),c(expression(-pi),expression(-pi/2),0,expression(pi/2),expression(pi)))
      }else if(sum(abs(range(x$x[,2])-c(0,2*pi)))<10^(-6)){
        axis(2,c(0,pi/2,pi,3*pi/2,2*pi),c(0,expression(pi/2),expression(pi),expression(3*pi/2),expression(2*pi)))
      }else{
        axis(2)
      }
    }
    if(data.points==T){points(x$data,pch=16,cex=0.6)}
  }
  
}


#############################################################################
#############################################################################


##### 7 ####

## Other auxiliary functions



# This function computes the optimal smoothing parameter for both, the KCDE and the circula once the values depending on the true F or C are provided
# fsmpi is a vector containing the quantities that appear in the numerator
# if22 is a vector containing the quantities that appear in the denominator

bw.both.torus=function(ndata, dimdata, kernel="vonmises", approximate=TRUE, undersmooth=FALSE,fsmpi,if22){
  
  # Plug-in value of h
  
  if(kernel=="vonmises"|kernel=="wrappednormal"){
    hamise=sum(fsmpi)/(pi^(1/2)*ndata*sum(if22))
    hamise=hamise^(2/3)
  }
  
  if(kernel=="wrappedt2"){
    hamise=(pi^2*sqrt(2)/4)*sum(fsmpi)/(4*pi*ndata*sum(if22))
  }
  
  
  
  # From h, the optimal value of the smoothing parameter (kappa, rho or lamba) is computed 
  
  kamise=numeric()
  
  if(kernel=="wrappednormal"|kernel=="vonmises"){
    
    
    if(hamise>=pi^2/3){
      kamise=0
    }else{
      
      if(kernel=="wrappednormal"){
        hamisefunc=function(kappa){ 
          res=pi^2/3
          seqj=1:1000
          res=res+4*rowSums(outer(kappa,seqj,function(kappa,jvec)(-1)^jvec*kappa^(jvec^2)/(jvec^2)))
          return(res)
        }
      }
      
      if(kernel=="vonmises"){
        
        hamisefunc=function(kappa){ 
          
          tol=10^(-20)
          res=pi^2/3
          seqjt=0
          
          seqj=1:1000
          rest=4*rowSums(outer(kappa,seqj,function(kappa,jvec)(-1)^jvec*besselI(kappa,jvec, expon.scaled = FALSE)/besselI(kappa,0, expon.scaled = FALSE)/(jvec^2)))
          res=res+rest
          
          while((max(abs(rest))>tol)&(seqj[1]<10^(7))){
            seqj=seqj+1000
            rest=4*rowSums(outer(kappa,seqj,function(kappa,jvec)(-1)^jvec*besselI(kappa,jvec, expon.scaled = FALSE)/besselI(kappa,0, expon.scaled = FALSE)/(jvec^2)))
            res=res+rest
          }
          
          
          res[is.na(res)]=(1/kappa)[is.na(res)]
          return(res)
        }
        
        
        
        
        m1amisefunc=function(kappa){ 
          
          tol=10^(-20)
          res=0
          seqjt=0
          
          seqj=1:1000
          alphaj=t(outer(kappa,seqj,function(kappa,seqj) besselI(kappa,seqj,expon.scaled = T)/(besselI(kappa,0,expon.scaled = T))))
          rest=colSums(2*(-1)^seqj*alphaj/seqj^2+alphaj^2/seqj^2)
          res=res+rest
          
          while((max(abs(rest))>tol)&(seqj[1]<10^(8))){
            seqj=seqj+1000
            alphaj=t(outer(kappa,seqj,function(kappa,seqj) besselI(kappa,seqj,expon.scaled = T)/(besselI(kappa,0,expon.scaled = T))))
            rest=colSums(2*(-1)^seqj*alphaj/seqj^2+alphaj^2/seqj^2)
            res=res+rest
          }
          
          res[is.na(res)]=(1/kappa)[is.na(res)]
          return(-res)
        }
        
      }
      
      if(kernel=="vonmises"){
        
        if(approximate==F){
          if(hamise>hamisefunc(709)){
            
            amise=function(kappa){
              hamiset=hamisefunc(kappa)
              return(hamiset^2*sum(if22)/4-(hamiset+m1amisefunc(kappa)*sum(fsmpi))/(pi*ndata))
            }
            kamise=try(optimize(amise,c(0,709))$minimum)
          }else{
            kamise=1/hamise
          }
        }else{
          kamise=1/hamise
        }
      }
      
      
      
      
      if(kernel=="wrappednormal"){
        if(approximate==F){
          kamise=try(uniroot(function(kappa)hamisefunc(kappa)-hamise,c(0,1))$root)
        }else{
          kamise=(1-2*hamise)^(1/4)
        }
      }
      
      if(class(kamise)=="try-error"){
        
        if(kernel=="vonmises"){
          kamise=1/hamise
        }
        if(kernel=="wrappednormal"){
          kamise=(1-2*hamise)^(1/4)
        }
        
      }
    }
  }
  
  if(kernel=="wrappedt2"){
    
    
    hamisefunc=function(kappa){ 
      
      tol=10^(-20)
      res=pi^2/3
      seqjt=0
      
      seqj=1:10000
      rest=4*rowSums(outer(kappa,seqj,function(kappa,jvec)(-1)^jvec*besselK(jvec*kappa*sqrt(2),1)*kappa*sqrt(2)/(jvec)))
      res=res+rest
      
      while((max(abs(rest))>tol)&(seqj[1]<10^(8))){
        seqj=seqj+10000
        rest=4*rowSums(outer(kappa,seqj,function(kappa,jvec)(-1)^jvec*besselK(jvec*kappa*sqrt(2),1)*kappa*sqrt(2)/(jvec)))
        res=res+rest
      }
      
      res[is.na(res)]=(1/kappa)[is.na(res)]
      return(res)
    }
    
    
    m1amisefunc=function(kappa){ 
      
      tol=10^(-20)
      res=0
      seqjt=0
      
      seqj=1:10000
      alphaj=t(outer(kappa,seqj,function(kappa,seqj)besselK(seqj*kappa*sqrt(2),1)*seqj*kappa*sqrt(2)))
      rest=colSums(2*(-1)^seqj*alphaj/seqj^2+alphaj^2/seqj^2)
      res=res+rest
      
      while((max(abs(rest))>tol)&(seqj[1]<10^(8))){
        seqj=seqj+10000
        alphaj=t(outer(kappa,seqj,function(kappa,seqj)besselK(seqj*kappa*sqrt(2),1)*seqj*kappa*sqrt(2)))
        rest=colSums(2*(-1)^seqj*alphaj/seqj^2+alphaj^2/seqj^2)
        res=res+rest
      }
      
      res[is.na(res)]=(1/kappa)[is.na(res)]
      return(-res)
    }
    
    amise=function(kappa){
      hamiset=hamisefunc(kappa)
      return(hamiset^2*sum(if22)/4-(hamiset+m1amisefunc(kappa)*sum(fsmpi))/(pi*ndata))
    }
    
    approximate2=approximate
    if(hamise >(4*exp(-2)/9)){approximate2=F}
    
    if(approximate2==F){
      kamise=try(optimize(amise,c(0,10))$minimum)
      
      if(class(kamise)=="try-error"){
        
        kamise=try(uniroot(function(lambda) lambda^3*log(lambda)^2-hamise,c(10^(-100),1/exp(2/3)))$root)
        
      }    
    }else{
      kamise=try(uniroot(function(lambda) lambda^3*log(lambda)^2-hamise,c(10^(-100),1/exp(2/3)))$root)
    }
    
    if(class(kamise)=="try-error"){kamise=exp(-2/3)}
    
    
    
    
    
  }
  
  if(undersmooth==TRUE){
    if(kernel=="vonmises"){
      kamise=kamise*log(ndata)
    }
    if(kernel=="wrappednormal"){
      hamiset=((1-kamise^4)/2)/log(ndata)
      kamise=(1-2*hamiset)^(1/4)
    }
    if(kernel=="wrappedt2"){
      kamise=kamise/log(ndata)
    }
  }
  
  
  kamise=rep(kamise,dimdata)
  
  return(kamise)
}


#############################################################################



## This function computes the cdf of a von Mises distribution with F(mu)=0

pvonmises2=function(x,mu=0,kappa){
  nw=floor((x+pi-mu)/(2*pi))
  x=x-nw*2*pi
  
  if(kappa>0){
    tol = 1e-20
    sumpvm=rep(0,length(x))
    pvals=1:100
    control=0
    while(control==0){
      frac1=besselI(kappa,pvals,expon.scaled = T)/(pvals*besselI(kappa,0,expon.scaled = T))
      sinpx=outer(pvals,x-mu,function(u1,u2) sin(u1*u2))
      sumpvm=sumpvm+colSums(frac1*sinpx)
      pvals=pvals+100
      if(frac1[100]<tol){control=1}
    }
  }else{
    sumpvm=0
  }
  
  Fres=(x-mu+2*sumpvm)/(2*pi)
  
  Fres=Fres+nw
  return(Fres)
}

## This function computes the cdf of a von Mises distribution with F(-pi)=0

pvonmises3=function(x,mu=0,kappa){ pvonmises2(x,mu,kappa)-pvonmises2(-pi,mu,kappa)}


## This function computes the cdf of a wrapped Cauchy distribution with F(mu)=0

pwrappedcauchy=function(x,mu=0,rho){
  nw=floor((x+pi-mu)/(2*pi))
  x=x-nw*2*pi
  sgnd=sign(x-mu)
  Fres=as.double(sgnd)*(acos(((1+rho^2)*cos(x-mu)-2*rho)/(1+rho^2-2*rho*cos(x-mu)))/(2*pi))
  Fres=Fres+nw
  return(Fres)
}


## This function computes the cdf of a wrapped student t distribution with F(mu)=0 (mu is the location parameter and lambda, the dispersion, see Table 1)

pwrappedt=function(x,mu=0,lambda,eta=2){
  nw=floor((x+pi-mu)/(2*pi))
  x=x-nw*2*pi
  
  tol = 1e-20
  sumpvm=rep(0,length(x))
  pvals=1:100
  denom=gamma(eta/2)*2^(eta/2-1)
  control=0
  while(control==0){
    frac1=besselK(pvals*lambda*sqrt(eta),1)*(pvals*lambda*sqrt(eta))^(eta/2)/(pvals*denom)
    sinpx=outer(pvals,x-mu,function(u1,u2) sin(u1*u2))
    sumpvm=sumpvm+colSums(frac1*sinpx)
    pvals=pvals+100
    if(frac1[100]<tol){control=1}
  }
  
  
  Fres=(x-mu+2*sumpvm)/(2*pi)
  
  Fres=Fres+nw
  return(Fres)
}


## This function computes the cdf of a wrapped normal distribution with F(mu)=0

pwrappednormal2=function(x,mu=0,rho){
  nw=as.double(floor((x+pi-mu)/(2*pi)))
  x=x-nw*2*pi
  Fres=pwrappednormal(x,mu,rho)
  Fres=Fres+nw
  return(Fres)
}

# Equivalent functions, employed for the kcde estimation F(-pi)=0, all centred at 0
pwrappedcauchyc=function(x,rho){ pwrappedcauchy(x,0,rho)+0.5}
pvonmisesc=function(x,kappa){ pvonmises2(x,0,kappa)+0.5}
pwrappednormalc=function(x,rho){ pwrappednormal2(circular(x),circular(0),rho)}
pwrappedtc=function(x,lambda,eta=2){ pwrappedt(x,mu=0,lambda,eta=2)+0.5}


## This function computes the cdf of a cardioid distribution with F(mu)=0

pcardioid=function(x,mu=0,rho){  (x-mu+2*rho*(sin(x-mu)))/(2*pi) }


#############################################################################


# These two functions compute the integral of quantities related to the second partial derivative of the circula C that appear in the expression of the optimal smoothing parameter.

# This function computes the quantities in the denominator of the optimal h

intC2=function(binding="wrappedcauchy",mu,kappa,q,tol=1e-05){

  ffunc=function(x) eval(parse(text=paste("d",binding,"(circular(x),mu,kappa)",sep="")))
  pfunc=function(x) eval(parse(text=paste("p",binding,"(circular(x),mu,kappa)",sep="")))
  
  ff0=ffunc(-pi+q*pi)
  C2a=function(x,y) (ffunc(y-q*x))^2
  C2av=adaptIntegrate(function(t)C2a(t[1],t[2]), c(-pi, -pi), c(pi, pi), tol = tol)$integral
  C2e=function(x) ffunc(-pi-q*x)^2
  C2ev=integrate(C2e,-pi,pi)$val
  C2f=function(y) ffunc(y+q*pi)^2
  C2fv=integrate(C2f,-pi,pi)$val
  
  C2val=(C2av -2 -2 +4*pi*ff0 + 2*pi*C2ev + 2 - 4*pi*ff0 + 2*pi*C2fv -4*pi*ff0+4*pi^2*ff0^2)/(4*pi^2)
  
  
  return(c(C2val,2*C2val,C2val))

}

# This function computes the quantities in the numerator of the optimal h


csmpi=function(binding="wrappedcauchy",mu,kappa,q){
    pfunc=function(x) eval(parse(text=paste("p",binding,"w(x,mu,kappa)",sep="")))
  
  
  csmpi1=function(y) (pfunc(y+q*pi)-pfunc(-pi+q*pi))
  val1=pi+integrate(csmpi1,-pi,pi)$v
  
  csmpi2=function(x) -q*(pfunc(-pi-q*x)-pfunc(-pi+q*pi))
  val2=pi+integrate(csmpi2,-pi,pi)$v
  
  return(c(val1,val2))
  
}

# Needed to compute the previous function

pvonmisesw=function(x,mu=0,kappa){ pvonmises2(x,mu,kappa) }
pwrappedcauchyw=function(x,mu=0,kappa){ pwrappedcauchy(x,mu,kappa) }
pcardioidw=function(x,mu=0,kappa){ pcardioid(x,mu,kappa) }
pwrappednormalw=function(x,mu=0,kappa){ pwrappednormal(x,mu,kappa) }
