#############################################################################
######                        Smoothed circulas                        ######
######                                                                 ######
######       Jose Ameijeiras-Alonso            and    Irène Gijbels    ###### 
###### Universidade de Santiago de Compostela           KU Leuven      ######
######												                   ######
#############################################################################

## Examples of how to use the Supplementary R code for the paper 
## "Smoothed circulas: nonparametric estimation of circular cumulative 
## distribution functions and circulas" 
 

# Note that the libraries circular, cubature, and movMF need to be installed

source("circula_Sup_Mat_Bernoulli.R")


#############################################################################


# For computing the plug-in smoothing parameter for the KCDE

 set.seed(2022)
 x <- cbind(rvonmises(100,circular(0),2),rvonmises(100,circular(pi),5))
 bw.kcde.torus(x) # Plug-in vector kappa when employing the von Mises kernel

#############################################################################
 
 
# For computing the plug-in smoothing parameter for the circula distribution estimation

# We generate data from the Jones et al. (2015) circula (q=1, the binding density is the wrapped Cauchy density)
 x1 <- runif(100,-pi,pi)
 x2t <- rwrappedcauchy(100,circular(0),0.5)
 x2 <- x2t+x1
# For having both in the interval [-pi,pi)
 x2b=x2
 x2[x2b>pi]=x2b[x2b>pi]-2*pi
 x2[x2b<(-pi)]=x2b[x2b<(-pi)]+2*pi
 x <- cbind(x1,x2)
 bw.circula.torus(x) # Plug-in vector kappa when employing the von Mises kernel


 #############################################################################
 


# For computing the circula-marginals distribution estimator

# Random data generated from a toroidal distribution, with the previous circula and marginals being von Mises (see Section 3.4 of Jones et al., 2015)

 y1 <- qvonmises((x1+pi)/(2*pi),circular(0),0.5) 
 y2 <- qvonmises((x2+pi)/(2*pi),circular(0),1.5) 

# We are going to use the von Mises kernel to estimate the circula, but first, we need to estimate the marginals
# Following Section 3.1, we can do two strategies:

# 1. a) Use the same kernel and undersmooth in the estimation of the marginals
 F1us <- kcde.torus(y1, bw="PI.us")
 F2us <- kcde.torus(y2, bw="PI.us")
# We can represent the KCDE in a grid of values between -pi and pi
 par(mfrow=c(1,2))
 plot(F1us, main="First (undersmoothed) marginal estimation")
# We can also add the data values to the plot
 plot(F2us, data.points=T, main="Second (undersmoothed) marginal estimation")

# We will need to obtain the pseudo-observations. The KCDE can be computed in the sample points

 F1a <- kcde.torus(y1, bw="PI.us", grid=matrix(y1,ncol=1))
 F2a <- kcde.torus(y2, bw="PI.us", grid=matrix(y2,ncol=1))


# 1. b) Use a different kernel (the wrapped t with 2 degrees of freedom)

# We evaluate the KCDE in a grid of values between -pi and pi
 F1wt2 <- kcde.torus(y1, kernel="wrappedt2")
 F2wt2 <- kcde.torus(y2, kernel="wrappedt2")

# We evaluate the KCDE in the sample points
 F1b <- kcde.torus(y1, kernel="wrappedt2", grid=matrix(y1,ncol=1))
 F2b <- kcde.torus(y2, kernel="wrappedt2", grid=matrix(y2,ncol=1))

# From the pseudo-observations, we compute the circula distribution estimator

 Ca=kcde.torus(2*pi*cbind(F1a$y,F2a$y)-pi,bw="PI.c")
 Cb=kcde.torus(2*pi*cbind(F1b$y,F2b$y)-pi,bw="PI.c")

# The contour plots of these two estimators can be obtained as follows
 plot(Ca, main="Circula estimation (undersmoothing marginals)")
 plot(Cb, main="Circula estimation (using different kernels)")

#############################################################################
 
 
# We can compute the CDF estimator using the estimated circula and marginals
# Using the undersmoothing approach
 F1 <- kcde.torus(2*pi*cbind(F1a$y,F2a$y)-pi,Ca$bw,grid=2*pi*cbind(F1us$y,F2us$y)-pi)
 seqgrid <- seq(-pi,pi,len=100)
 contour(seqgrid, seqgrid ,F1$y, main="CDF estimation (undersmoothing marginals)")
 
# Using a different kernel
 F2 <- kcde.torus(2*pi*cbind(F1b$y,F2b$y)-pi,Cb$bw,grid=2*pi*cbind(F1wt2$y,F2wt2$y)-pi)
  contour(seqgrid, seqgrid ,F2$y, main="CDF estimation", xaxt="n", yaxt="n")
 
# Comparing with the direct CDF estimator 
# (we can also add the sample points to the graph)
 F3 <- kcde.torus(cbind(y1,y2))
 plot(F3, data.points=T, col=2, add=T)
 legend("bottomleft",c("circula-marginal estimator (using different kernels)", "direct KCDE"),col=1:2, lty=rep(1,2))