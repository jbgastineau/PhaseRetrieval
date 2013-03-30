#################################################################################
#
#   File:       LevenbergPhaseRetrieval.py
#
#   Summary:    A collection of functions/methods that can be used to perform image
#               analysis.  DFT, filters, convolutions, covariance analysis and etc
#
#
#   Author:     Robert Upton (Sigma Space Corporation)
#
#   Date:       April 10, 2012
#
#################################################################################

import numpy as num
import pylab as py
from scipy import optimize
import ImageScienceToolBox as ISTB
import pdb

# setup the psf function and error function
#-------------------------------------------------
def myfunc(zVec):
    """the model function"""
    zernCalc = lambda RR, TH, mask, nterm: self.waveCreate(zVec)
    waveFunc = zernCalc(RR,TH,mask,nterm) 
    self.psfOTF(mask,waveFunc.waveFunc)
    psf = self.psf
    OTF = self.OTF
    OTF = (OTF==0).choose(OTF,1e-011)
    return psf, OTF

#-------------------------------------------------
def errorfunc(zVec,data):
    """the objective function"""
    errorfunc     = lambda zVec: num.ravel(myfunc(zVec)[0] - data)
    pdb.set_trace()
    result = optimize.leastsq(errorfunc, zVec, full_output=1) 
    return result

#-------------------------------------------------
def LMphaseRetreival(zStart,data):
    """function that uses parametric phase retrieval
        approach
        This function calls the ImageScienceToolBox
        and assigns the class ImageScienceClass to self
        
        self = ISTB.ImageScienceClass()
        
        The image science toolbox (ISTB) contains filtering
        functions, psf and OTF generating functions, RMS
        Zernike weighted functions and etc
        """
    # assign the ImageScienceClass to self and calculate the 
    # compact support (pupil), OTF, and resulting PSF
    # initialization
    # perform the Levenberg-Marquardt optimization
    parms      = errorfunc(zVecStart,data)
    zVecRes    = parms[0]
    zVecRes[0] = 0
    
    print '#----------------------------'
    print 'parameters for the best fit: '
    kk = 0
    for ii in zVecRes:
        strVal = 'Z' + str(kk) + '  = ' + '%6.3f '%ii
        print strVal
        kk = kk + 1
    
    numEvals = parms[2].__getitem__('nfev')
    print 'number of function evaluations = ' + '%i'%(numEvals)
    
    return zVecRes, numEvals

#-------------------------------------------------
if __name__ == '__main__':
    """main calling function of the file
       The Levenberg-Marquardt non-linear optimization
       routine is called with starting defocus phase diversity
       to determine the PSF phase error.  Current implementation
       uses the MSE as a merit function.  Future approaches will
       include the Gonzalvez/Paxman error function
        
    """
    import time
    start = time.time()

    # define parameters    
    nsamp         = 256
    padFac        = 2
    nterm         = 12
    SNR           = 100
    nWavesDivRMS  = 1
    zVecData      = 0.2*(2-num.random.rand(nterm))
    zVecData[0:3] = num.zeros(3)
    zVecData[4]   = zVecData[4]+nWavesDivRMS
    zVecStart     = num.zeros(nterm)
    zVecStart[4]  = nWavesDivRMS

    self = ISTB.ImageScienceClass()
    self.pol2cartMat(nsamp,padFac)

    # set-up the optimization region
    RR    = self.RR
    TH    = self.TH
    mask  = self.maskPol

    # set up the data
    psf, OTF = myfunc(zVecData)
    data     = psf + (num.max(psf)/SNR)*num.random.rand(len(RR),len(RR))
    start    = myfunc(zVecStart)

    # optimize the function/perform the phase retrieval
    zVecRes, nEvals = LMphaseRetreival(zVecStart,data)

    # evaluate the RMS error
    zVecDat = num.array(zVecData)
    delta   = zVecDat[3:] - zVecRes[3:]
    error   = num.sqrt(num.dot(delta,num.transpose(delta)))\
        /num.sqrt(num.dot(zVecRes,num.transpose(zVecRes)))\
        /num.sqrt(nterm)

    # calculate the ending function    
    recon  = myfunc(zVecRes)

    print 'the RMS relative error of the LM optimization  = ' + '%6.4e'%(error)

    # plot the point spread functions
    #--------------------------------
    # the data
    py.ion()
    py.figure(figsize = (7,3), dpi = 70)
    py.subplot(131)
    py.imshow(data,'bone')
    axis = py.gca()
    axis.axis('off')
    py.title(r'$psf_{data}(\mathbf{x})$')

    # the starting guess
    py.subplot(132)
    py.imshow(start[0],'bone')
    axis = py.gca()
    axis.axis('off')
    py.title(r'$psf_{start}(\mathbf{x})$')

    # the reconstructed psf
    py.subplot(133)
    py.imshow(recon[0],'bone')
    axis = py.gca()
    axis.axis('off')
    py.title(r'$psf_{res}(\mathbf{x})$')

    # plot the optical transfer functions
    #--------------------------------
    # the data
    py.figure(figsize = (7,3), dpi = 70)
    py.subplot(131)
    py.imshow(num.log10(OTF/num.max(OTF)),'bone')
    axis = py.gca()
    axis.axis('off')
    py.clim(-5,0)
    py.title(r'$log_{10}OTF_{data}(\mathbf{u})$')

    # the initial guess
    py.subplot(132)
    py.imshow(num.log10(start[1]/num.max(start[1])),'bone')
    axis = py.gca()
    axis.axis('off')
    py.clim(-5,0)
    py.title(r'$log_{10}OTF_{start}(\mathbf{u})$')

    # the reconstructed OTF
    py.subplot(133)
    py.imshow(num.log10(recon[1]/num.max(recon[1])),'bone')
    axis = py.gca()
    axis.axis('off')
    py.clim(-5,0)
    py.title(r'$log_{10}OTF_{res}(\mathbf{u})$')

    py.figure(figsize = (7,5), dpi = 70)
    py.plot(zVecDat[1:],'*')
    py.plot(zVecStart[1:],'v')
    py.plot(zVecRes[1:],'o')
    py.legend((r'$\mathbf{z}_{data}$',r'$\mathbf{z}_{start}$',r'$\mathbf{z}_{res}$'),0)
    py.xlabel('Zernike mode (RMS weighted)')
    py.ylabel('Zernike coefficients' ' ' r'($\lambda_{RMS}$)')

    elapsed = (time.time()-start)
    print 'time elapsed = ' + '%.2f'%(elapsed)

    raw_input('%Press <return to close>')