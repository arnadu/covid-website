import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from scipy.optimize import minimize
from scipy.special import loggamma
from sklearn.linear_model import LinearRegression
from scipy.optimize import differential_evolution    

#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
#import matplotlib.ticker as mtick
#from matplotlib.ticker import NullFormatter
#from matplotlib.ticker import FuncFormatter
##from matplotlib.dates import WeekdayLocator
#from matplotlib.dates import MonthLocator
#from matplotlib.dates import AutoDateLocator
#from matplotlib.pyplot import cm


#MAXIMUM LIKELIHOOD
#GENERAL THEORY OF FITTING EPIDEMIOLOGIVAL MODEL: https://www.sciencedirect.com/science/article/pii/S2468042719300491
#INTRO TO MLE, POISSON and NEGATIVE BINOMIAL
#https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-9fbff27ea12f
#https://towardsdatascience.com/an-illustrated-guide-to-the-poisson-regression-model-50cccba15958
#https://towardsdatascience.com/negative-binomial-regression-f99031bb25b4
#THEORY OF POISSON+GAMMA MIXTURE and equivalence to Negative Binomial https://gregorygundersen.com/blog/2019/09/16/poisson-gamma-nb/
#Generalized Linear Model and STATSMODELS:
#https://towardsdatascience.com/generalized-linear-models-9cbf848bb8ab
#https://www.statsmodels.org/stable/glm.html   
#HOW TO CREATE CUSTOM MODEL FOR STATSMODELS https://austinrochford.com/posts/2015-03-03-mle-python-statsmodels.html 
from scipy import stats
#stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

#--------------------------------------------------------------
#various exponential growth models that can be fit using fit_model()
def model_expgrowth_continuousplit(x, params, constants):  
    r0 = params[0]
    b0 = params[1]
    b1 = params[2]
    t1 = constants['t1']
    
    y = np.zeros_like(x)
    
    y = r0 * np.exp(b0 * x)
    
    r1 = y[np.where(x==t1)[0]]
    y = np.where(t1<x, r1 * np.exp(b1 * (x - t1)), y)

    return y

#x = np.arange(10,100)
#y = model_expgrowth_continuousplit(x, [1,2/7,0.8/7],{'t1':40})
#plt.plot(x,y)
#plt.yscale('log')
#plt.show()

def model_expgrowth(x, params, constants):  
    r = params[0]
    b = params[1]
    return np.exp(b * x) * r

def model_expgrowthquad(x, params, constants):  
    r = params[0]
    b2 = params[1]
    b1 = params[2]
    return np.exp(b2 * np.power(x,2) + b1 * x) * r


#--------------------------------------------------------------
#various log-likelihood functions that can be used by fit_model()
def loglik_leastsquarerel(y, yhat, alpha):
    #y[] are observed values; yhat are model prediction
    leastsquare = np.nansum( (np.log(yhat)-np.log(y))**2  )  
    return leastsquare

def loglik_leastsquare(y, yhat, alpha):
    #y[] are observed values; yhat are model prediction
    leastsquare = np.nansum( (yhat-y)**2  )  
    return leastsquare

def loglik_poisson(y, yhat, alpha):
    #y[] are observed values; yhat are model prediction
    negLL = -np.nansum( -yhat + y * np.log(yhat)  )  #removed constant terms for minimization
    return negLL

def loglik_negbin(y, yhat, alpha):
    #y[] are observed values; yhat are model prediction
    r = 1/alpha
    #negLL = -np.nansum( loggamma(y+r) -loggamma(y+1) - loggamma(r) + y*np.log(yhat) + r*np.log(r) - (y+r)*np.log(yhat+r) )
    negLL = -np.nansum( y*np.log(yhat)  - (y+r)*np.log(yhat+r) )  #removed constant terms to speed minimization <more sensitive to initial guess ???
    return negLL       

#------------------------------
#estimate alpha for negative binomial
#https://dius.com.au/2017/08/03/using-statsmodels-glms-to-model-beverage-consumption/
def calc_alpha(data,fit):
    var = (np.power(data-fit,2) - data)/fit
    X = fit[:,np.newaxis]
    ols = LinearRegression(fit_intercept=False).fit(np.nan_to_num(X), np.nan_to_num(var))
    alpha = ols.coef_[0]
    return alpha

#--------------------------------------------------------------
#fit the given model using the given maximum likelihood
def fit_model(x, y, model_func, constants, loglik_func, guess, bounds, alpha=1):
   
    #this function is called by the scipy's minimize()
    #it returns the negative log likelihood of the model prediction given the model parameters (optimization target) and the constants
    #it is closure to access calibration data x (to compute the prediction) and y (to compute the likelihood)
    def regression_func(params, constants):
        #make a prediction using the given model params
        yhat = model_func(x, params, constants)
        # compute negative of likelihood 
        negLL = loglik_func(y, yhat, alpha)
        return negLL    
    
    mle = minimize(regression_func, x0=guess, bounds=bounds, args=constants, method="L-BFGS-B")#, method ="Nelder-Mead")
    #display(mle)
    
    res = model_func(x, mle.x, constants)
    return mle.x, mle.fun, res    #mle.x is the array of calibrated model parameters; mle.fun is the loglikelihood (without constant terms); res is the model forecast for input range

#--------------------------------------------------------------
#find the inflection point in the data by comparing maximum likelihood of all possible inflection points after separately fitting a model to the left and right sides of a candidate inflection point
def findsplit(x, y, model_func, constants, loglik_func, guess, bounds, n_min, n_max, window=0, alpha=1, conf=0.05):
    
    #calculate likelihood of fit at every possible split
    for split in range(n_min, n_max):
        
        params_left, mle_left, res_l = fit_model(x[:split-window], y[:split-window], model_func, constants, loglik_func, guess, bounds, alpha)
        params_right, mle_right, res_r = fit_model(x[split+window:], y[split+window:], model_func, constants, loglik_func, guess, bounds, alpha)

        if (split==n_min) or (mle_left+mle_right <= min_mle):
            min_split = split
            min_mle = mle_left+mle_right
            res_left = res_l
            res_right = res_r
            p_left = params_left
            p_right = params_right
    
    #calculate liklihood of fit over entire range, without split
    params, mle, res = fit_model(x, y, model_func, constants, loglik_func, guess, bounds, alpha)

    #test for significant improvement if we split the range in two
    lr = - 2 * (min_mle - mle)
    p = stats.chi2.sf(lr, 2) #2 more degrees of freedom in split regression than in full regression
    #print('split ll:{:,.0f} full ll:{:,.0f} lr:{:,.0f} p:{}'.format(min_mle, mle, lr, p))
    
    if p>conf:
        min_split = 0
        min_mle = mle
        res_left = res
        res_right = []
        p_left = params
        p_right = []
    
    buff = np.empty(2*window)
    buff[:]=np.nan
    res = np.append(res_left, buff)
    res = np.append(res, res_right)
    
    r = objdict({})
    r.Split = min_split
    r.Stages = []
    r.Stages.append(p_left)
    r.Stages.append(p_right)
    r.Predict = res

    return r #min_split, p_left, p_right, res   #min_split will be zero if the optimal solution is no split; params are in the left variables


#--------------------------------------------------------------
def findsplit_continuous(x, y, loglik_func, guess, bounds, n_min, n_max, alpha=1, conf=0.05):
    
    #calculate likelihood of fit at every possible split
    for split in range(n_min, n_max):
        
        params, mle, res = fit_model(x, y, model_expgrowth_continuousplit, {'t1':split}, loglik_func, guess, bounds, alpha)

        if (split==n_min) or (mle <= min_mle):
            min_split = split
            min_mle = mle
            min_res = res
            min_params = params
 
    r = objdict({})
    r.split = min_split
    r.mle = min_mle
    r.predict = min_res
    r.params = min_params  #r0 b0 b1

    return r

#==================================================================================

def piecewiseexp(x, params, constants):
    
    r0 = params[0]
    ai = params[1]
    b0 = x[0]

    breakpoints = constants['breakpoints']
    
    res = r0 * np.exp(ai*(x-b0))

    if 'debug' in constants:
        print(0, ai, ai)
    
    for i, bi in enumerate(breakpoints):
        ai = params[i+2] - params[i+1]
        s = (x>=bi)
        res[s] *= np.exp(ai * (x[s]-bi))
        
        if 'debug' in constants:
            print(i+1, params[i+2], ai)
     
    return res

#transform auxiliary values (which are bounded [0,1] into the breakpoint dates)
def aux_to_breakpoints(aux, x0, xn, minwindow):
    breakpoints = []
    bi1 = x0
    n=len(aux)
    for i,a in enumerate(aux):
        bi = a * (xn-(n-i)*minwindow - (bi1+minwindow)) + bi1+minwindow
        breakpoints.append(bi)
        bi1=bi   
    return breakpoints

def piecewiseexp_diffevol_inner(aux, constants):
    
    x = constants['x']
    y = constants['y']
    bounds = constants['bounds']
    guess = constants['guess']
    alpha = constants['alpha']
    
    breakpoints = aux_to_breakpoints(aux, x[0], x[-1], constants['minwindow'])    

    params, likelihood, fit = fit_model(x, y, model_func=piecewiseexp, constants={'breakpoints':breakpoints}, 
                                        loglik_func=loglik_negbin, 
                                        guess=guess, bounds=bounds, alpha=alpha)

    return likelihood
 
    
#https://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.differential_evolution.html
def piecewiseexp_diffevol(x, y, breaks=1, minwindow=14):
    
    #prepare the bounds and guess vectors, and the other constants that will be used in the inner optimizer
    guess = [1, 1/7]   #initial count and initial growth rate
    bounds = [(1e-1,1e3),(-10/7, 10/7)]  #bounds for initial count and initial growth rate
    alpha = 1
    
    for i in range(breaks):
        guess.append(1/7)  #growth rate after breakpoint
        bounds.append((-10/7,10/7))
    
    constants = {'minwindow': minwindow, 'x':x, 'y':y, 'guess':guess, 'bounds':bounds, 'alpha':alpha}
    
    #bounds for the auxiliary variables that will be used to set the breakpoints
    #these auxiliary variables are optimized by differential_evolution()
    auxBounds = [(0,1)] * breaks  
    
    #find the optimal breakpoints
    res = differential_evolution(func=piecewiseexp_diffevol_inner, bounds=auxBounds, args=[constants])
    #print(res)
    
    #do the regression for the optimum breakpoints
    aux = res.x
    breakpoints = aux_to_breakpoints(aux, x[0], x[-1], minwindow) 
    
    params, likelihood, fit = fit_model(x, y, model_func=piecewiseexp, constants={'breakpoints':breakpoints}, 
                                        loglik_func=loglik_negbin, 
                                        guess=guess, bounds=bounds, alpha=alpha)


    return params, breakpoints, likelihood, fit




