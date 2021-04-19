#---------------------------------------------------------------------------------
#Calibrate SISV on historical data, solving for the contact rate shape over time a given number of time segments 
#---------------------------------------------------------------------------------

from lmfit import Minimizer, Parameters, report_fit
import SISV as s
from data import Data
from ContactRate import contact_rate


#---------------------------------------------------------------------------------
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

#---------------------------------------------------------------------------------
def breakpoints_to_aux(breakpoints, x0, xn, minwindow):
    aux = []
    bi1 = x0
    n=len(breakpoints)
    for i,b in enumerate(breakpoints):
        temp = xn - (n-i)*minwindow - (bi1+minwindow)
        a = 1.0 if temp==0 else (b - bi1 - minwindow) / temp
        aux.append(a)
        bi1=b
    return aux  
    
#aux_to_breakpoints([0.5, 0.5], 0, 100, 10)


#---------------------------------------------------------------------------------


def override(key, overrides, default):
    return default if key not in overrides else overrides[key]

#---------------------------------------------------------------------------------
def SISV_lmfit(d, overrides={}, solver='leastsq'):

    #--------------------------------------
    def merge_params(params, constants, solve_ti=False):
    
        p = params.valuesdict()  #params should be a LMFIT Parameters instance; constants should be a dict
        for i, (k,v) in enumerate(constants.items()):
            if k not in p:  #do not override values that may already be in the Parameters array
                p[k]=v
    
        if solve_ti:  #we are solving for time breakpoints
            #calculate "ti" variables from "auxi" variables
            n = p['segments']
            
            aux = []
            for i in range(1, n+1):
                aux.append(params['aux{}'.format(i-1)])
                
            breakpoints = aux_to_breakpoints(aux, d.x[0], d.x[-1], minwindow)
            
            for i in range(1,n+1):
                #p['beta{}'.format(i)] = params[len(param_list)+i-1]
                p['t{}'.format(i)] = breakpoints[i-1]
                
        return p


    #--------------------------------------
    def lmfit_inner(params, x, constants, column, data=None):
        p = merge_params(params, constants, solve_ti=True if column==s.cF else False)   #solve for time breakpoints Ti through Auxi variables when using fatalities data
        yhat = s.SISV(x, p)
        if data is None:
            return yhat
        else:
            return yhat[:,column] - data



    #--------------------------------------
    #first stage: calibrate initial infectious population and contact rate over time on fatalities data
    #--------------------------------------
    
        
    gamma = override('gamma', overrides, 1/3)
    segments = override('segments', overrides, 7)
    minwindow = override('minwindow', overrides, 7)

    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('exp_stages',    1, False),
                     ('inf_stages',    1, False),
                     ('crit_stages',   1, False),
                     ('test_stages',   1, False),
                     
                     ('gamma_exp',      gamma, False),
                     ('gamma',          gamma, False),
                     ('gamma_pos',      1/14, False),
                     ('gamma_crit',     1/14, False),
                     
                     ('death_rate',     0.5e-2, False),
                     ('detection_rate', 5e-2, False),

                     ('population',     d.population, False),
                     ('f0',             d.fatalities[0], False),
                     ('p0',             d.positives[0], False),

                     ('immun',          0, False),
                     ('vacc_start',     365, False),
                     ('vacc_rate',      0, False),
                     ('vacc_immun',     1/180, False),
        
                     ('segments',       segments         , False),

                     ('i0',             10        , True, 1, 1e4),
                     ('c0',             10        , False, 10, 1e6),
                     ('beta0',          2*gamma       , True, 0.05*gamma, 10*gamma),                 
                   )

    for i in range(1, segments+1):  
        params.add('aux{}'.format(i-1),value=0.1, vary=True, min=0, max=1)
        params.add('beta{}'.format(i), value= 1*gamma, vary=True,  min=0.05*gamma, max=10*gamma)

    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 
        'interv':'piecewise linear',
        'init_beta':'const',
    }

    for idx, (k,v) in enumerate(overrides.items()):
        if k in params:
            params[k].set(value=v)
        else:
            constants[k]=v

    #params.pretty_print()
    #print(type(d.x))
    
    fitter = Minimizer(lmfit_inner, params, fcn_args=(d.x, constants, s.cF, d.fatalities))
    result = fitter.minimize(method=solver)

    p = merge_params(result.params, constants, solve_ti=True)  #merge the calibrated variables into the dictionary of params
    #print(p)    
    #-------------------------
    #second stage: calibrate detection rate on positive test results data
    #------------------------

    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('detection_rate', 15e-2, True, 0, 1),
                   )

    fitter = Minimizer(lmfit_inner, params, fcn_args=(d.x, p, s.cP, d.positives))
    result = fitter.minimize(method=solver)

    p = merge_params(result.params, p, solve_ti=False) #merge the calibrated variables into the dictionary of params




    
    #yy = s.SISV(d.x, p)
    return p
    
