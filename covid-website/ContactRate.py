import numpy as np

#============================================================
def piecewiselin(x, params):  
    #r  = params[0]
    b0 = params['beta0']
    n  = params['segments'] #number of segments
    init_beta = params['init_beta']

    bi1 = b0
    ti1 = x[0]

    if n==0:
        b = b0 * np.ones(len(x))
    else:
        b = np.zeros(len(x))
        i=1
        while i<=n:

            bi = params['beta{}'.format(i)]
            ti = params['t{}'.format(i)]
            
            if i==1 and init_beta == 'const':
                b = np.where( (ti1<=x)&(x<=ti), bi1, b)
            else:
                b = np.where( (ti1<=x)&(x<=ti), (x-ti1)*(bi-bi1)/(ti-ti1)+bi1, b)
            i=i+1
            bi1=bi
            ti1=ti

        b = np.where(x>=ti,bi,b)
    
    return b

#x = np.arange(100)
#b0 = piecewiselin(x, {'init_beta':'', 'segments':0, 'beta0':3/7})
#b1 = piecewiselin(x, {'init_beta':'', 'segments':1, 't1':20, 'beta0':3.1/7, 'beta1':2.1/7})
#b2 = piecewiselin(x, {'init_beta':'const', 'segments':2, 't1':20, 't2':80, 'beta0':3.2/7, 'beta1':2.2/7, 'beta2':1/7})
#b2l = piecewiselin(x, {'init_beta':'', 'segments':2, 't1':20, 't2':80, 'beta0':3.2/7, 'beta1':2.2/7, 'beta2':1/7})
#plt.plot(x,b0,label='b0')
#plt.plot(x,b0,label='b0')
#plt.plot(x,b1,label='b1')
#plt.plot(x,b2,label='b2')
#plt.plot(x,b2l,label='b2l')
#plt.grid()
#plt.legend()
#plt.show()

#============================================================
def piecewiseconst(x, params):  
    b0 = params['beta0']
    n  = params['segments'] #number of segments


    b = b0 * np.ones(len(x))
    i=1
    while i<=n:
        
        bi = params['beta{}'.format(i)]
        ti = params['t{}'.format(i)]
        b = np.where( x>=ti, bi, b)
        i=i+1
        
    return b




#============================================================
def contact_rate(x,params):
#time varying contact rate, using one of the available profiles
#calculate the contact rate over time, according to the profile given by the intervention function and the calibration params (betai, ti)

    interv_functions = {
                        'piecewise linear'   : piecewiselin,
                        'piecewise constant' : piecewiseconst,
                       }

    intervention    = params['interv']
    
    if intervention in interv_functions:
        interv_func = interv_functions[intervention]
        interv = interv_func(x, params)
    else:
        interv = params['beta0'] * np.ones_like(x)

    #print('---------')    
    #print('ContactRate:contact_rate')
    #print(x)
    #print(params)
    #print(interv)
    
    return interv



#x = np.arange(100)
#b0 = piecewiseconst(x, {'segments':0, 'beta0':3/7})
#b1 = piecewiseconst(x, {'segments':1, 't1':20, 'beta0':3.1/7, 'beta1':2.1/7})
#b2 = piecewiseconst(x, {'segments':2, 't1':20, 't2':80, 'beta0':3.2/7, 'beta1':2.2/7, 'beta2':1.3/7})
#plt.plot(x,b0,label='const 0')
#plt.plot(x,b1,label='const 1')
#plt.plot(x,b2,label='const 2')
#plt.grid()
#plt.legend()
#plt.show()


#x = np.arange(100)
#b0 = contact_rate(x, {'interv':'reopening', 'beta0':5/7, 'beta2':0.7/7, 'beta4':2/7, 't1':20, 't2':27, 't3':50, 't4':100})
#plt.plot(x,b0,label='Reopening')
#plt.grid()
#plt.legend()
#plt.show()
