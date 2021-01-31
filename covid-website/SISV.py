import numpy as np
import ContactRate as cr

nE   = 6  #maximum number of "Exposed" stages (incubation period)
nI   = 6  #maximum number of "Infectious" stages
nC   = 6  #maximum number of "Critical" stages (fatalities)
nT   = 6  #maximum number ot "Testing" stages  

cS   = 0  #Susceptible people
cE   = 1
cE1  = 2
cE2  = 3
cE3  = 4
cE4  = 5
cE5  = 6
cI   = 7  #total infectious 
cI0  = 8  #Infectious stage 1
cI1  = 9
cI2  = 10 
cI3  = 11
cI4  = 12
cI5  = 13
cR   = 14  #Recovered after infectious (cumulative)
cV   = 15  #Vaccinated
cT   = 16  #Testing: a fraction of infectious people will test positive; results will be available with a delay
cT1  = 17
cT2  = 18
cT3  = 19
cT4  = 20
cT5  = 21
cC   = 22  #Critical: seriously ill after end of infectious period, will die; assume isolated so not contagious
cC1  = 23
cC2  = 24
cC3  = 25
cC4  = 26
cC5  = 27
cF   = 28  #cumulative fatalities
cP   = 29  #cumulative tested positive
cNum = 30



def SISV(x, params):

    population          = params['population']

    exp_stages          = params['exp_stages']              #number of incubation stages [0,nE]
    inf_stages          = params['inf_stages']              #number of infectious stages [0,nI]
    crit_stages         = params['crit_stages']             #number of critical stages [0,nC]
    test_stages         = params['test_stages']             #number of testing stages [0,nT]
    
    death_rate          = params['death_rate']              #Infection Fatality Rate
    gamma_exp           = params['gamma_exp']               #1/test result delay
    gamma               = params['gamma']                   #1/length of infectious period
    gamma_pos           = params['gamma_pos']               #1/test result delay
    gamma_crit          = params['gamma_crit']              #1/(critical care + death reporting time)
    detection_rate      = params['detection_rate']          #percentage of infectious people that test positive
    immun               = params['immun']                   #1/length of natural immunity after recovery
    vacc_start          = params['vacc_start']
    vacc_rate           = params['vacc_rate']               #fraction of susceptible population vaccinated per day
    vacc_immun          = params['vacc_immun']              #1/length of immunity by vaccination
    
    i0                  = params['i0']
    #f0                  = params['f0']
    beta0               = params['beta0']
    #c0                  = params['c0']
    
    y = np.zeros((x.size, cNum))
    
    y[0, cI]  = i0
    y[0, cI0] = i0
    y[0, cC]  = i0 * death_rate * gamma / (beta0+gamma_crit-gamma)
    
    #y[0, cR] = f0 / death_rate
    #y[0, cF] = 0 if "f0" not in params else params['f0']
    #y[0, cP] = 0 if "p0" not in params else params['p0']
    #y[0, cS] = population - y[0,cI] if "s0" not in params else params['s0']
    #y[0, cR] = population - (y[0, cS]+y[0,cI]+y[0, cF])
    y[0, cS] = population - y[0,cI] - y[0,cC] - y[0,cF] - y[0,cR]
    
    #the contact rate beta depends on time
    interv = cr.contact_rate(x, params)   
    

    for i in range(1, x.size): #x[1:]: #

        #calculate the flows from one compartment to another at each time step
        
        flows = [] #list of tuples; each tuple is a flow from one bucket to another (source, dest, amount)

        #infection
        beta = interv[i-1]
        #newlyinfectious = beta * y[i-1, cS] * y[i-1, cI] / population 
        #flows.append((cS, cI0, newlyinfectious))  #move to first infectious stage cI0
        
        newlyexposed = beta * y[i-1, cS] * y[i-1, cI] / population 
        
        if exp_stages==0:
            newlyinfectious = newlyexposed
            flows.append((cS, cI0, newlyinfectious))  #move to first stage of infectious period, no incubation
        else:
            flows.append((cS, cE, newlyexposed))  #move to first stage of incubation period cE
        
            #several stages of incubation change the proba distribution from exponential to gamma
            #first stage from cE ; last stage to cE=cE+nE
            for j in range(exp_stages):  #exp_stages must be in [1,nE]
                exp_i = exp_stages * gamma_exp * y[i-1, cE+j]
                if j==exp_stages-1: #last stage moves to cI0
                    newlyinfectious = exp_i #will be needed for the Testing pathway
                    flows.append((cE+j, cI0, newlyinfectious)) 
                else: #move to next incubation stage
                    flows.append((cE+j, cE+j+1, exp_i)) 
                
        #several infectious stages to change the proba distribution from exponential to gamma shape
        #first stage from cI0 ; last stage to cI0+inf_stages-1
        for j in range(inf_stages-1):  #inf_stages must be in [0,nI]
            inf_i = inf_stages * gamma * y[i-1, cI0+j]
            flows.append((cI0+j, cI0+j+1, inf_i)) 
        
        leave_infectious = inf_stages * gamma * y[i-1, cI0+inf_stages-1]
        
        #a portion of infectious people go into critical care for a while (and will eventually die)
        newlycritical = death_rate * leave_infectious
        flows.append((cI0+inf_stages-1, cC, newlycritical))

        #the rest of infectious people recover
        newlyrecovered = (1-death_rate) * leave_infectious
        flows.append((cI0+inf_stages-1, cR, newlyrecovered))

        #several stages of critical care to change the proba distribution from exponential to gamma
        #first stage from cC ; last stage to cF=cI+nc
        for j in range(crit_stages):  #crit_stages must be in [0,nC]
            crit_i = crit_stages * gamma_crit * y[i-1, cC+j]
            if j==crit_stages-1: #last stage moves to cF
                flows.append((cC+j, cF, crit_i)) 
            else: #move to next critical stage
                flows.append((cC+j, cC+j+1, crit_i)) 
        
        #people in critical care die and their death is reported
        #newlydead = gamma_crit * y[i-1, cC]
        #flows.append((cC, cF, newlydead))
        
        #loss of natural immunity after recovery from exposure
        newlysusceptible_r = immun * y[i-1, cR]
        flows.append((cR, cS, newlysusceptible_r))

        #vaccination
        newlyvaccinated = vacc_rate * y[i-1, cS] if i >= vacc_start else 0
        flows.append((cS, cV, newlyvaccinated))
        
        #loss of immunity after vaccination
        newlysusceptible_v = vacc_immun * y[i-1, cV]
        flows.append((cV, cS, newlysusceptible_v))
        
        #testing of infectious people
        newlytested = detection_rate * newlyinfectious
        flows.append((-1, cT, newlytested))  #-1 to avoid removing this flow from its source

        #several stages of testing to change the proba distribution from exponential to gamma
        #first stage from cT ; last stage to cT+test_stages-1
        for j in range(test_stages):  #test_stages must be in [0,nT]
            test_i = test_stages * gamma_pos * y[i-1, cT+j]
            if j==test_stages-1: #last stage moves to cP
                flows.append((cT+j, cP, test_i)) 
            else: #move to next critical stage
                flows.append((cT+j, cT+j+1, test_i)) 
        
        #publication of test results after a delay
        #newpositives = gamma_pos * y[i-1, cT]
        #flows.append((cT, cP, newpositives))
        
        #apply the flows to the source and destination compartments
        y[i] = y[i-1].copy()
        for (source, dest, flow) in flows:
            y[i,source] -= flow if source>-1 else 0  
            y[i, dest] += flow if dest>-1 else 0

        #tabulate total of all infectious stages and store in cI; will be used to calculate newlyinfectious
        y[i, cI] = 0 #previous row was copied when tabulating flows (see above)
        for j in range(inf_stages):  #inf_stages must be in [0,nI]
            y[i, cI] += y[i, cI0+j]

        
    return y

