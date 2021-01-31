import numpy as np
import ContactRate as cr


cS   = 0  #Susceptible people
cI   = 1  #Infectious 
cR   = 2  #Recovered after infectious (cumulative)
cV   = 3  #Vaccinated
cT   = 4  #Testing: a fraction of infectious people will test positive; results will be available with a delay
cC   = 5  #Critical: seriously ill after initial infectious period, will die, assume isolated
cF   = 6  #cumulative fatalities
cP   = 7  #cumulative tested positive
cNum = 8



def SISV(x, params):

    population          = params['population']
    death_rate          = params['death_rate']              #Infection Fatality Rate
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
    
    y[0, cI] = i0
    y[0, cC] = i0 * death_rate * gamma / (beta0+gamma_crit-gamma)
    #y[0, cR] = f0 / death_rate
    #y[0, cF] = 0 if "f0" not in params else params['f0']
    #y[0, cP] = 0 if "p0" not in params else params['p0']
    #y[0, cS] = population - y[0,cI] if "s0" not in params else params['s0']
    #y[0, cR] = population - (y[0, cS]+y[0,cI]+y[0, cF])
    y[0, cS] = population - y[0,cI] - y[0,cC] - y[0,cF] - y[0,cR]
    
    #the contact rate beta depends on time
    interv = cr.contact_rate(x, params)   
    

    for i in range(1, x.size):

        #calculate the flows from one compartment to another at each time step
        
        flows = [] #list of tuples; each tuple is a flow from one bucket to another (source, dest, amount)

        #infection
        beta = interv[i]
        newlyinfectious = beta * y[i-1, cS] * y[i-1, cI] / population 
        flows.append((cS, cI, newlyinfectious))

        #a portion of infectious people go into critical care for a while (and will eventually die)
        newlycritical = death_rate * gamma * y[i-1, cI]
        flows.append((cI, cC, newlycritical))

        #the rest of infectious people recover
        newlyrecovered = (1-death_rate) * gamma * y[i-1, cI]
        flows.append((cI, cR, newlyrecovered))

        #people in critical care die and their death is reported
        newlydead = gamma_crit * y[i-1, cC]
        flows.append((cC, cF, newlydead))
        
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
        
        #publication of test results after a delay
        newpositives = gamma_pos * y[i-1, cT]
        flows.append((cT, cP, newpositives))
        
        #apply the flows to the source and destination compartments
        y[i] = y[i-1].copy()
        for (source, dest, flow) in flows:
            y[i,source] -= flow if source>-1 else 0  
            y[i, dest] += flow if dest>-1 else 0
            
    return y
