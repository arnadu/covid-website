import numpy as np
import ContactRate as cr


cS   = 0  #Susceptible people
cI   = 1  #Infectious 
cR   = 2  #Recovered after infectious (cumulative)
cV   = 3  #Vaccinated
cNum = 4  

def SISV(x, params):

    population          = params['population']
    #beta                = params['beta']
    gamma               = params['gamma']                      #1/length of infectious period
    immun               = params['immun']                   #1/length of natural immunity after recovery
    vacc_start          = params['vacc_start']
    vacc_rate           = params['vacc_rate']           #fraction of susceptible population vaccinated per day
    vacc_immun          = params['vacc_immun']       #1/length of immunity by vaccination

    y = np.zeros((x.size, cNum))
    
    y[0, cI] = params["i0"]
    y[0, cS] = population - y[0,cI]

    #the contact rate beta depends on time
    interv = cr.contact_rate(x, params)   
    

    for i in range(1, x.size):

        #calculate the flows from one compartment to another at each time step
        
        flows = [] #list of tuples; each tuple is a flow from one bucket to another (source, dest, amount)

        #infection
        beta = interv[i]
        newlyinfectious = beta * y[i-1, cS] * y[i-1, cI] / population 
        flows.append((cS, cI, newlyinfectious))

        #recovery from first exposure
        newlyrecovered = gamma * y[i-1, cI]
        flows.append((cI, cR, newlyrecovered))
        
        #loss of natural immunity after recovery from exposure
        newlysusceptible_r = immun * y[i-1, cR]
        flows.append((cR, cS, newlysusceptible_r))

        #vaccination
        newlyvaccinated = vacc_rate * y[i-1, cS] if i >= vacc_start else 0
        flows.append((cS, cV, newlyvaccinated))
        
        #loss of immunity after vaccination
        newlysusceptible_v = vacc_immun * y[i-1, cV]
        flows.append((cV, cS, newlysusceptible_v))
        
        #apply the flows to the source and destination compartments
        y[i] = y[i-1].copy()
        for (source, dest, flow) in flows:
            y[i,source] -= flow if source>-1 else 0  
            y[i, dest] += flow if source>-1 else 0
            
    return y
