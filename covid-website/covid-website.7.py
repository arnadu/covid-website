import math

#https://github.com/bokeh/bokeh/blob/branch-2.3/examples/app/sliders.py
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CheckboxGroup, Slider, Button, TextInput, CustomJS, Div, MultiChoice, MultiSelect, Select, Panel, Tabs
from bokeh.plotting import figure
from bokeh.palettes import Category20 as palette
from bokeh.models.widgets import Tabs, Panel
from bokeh.models import NumeralTickFormatter

from bokeh.models import DataTable, TableColumn

import numpy as np
import SISV as s
from data import Data


#--------------------------------------------------
#--------------------------------------------------
def validate_int(s):
    try:
        return int(s)
    except:
        return None

#--------------------------------------------------
def validate_float(s):
    try:
        f = float(s)
        return f if math.isfinite(f) else None
    except:
        return None

#add the error message to the set 'e' if the input cannot be validated and return None
def validate_input(s, e, t='int', min_val='None', max_val='None', error=''):
    
    if t=='int':
        v = validate_int(s) 
    elif t=='float':
        v = validate_float(s)
    elif t=='choice':
        v = s if s in min_val else None
    else:
        v = None
        
    
    if v==None:
        e.add(error)
        return None
    
    if t=='float' or t=='int':
        if min_val is not None and v<min_val:
            e.add(error)
            return None
        
        if max_val is not None and v>max_val:
            e.add(error)
            return None

    return v
    
#--------------------------------------------------
class ParametersForm:

    wMessage = Div(text="") #a placeholder to display the list of error messages on the screen

    wPopulation = TextInput(value='1.0', title='Population (millions)')
    wSimulHorizon = TextInput(value='730', title='Simulation Horizon (days)')
    
    #-----------------
    #use a bokeh widget DataTable for the entry of parameters; 
    #first column is the names, second column the values for scenario1 and 3rd column the values for scenario2

    #meta data list of parameters for the DataTable input
    #tuple       = ('name',    row, default1, default2)
    cI0          = ('Initially Infectious',        0, 1, 1)
    cGamma       = ('Infectious Period (days)',    1, 5, 5)
    cGammaCrit   = ('Time to death (days)',        2, 10, 10)
    cDeathRate   = ('Infection Fatality Rate (%)', 3, 1, 1)
    cGammaPos    = ('Time to test results (days)', 4, 5, 5)
    cDetectionRate = ('Detection Rate (%)',        5, 10, 10)
    cSigma       = ('Immunity (days)',             6, 0, 365)
    cVaccStart   = ('Vaccination Start Day',       7, 0, 365)
    cVaccRate    = ('Vaccination Rate %/year',     8, 0, 30)
    cVaccSigma   = ('Vaccination Immunity (days)', 9, 0, 180)
    
    cInterv      = ('Contact Rate Shape',          10, 'piecewise constant', 'piecewise constant')
    cSegments    = ('Number of Segments',          11, 1, 4)
    cR0          = ('R0',                          12, 3, 3)
    
    #THIS LIST NEEDS TO BE IN THE SAME ORDER AS INDICATED IN THE TUPLES ABOVE!!!
    param_list = [ cI0, cGamma, cGammaCrit, cDeathRate, cGammaPos, cDetectionRate, cSigma, cVaccStart, cVaccRate, cVaccSigma, cInterv, cSegments, cR0 ]

    params = [name for (name, row, default1, default2) in param_list]
    simul1 = [default1 for (name, row, default1, default2) in param_list]
    simul2 = [default2 for (name, row, default1, default2) in param_list]
    
    params += [30, 120, 210]
    simul1 += [4,4,4]
    simul2 += [0.9, 1.1, 1.7]

    source = ColumnDataSource( dict(
        params= params,      
        simul1= simul1,  
        simul2= simul2,  
    ))
    
    
    columns = [
        TableColumn(field='params', title='Parameter'),
        TableColumn(field='simul1', title='Simul #1'),
        TableColumn(field='simul2', title='Simul #2')
    ]
    
    wTable= DataTable(source=source, columns=columns, editable=True, sortable=False)
    
    wAddRow = Button(label="+Row")

    layout = column(wMessage, wPopulation, wSimulHorizon, wTable, wAddRow)
    
    #-----------------
    
    #get the params for a given scehario
    def scenario_params(self, i):
        
        e=set()
        
        n              = validate_input(self.wSimulHorizon.value, e, 'int', 50, None, 'Simulation Horizon should be greater than 50 days')
        population     = validate_input(self.wPopulation.value, e, 'float', 1, None, '"Population (millions)"should be greater than 1 (million)')

        data = self.source.data['simul1'] if i==1 else self.source.data['simul2']

        i0               = validate_input(data[self.cI0[1]], e, 'int', 1, population*1e6, '"Initially Infectious" should be greater than one and less than Population')
        gamma            = validate_input(data[self.cGamma[1]], e, 'int', 1, None, '"Infectious Period (days)" should be 1 or greater')
        gamma_crit       = validate_input(data[self.cGammaCrit[1]], e, 'int', 1, None, '"Time to death (days)" should be 1 or greater')
        gamma_pos        = validate_input(data[self.cGammaPos[1]], e, 'int', 1, None, '"Time to test results (days)" should be 1 or greater')
        detection_rate   = validate_input(data[self.cDetectionRate[1]], e, 'float', 0, 100, '"Detection Rate (5)" is the percentage of infectious people tested positive')
        death_rate       = validate_input(data[self.cDeathRate[1]], e, 'float', 0, 100, '"Infection Fatality Rate (5)" is the percentage of infected people who will die')
        sigma            = validate_input(data[self.cSigma[1]], e, 'int', 0, None, '"Immunity" is the number of days a person remains immune after recovery from infection, or zero for permanent immunity')
        vacc_start       = validate_input(data[self.cVaccStart[1]], e, 'int', 0, None, '"Vaccination Start Day" is the day when the vaccination program starts')
        vacc_rate        = validate_input(data[self.cVaccRate[1]], e, 'float', 0, None, '"Vaccination Rate %/year" is the percentage of suscetible people vaccinated per year')
        vacc_immun       = validate_input(data[self.cVaccSigma[1]], e, 'int', 0, None, '"Vaccination Immunity (days)" is the number of days of immunity given by the vaccine, or zero for permanent immunity')

        interv           = validate_input(data[self.cInterv[1]], e, 'choice', ['piecewise constant', 'piecewise linear'], None, '"Contact Rate Shape" should be "piecewise constant" or "piecewise linear"')
        segments         = validate_input(data[self.cSegments[1]], e, 'int', 1, 15, '"Number of Segments" shoudl be between 1 and 15')
        R0               = validate_input(data[self.cR0[1]], e, 'float', 0.1, 10, '"R0" shoudl be between 0.1 and 10')


        params = {}
        if len(e) == 0:  #all inputs have been validated, with no errors ; this also confirms that none of the variables is None
            params = {
                "population"     : population * 1e6,
                "i0"             : i0,
                "gamma"          : 1/gamma,   #1/length of infectious period in days
                "gamma_crit"     : 1/gamma_crit,   #1/length of time to death after end of infectious period in days
                "gamma_pos"      : 1/gamma_pos,   #1/length of time to test results after being infected in days
                "death_rate"     : death_rate/100,
                "detection_rate" : detection_rate/100,
                "immun"          : 0 if sigma==0 else 1/sigma,
                "vacc_start"     : vacc_start,
                "vacc_rate"      : vacc_rate/365/100,
                "vacc_immun"     : 0 if vacc_immun==0 else 1/vacc_immun,

                "interv"         : interv,
                "segments"       : segments-1,
                "beta0"          : R0 / gamma,   #(here gamma_infec is a number of days, not a rate)
            }
            
            for i in range(1, segments):
                row = self.cR0[1]+i
                
                t = [row]
                ti = validate_input(self.source.data['params'][row], e, 'int', 1, None, 'cells below R0 should be integer')
                Ri = validate_input(data[row], e, 'float', 0.1, 10, '"Ri" should be between 0.1 and 10')
                params['t{}'.format(i)] = ti
                params['beta{}'.format(i)] = Ri / gamma
                
            print(params)
            
        return e, n, params

    def log(self, e):  #display the list error messages collected during input validation
        self.wMessage.text=""
        for m in e:
            self.wMessage.text += m + "<br>"


    #wTable.source.on_change("data", table_onchange)
    #add a row to the table of parameters, to allow for additional segments
    def table_addrow():
        data = ParametersForm.source.data.copy()
        data['params'].append('')
        data['simul1'].append('')
        data['simul2'].append('')
        ParametersForm.source.data = data
        
    wAddRow.on_click(table_addrow)
        
#--------------------------------------------------
class DataForm:

    REGIONS  = ["Europe", "US"]
    STATES   = ["","California", "New York"]
    COUNTIES = [""]
    
    global d
    
    wLoad = Button(label="Load")
    
    wMessage = Div(text="") #a placeholder to display the list of error messages on the screen

    wRegion = Select(title="Region", value="US", options=REGIONS)
    wState  = Select(title="State", value="New York", options=STATES)
    wCounty = Select(title="County", value="", options=COUNTIES)
    
    
    def log(self, e):  #display the list error messages collected during input validation
        self.wMessage.text=""
        for m in e:
            self.wMessage.text += m + "<br>"

    layout = column(wLoad, wMessage, wRegion, wState, wCounty)

            
#========================================================
COLORS = palette[13]
print(COLORS)

LINES = {
#   "display name"  : (scenario, column, daily? , color, dash)
    "1-Susceptible"       : (1, s.cS, False, COLORS[0], 'solid'), 
    "1-Infectious"        : (1, s.cI, False, COLORS[1], 'solid'),
    "1-Recovered"         : (1, s.cR, False, COLORS[2], 'solid'), 
    "1-Vaccinated"        : (1, s.cV, False, COLORS[3], 'solid'), 
    "1-Critical"          : (1, s.cC, False, COLORS[4], 'solid'), 
    "1-Cumul Fatalities"  : (1, s.cF, False, COLORS[5], 'solid'), 
    "1-Daily Fatalities"  : (1, s.cF, True,  COLORS[6], 'solid'), 
    "1-Positives"         : (1, s.cP, False, COLORS[7], 'solid'), 
    "1-Daily Positives"   : (1, s.cP, True,  COLORS[8], 'solid'), 
    
    
    "2-Susceptible"       : (2, s.cS, False, COLORS[0], 'dotted'), 
    "2-Infectious"        : (2, s.cI, False, COLORS[1], 'dotted'),
    "2-Recovered"         : (2, s.cR, False, COLORS[2], 'dotted'), 
    "2-Vaccinated"        : (2, s.cV, False, COLORS[3], 'dotted'), 
    "2-Critical"          : (2, s.cC, False, COLORS[4], 'dotted'), 
    "2-Cumul Fatalities"  : (2, s.cF, False, COLORS[5], 'dotted'), 
    "2-Daily Fatalities"  : (2, s.cF, True,  COLORS[6], 'dotted'), 
    "2-Positives"         : (2, s.cP, False, COLORS[7], 'dotted'), 
    "2-Daily Positives"   : (2, s.cP, True,  COLORS[8], 'dotted'), 
    
    "Cumul Positives"     : (3, 'positives', False, COLORS[9], 'solid'),
    "Daily Positives"     : (3, 'dpositives', True, COLORS[10], 'solid'),
    "Cumul Fatalities"    : (3, 'fatalities', False, COLORS[11], 'solid'),
    "Daily Fatalities"    : (3, 'dfatalities', True, COLORS[12], 'solid'),

    
}

def on_change_plot(attr, old, new): #a wrapper, required to meet the prototype expected for handlers of Bokeh's on_change events
    return update_plot()
    
def update_plot(new=None):  #the new=None is the prototype expected for handlers of Bokeh's on_click events

    #these globals are refreshed by the simulate() function
    global e
    global n
    global params1
    global params2
    global x
    global y1
    global y2
    global d
    
    scale = wPlot.scale()  #"log" or "linear"
    #print(scale)

    
    
    #create a new figure
    #new_p = figure(plot_height=400, plot_width=800, title="SIRF simulation", y_axis_type=scale, name='plot')
    new_p = figure(title="SIRF simulation", y_axis_type=scale, name='plot')
    new_p.sizing_mode = 'scale_width'

    #get out if there was an error
    wForm.log(e)
    if len(e) != 0:
        print('errors', e)
        return

    #decide whether to plot
    if wPlot.scale_rel(): #True if displaying numbers per 100k, False to display absolute count
        if d is not None:
            population = d.population
        else:
            population = params1['population']  * 1e6
        scale_rel = 100e3/population
    else:
        scale_rel = 1
    print(scale_rel)

    if d is not None:
        series = {
            'xfatalities' : d.x[d.minD:],
            'xpositives'  : d.x[d.minP:],
            'xdfatalities': d.x[d.minD+1:],
            'xdpositives' : d.x[d.minP+1:],
            'positives'   : d.positives,
            'dpositives'  : d.dpositives,
            'fatalities'  : d.fatalities,
            'dfatalities' : d.dfatalities,
        }
    else:
        series = None

    #print(series)
    
    #source = ColumnDataSource(dict(x=x, I1=y1[:,s.cI], S1=y1[:,s.cS], R1=y1[:,s.cR], V1=y1[:,s.cV], I2=y2[:,s.cI], S2=y2[:,s.cS], R2=y2[:,s.cR], V2=y2[:,s.cV] ))
    for idx, (line,(scenario, column, daily, color, dash)) in enumerate(LINES.items()):
        
        if wPlot.show_line(line):

            if scenario==3: #plot from historical data
            
                if series is not None:
                    y = series[column] * scale_rel
                    new_p.line(series['x'+column], y, line_width=3, line_alpha=1, legend_label=line, line_color=color, line_dash=dash)
                    
            else:  #plot from simulation results
                
                y=y1[:,column] if scenario==1 else y2[:,column]
                y *= scale_rel
                
                if daily:
                    new_p.line(x[1:], np.diff(y), line_width=3, line_alpha=1, legend_label=line, line_color=color, line_dash=dash)
                else:
                    new_p.line(x, y, line_width=3, line_alpha=1, legend_label=line, line_color=color, line_dash=dash)
            
    #if wPlot.show_line('1-Infectious'):
    #    line_I1 = new_p.line('x', 'I1', source=source, line_width=3, line_alpha=0.6, legend_label='1-Infectious', line_color=colors[0], line_dash='solid')

    new_p.yaxis.formatter = NumeralTickFormatter(format='0,0.0')
    
    new_p.xaxis.axis_label = 'days'
    new_p.yaxis.axis_label = 'count'
    
    new_p.legend.location = "top_right"
    new_p.legend.click_policy="hide"

    #push the new figure to the page

    wPlot.layout.children[2] = new_p
    
    #layouts = curdoc().get_model_by_name('col2')
    #if layouts is not None:
    #    layouts.children[1]=p
    #    #layouts.children[1].children[1]=p

def load_data():
    
    e = set()
    global d

    source = 'Johns Hopkins'    
    cutoff_positive = 1
    cutoff_death = 1
    truncate = 0

    region = validate_input(wDataForm.wRegion.value, e, 'choice', wDataForm.REGIONS, None, "Not a valide Region")
    state  = validate_input(wDataForm.wState.value, e, 'choice', wDataForm.STATES, None, "Not a valide State")
    county = validate_input(wDataForm.wCounty.value, e, 'choice', wDataForm.COUNTIES, None, "Not a valide County")
    
    if (len(e)==0):
        d = Data(source=source, region=region, state=state, county=county, cutoff_positive=cutoff_positive, cutoff_death=cutoff_death, truncate=truncate) 
        wForm.wPopulation.value = str(d.population / 1e6)
    else:
        d = None
        wDataForm.log(e)
    
    #print(d)
    update_plot()
    
class PlotForm():
    
    #--------------
    wLineSelection = MultiChoice(value=["1-Infectious", "2-Infectious"], options=list(LINES.keys()))
    wLineSelection.on_change("value", on_change_plot)
    def show_line(self, s):
        return True if s in self.wLineSelection.value else False

    #--------------
    wScale = CheckboxGroup(labels=['Log', 'per 100k'], active=[])
    wScale.on_click(update_plot)
    
    def scale(self):
        return "log" if 0 in self.wScale.active else "linear"
    
    def scale_rel(self): #disply results per 100k or absolute numbers
        return True if 1 in self.wScale.active else False

    wFigure = Div(text="")

    layout = column(wLineSelection, wScale, wFigure)
    
    
# create a callback that will perform the simulation and update the chart
def simulate():

    global wForm
    global e
    global n
    global x
    global y1
    global y2
    global params1
    global params2

    e1, n, params1 = wForm.scenario_params(1)
    e2, n, params2 = wForm.scenario_params(2)
    e = e1.union(e2)

    print('=============')
    print(e)
    print('-------------')
    print(params1)
    print('-------------')
    print(params2)
    
    
    if len(e) == 0:
        x = np.arange(n)
        y1 = s.SISV(x, params1)
        y2 = s.SISV(x, params2)
    
    update_plot()
    

# add a button widget and configure with the call back
button = Button(label="Simulate")
button.on_click(simulate)

d=None

#form to get all the simulation parameters for the two scenarios


wForm = ParametersForm()
wDataForm = DataForm()
wPlot = PlotForm()

wDataForm.wLoad.on_click(load_data)

#do a first calculation on default parameters and display the results
#simulate()

# put the button, controls and parameters widgets and plot in a layout and add to the document
#c = column([button, logscale_checkbox, p, wForm.layout], name='layout')
#c.sizing_mode = 'scale_width'

tab1 = Panel(child=column([button, wForm.layout], name='col1'), title='Simulate')
tab2 = Panel(child=wDataForm.layout, title='Data')
col1  = Tabs(tabs=[tab1, tab2])

col2 = column([wPlot.layout], name='col2')

c = row([col1,col2], name='layout')
curdoc().add_root(c)
curdoc().title = "COVID Simulation"

simulate()

