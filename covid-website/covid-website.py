import math

#https://github.com/bokeh/bokeh/blob/branch-2.3/examples/app/sliders.py
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CheckboxGroup, Slider, Button, TextInput, CustomJS, Div, MultiChoice, MultiSelect
from bokeh.plotting import figure
from bokeh.palettes import Category20 as palette
from bokeh.models.widgets import Tabs, Panel
from bokeh.models import NumeralTickFormatter

from bokeh.models import DataTable, TableColumn

import numpy as np
import SISV as s



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
    cGamma       = ('Infectious Period (days)',    0, 6, 6)
    cSigma       = ('Immunity (days)',             1, 0, 365)
    cVaccStart   = ('Vaccination Start Day',       2, 0, 365)
    cVaccRate    = ('Vaccination Rate %/year',     3, 0, 30)
    cVaccSigma   = ('Vaccination Immunity (days)', 4, 0, 180)
    
    cInterv      = ('Contact Rate Shape',          5, 'piecewise constant', 'piecewise constant')
    cSegments    = ('Number of Segments',          6, 1, 4)
    cR0          = ('R0',                          7, 3, 3)
    
    param_list = [ cGamma, cSigma, cVaccStart, cVaccRate, cVaccSigma, cInterv, cSegments, cR0 ]

    params = [name for (name, row, default1, default2) in param_list]
    simul1 = [default1 for (name, row, default1, default2) in param_list]
    simul2 = [default2 for (name, row, default1, default2) in param_list]
    
    params += [30, 120, 210]
    simul1 += [4,4,4]
    simul2 += [0.9, 1.1, 1.7]

    source = ColumnDataSource( dict(
        params= params,      #["Days Infectious", "R0", "Immunity (days)", "Vaccination Start", "Vaccination Rate", "Vaccination Immunity (days)"],
        simul1= simul1,  #[6, 1.7, 0, 0, 0, 0],
        simul2= simul2,  #[6, 1.7, 180, 365, 30, 90],
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
        #gamma_infec = validate_float(data[0])
        #if gamma_infec is None or gamma_infec < 1:
        #    e.add('"Days Infectious" should be 1 or greater')
        
        gamma       = validate_input(data[self.cGamma[1]], e, 'int', 1, None, '"Infectious Period (days)" should be 1 or greater')
        sigma       = validate_input(data[self.cSigma[1]], e, 'int', 0, None, '"Immunity" is the number of days a person remains immune after recovery from infection, or zero for permanent immunity')
        vacc_start  = validate_input(data[self.cVaccStart[1]], e, 'int', 0, None, '"Vaccination Start Day" is the day when the vaccination program starts')
        vacc_rate   = validate_input(data[self.cVaccRate[1]], e, 'float', 0, None, '"Vaccination Rate %/year" is the percentage of suscetible people vaccinated per year')
        vacc_immun  = validate_input(data[self.cVaccSigma[1]], e, 'int', 0, None, '"Vaccination Immunity (days)" is the number of days of immunity given by the vaccine, or zero for permanent immunity')

        interv      = validate_input(data[self.cInterv[1]], e, 'choice', ['piecewise constant', 'piecewise linear'], None, '"Contact Rate Shape" should be "piecewise constant" or "piecewise linear"')
        segments    = validate_input(data[self.cSegments[1]], e, 'int', 1, 15, '"Number of Segments" shoudl be between 1 and 15')
        R0          = validate_input(data[self.cR0[1]], e, 'float', 0.1, 10, '"R0" shoudl be between 0.1 and 10')

        params = {}
        if len(e) == 0:  #all inputs have been validated, with no errors ; this also confirms that none of the variables is None
            params = {
                "population" : population * 1e6,
                "i0"         : 1,
                "gamma"      : 1/gamma,   #1/length of infectious period in days
                "immun"      : 0 if sigma==0 else 1/sigma,
                "vacc_start" : vacc_start,
                "vacc_rate"  : vacc_rate/365/100,
                "vacc_immun" : 0 if vacc_immun==0 else 1/vacc_immun,

                "interv"     : interv,
                "segments"   : segments-1,
                "beta0"       : R0 / gamma,   #(here gamma_infec is a number of days, not a rate)
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
        
            
            
#========================================================
COLORS = palette[4]
print(COLORS)

LINES = {
#   "display name"  : (scenario, column, color, dash)
    "1-Susceptible" : (1, s.cS, COLORS[0], 'solid'), 
    "1-Infectious"  : (1, s.cI, COLORS[1], 'solid'),
    "1-Recovered"   : (1, s.cR, COLORS[2], 'solid'), 
    "1-Vaccinated"  : (1, s.cV, COLORS[3], 'solid'), 
    
    "2-Susceptible" : (2, s.cS, COLORS[0], 'dotted'), 
    "2-Infectious"  : (2, s.cI, COLORS[1], 'dotted'), 
    "2-Recovered"   : (2, s.cR, COLORS[2], 'dotted'), 
    "2-Vaccinated"  : (2, s.cV, COLORS[3], 'dotted'),

}

def on_change_plot(attr, old, new): #a wrapper, required to meet the prototype expected for handlers of Bokeh's on_change events
    return update_plot()
    
def update_plot(new=None):  #the new=None is the prototype expected for handlers of Bokeh's on_click events

    #these globals are refreshed by the simulate() function
    global e
    global n
    global x
    global y1
    global y2

    scale = wPlot.scale()  #"log" or "linear"
    print(scale)

    #create a new figure
    #new_p = figure(plot_height=400, plot_width=800, title="SIRF simulation", y_axis_type=scale, name='plot')
    new_p = figure(title="SIRF simulation", y_axis_type=scale, name='plot')
    new_p.sizing_mode = 'scale_width'

    #get out if there was an error
    wForm.log(e)
    if len(e) != 0:
        print('errors', e)
        return

    #source = ColumnDataSource(dict(x=x, I1=y1[:,s.cI], S1=y1[:,s.cS], R1=y1[:,s.cR], V1=y1[:,s.cV], I2=y2[:,s.cI], S2=y2[:,s.cS], R2=y2[:,s.cR], V2=y2[:,s.cV] ))
    for idx, (line,(scenario, column, color, dash)) in enumerate(LINES.items()):
        
        if wPlot.show_line(line):
            
            y=y1[:,column] if scenario==1 else y2[:,column]
            
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
    
class PlotForm():
    
    wLineSelection = MultiChoice(value=["1-Infectious", "2-Infectious"], options=list(LINES.keys()))
    wLineSelection.on_change("value", on_change_plot)
    def show_line(self, s):
        return True if s in self.wLineSelection.value else False


    wScale = CheckboxGroup(labels=['Log'], active=[])
    wScale.on_click(update_plot)
    def scale(self):
        return "log" if 0 in self.wScale.active else "linear"

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

#form to get all the simulation parameters for the two scenarios
wForm = ParametersForm()
wPlot = PlotForm()

#do a first calculation on default parameters and display the results
#simulate()

# put the button, controls and parameters widgets and plot in a layout and add to the document
#c = column([button, logscale_checkbox, p, wForm.layout], name='layout')
#c.sizing_mode = 'scale_width'
col1 = column([button, wForm.layout], name='col1')
col2 = column([wPlot.layout], name='col2')
c = row([col1,col2], name='layout')
curdoc().add_root(c)
curdoc().title = "COVID Simulation"

simulate()

