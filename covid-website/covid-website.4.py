import math

#https://github.com/bokeh/bokeh/blob/branch-2.3/examples/app/sliders.py
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CheckboxGroup, Slider, Button, TextInput, CustomJS, Div, MultiChoice
from bokeh.plotting import figure
from bokeh.palettes import Category20 as palette
from bokeh.models.widgets import Tabs, Panel
from bokeh.models import NumeralTickFormatter

from bokeh.models import DataTable, TableColumn

import numpy as np
from SIR import *




#print(palette)
colors = palette[3]
print(colors)

# create a callback that will perform the simulation and update the chart
n = 300

params = {
    "population": 100e3,
    "i0": 1,
    "p0": 1,
    "f0": 0,
    "gamma_infec": 1/6,   #1/length of infectious period in days
    "beta0": 2.5 * (1/6),   #R0 =beta/gamma
    "death_rate": 0.5e-2,
    "detection_rate": 10e-2,
    "testing_segments":0,
    "mixing":1,
    "interv":"none"
}



def validate_int(s):
    try:
        return int(s)
    except:
        return None

def validate_float(s):
    try:
        f = float(s)
        return f if math.isfinite(f) else None
    except:
        return None

class ParametersForm:

    wMessage = Div(text="")

    wPopulation = TextInput(value='1.0', title='Population (millions)')
    wSimulHorizon = TextInput(value='300', title='Simulation Horizon (days)')
    
    source = ColumnDataSource(dict(
        params=["Days Infectious", "R0", "IFR (%)", "Detection (%)"],
        simul1=[6, 1.7, 0.5, 10],
        simul2=[5, 2,   1,   15]
    ))
    
    columns = [
        TableColumn(field='params', title='Parameter'),
        TableColumn(field='simul1', title='Simul #1'),
        TableColumn(field='simul2', title='Simul #2')
    ]
    
    wTable= DataTable(source=source, columns=columns, editable=True, sortable=False)
    
    layout = column(wMessage, wPopulation, wSimulHorizon, wTable)

        
    #these params apply to all scenarios
    def meta_params(self):
        
        e=set()
        
        n = validate_int(self.wSimulHorizon.value)
        population = validate_float(self.wPopulation.value)
        
        if n < 50:
            e.add('Simulation Horizon should be greater than 50')

        if population < 1:
            e.add('Population should be greater than 1 (million)')

        return e, n, population
        
    #get the params for a given scehario
    def scenario_params(self, i):
        
        e, n, population = self.meta_params()
        
        data = self.source.data['simul1'] if i==1 else self.source.data['simul2']
        #params=["Days Infectious", "R0", "IFR (%)", "Detection (%)"],

        gamma_infec = validate_float(data[0])
        if gamma_infec is None or gamma_infec < 1:
            e.add('"Days Infectious" should be 1 or greater')

        r0 = validate_float(data[1])
        if r0 is None or r0 < 0.1 or r0 > 10:
            e.add('"R0" shoudl be between 0.1 and 10')
          
        death_rate = validate_float(data[2])
        if death_rate is None or death_rate < 0.01 or death_rate > 5:
            e.add('"IFR (%)" should be between 0.01 and 5 (%)')
        
        detection_rate = validate_float(data[3])
        if detection_rate is None or detection_rate < 1 or detection_rate > 100:
            e.add('"Detection (%)" should be between 1 and 100')
        
        params = {}
        if len(e) == 0:
            params = {
                "population": population * 1e6,
                "i0": 1,
                "p0": 1,
                "f0": 0,
                "gamma_infec": 1/gamma_infec,   #1/length of infectious period in days
                "beta0": r0 / gamma_infec,   #(here gamma_infec is a number of days, not a rate)
                "death_rate": death_rate/100.0,
                "detection_rate": detection_rate/100.0,
                "testing_segments":0,
                "mixing":1,
                "interv":"none"
            }
            #print(params)
        return e, n, params

    def log(self, e):
        self.wMessage.text=""
        for m in e:
            self.wMessage.text += m + "<br>"


    
def plot(new=None):

    global e
    global n
    global x
    global y1
    global y2

    scale = wPlot.scale()
    print(scale)

    #create a new figure
    #new_p = figure(plot_height=400, plot_width=800, title="SIRF simulation", y_axis_type=scale, name='plot')
    new_p = figure(title="SIRF simulation", y_axis_type=scale, name='plot')
    new_p.sizing_mode = 'scale_width'

    #get out if there was an error
    wForm.log(e)
    if len(e) != 0:
        print('errors', e)
        p = new_p
        return


    
    source = ColumnDataSource(dict(x=x, I1=y1[:,cI], S1=y1[:,cS], R1=y1[:,cR],I2=y2[:,cI], S2=y2[:,cS], R2=y2[:,cR] ))
    
    if wPlot.show_line('1-Infectious'):
        line_I1 = new_p.line('x', 'I1', source=source, line_width=3, line_alpha=0.6, legend_label='1-Infectious', line_color=colors[0], line_dash='solid')
        
    if wPlot.show_line('1-Susceptible'):
        line_S1 = new_p.line('x', 'S1', source=source, line_width=3, line_alpha=0.6, legend_label='1-Susceptible', line_color=colors[1], line_dash='solid')

    if wPlot.show_line('1-Recovered'):
        line_R1 = new_p.line('x', 'R1', source=source, line_width=3, line_alpha=0.6, legend_label='1-Recovered', line_color=colors[2], line_dash='solid')

    if wPlot.show_line('2-Infectious'):
        line_I2 = new_p.line('x', 'I2', source=source, line_width=3, line_alpha=0.6, legend_label='2-Infectious', line_color=colors[0], line_dash='dotted')
    
    if wPlot.show_line('2-Susceptible'):
        line_S2 = new_p.line('x', 'S2', source=source, line_width=3, line_alpha=0.6, legend_label='2-Susceptible', line_color=colors[1], line_dash='dotted')

    if wPlot.show_line('2-Recovered'):
        line_R2 = new_p.line('x', 'R2', source=source, line_width=3, line_alpha=0.6, legend_label='2-Recovered', line_color=colors[2], line_dash='dotted')
    
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
    
    OPTIONS = ["1-Susceptible", "1-Infectious", "1-Recovered", "2-Susceptible", "2-Infectious", "2-Recovered"]

    wLineSelection = MultiChoice(value=["1-Infectious", "2-Infectious"], options=OPTIONS)
    #wLineSelection.on_change(plot)
    def show_line(self, s):
        return True if s in self.wLineSelection.value else False
    
    wScale = CheckboxGroup(labels=['Log'], active=[])
    wScale.on_click(plot)
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
        y1 = SIRF(x, params1)
        y2 = SIRF(x, params2)
    
    plot()
    

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

