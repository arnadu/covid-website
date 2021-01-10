
#https://github.com/bokeh/bokeh/blob/branch-2.3/examples/app/sliders.py
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CheckboxGroup, Slider, Button, TextInput, CustomJS
from bokeh.plotting import figure
from bokeh.palettes import Category20 as palette
from bokeh.models.widgets import Tabs, Panel
from bokeh.models import NumeralTickFormatter

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




#widgets to change the parameters of the simulation
gamma_infec_input = Slider(title="gamma", value=6, start=1, end=21, step=1)


def plot(new=None):

    global x
    global y
    global p
    global logscale_checkbox
    
    scale = "log" if 0 in logscale_checkbox.active else "linear"
    print(scale)

    #remove the current figure (if it exists)
    layouts = curdoc().get_model_by_name('layout')
    if layouts is not None:
        try:
            old_p = curdoc().get_model_by_name('plot')
            if old_p is not None:
                layouts.children.remove(old_p)
        except:
            pass
    
    #create a new figure
    new_p = figure(plot_height=400, plot_width=800, title="SIRF simulation", y_axis_type=scale, name='plot')
    new_p.sizing_mode = 'scale_width'
    
    source = ColumnDataSource(dict(x=x, I=y[:,cI], S=y[:,cS], R=y[:,cR]))
    
    line_I = new_p.line('x', 'I', source=source, line_width=3, line_alpha=0.6, legend_label='Infectious', line_color=colors[0])
    line_S = new_p.line('x', 'S', source=source, line_width=3, line_alpha=0.6, legend_label='Susceptible', line_color=colors[1])
    line_R = new_p.line('x', 'R', source=source, line_width=3, line_alpha=0.6, legend_label='Recovered', line_color=colors[2])
    
    new_p.yaxis.formatter = NumeralTickFormatter(format='0,0.0')
    
    new_p.xaxis.axis_label = 'days'
    new_p.yaxis.axis_label = 'count'
    
    new_p.legend.location = "top_right"
    new_p.legend.click_policy="hide"

    #push the new figure to the page
    if layouts is not None:
        layouts.children.append(new_p)
    p = new_p


# create a callback that will perform the simulation and update the chart
def simulate():

    global params
    global n
    global x
    global y

    params1 = params.copy()
    params1['gamma_infec'] = 1 / gamma_infec_input.value
    print(gamma_infec_input.value)

    params = params1.copy()
    
    x = np.arange(n)
    y = SIRF(x, params)
    
    plot()
    

# add a button widget and configure with the call back
button = Button(label="Simulate")
button.on_click(simulate)

#add a check box for log or linear scale
logscale_checkbox = CheckboxGroup(labels=['Log'], active=[])
logscale_checkbox.on_click(plot)

#do a first calculation on default parameters and display the results
simulate()

# put the button and plot in a layout and add to the document

c = column([button, gamma_infec_input, logscale_checkbox, p], name='layout')
c.sizing_mode = 'scale_width'

curdoc().add_root(c)
curdoc().title = "COVID Simulation"

