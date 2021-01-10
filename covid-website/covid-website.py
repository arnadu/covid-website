
#https://github.com/bokeh/bokeh/blob/branch-2.3/examples/app/sliders.py
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CheckboxGroup, Slider, Button, TextInput, CustomJS
from bokeh.plotting import figure
from bokeh.palettes import Category20 as palette
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
    "gamma_infec": 1/6,
    "beta0": 1.7 / 6,
    "death_rate": 0.5e-2,
    "detection_rate": 10e-2,
    "testing_segments":0,
    "mixing":1,
    "interv":"none"
}


x = np.arange(n)
y = SIRF(x, params)
source = ColumnDataSource(data=dict(x=x, I=y[:,cI], S=y[:,cS], R=y[:,cR]))


#widgets to change the parameters of the simulation
gamma_infec_input = Slider(title="gamma", value=6, start=1, end=21, step=1)

# create a callback that will perform the simulation and update the chart
def simulate():

    global params
    global n
    global x
    global source
    
    params1 = params.copy()
    params1['gamma_infec'] = 1 / gamma_infec_input.value
    
    y = SIRF(x, params1)
    source.data = dict(x=x, I=y[:,cI], S=y[:,cS], R=y[:,cR])
    
    params = params1.copy()
    
# create a plot and style its properties
p = figure(plot_height=400, plot_width=800, title="SIRF simulation", y_axis_type="log")
p.sizing_mode = 'scale_width'

line_I = p.line('x', 'I', source=source, line_width=3, line_alpha=0.6, legend_label='Infectious', line_color=colors[0])
line_S = p.line('x', 'S', source=source, line_width=3, line_alpha=0.6, legend_label='Susceptible', line_color=colors[1])
line_R = p.line('x', 'R', source=source, line_width=3, line_alpha=0.6, legend_label='Recovered', line_color=colors[2])
p.legend.location = "top_right"
p.legend.click_policy="hide"

# add a button widget and configure with the call back
button = Button(label="Simulate")
button.on_click(simulate)


# put the button and plot in a layout and add to the document

c = column([button, gamma_infec_input, p])
c.sizing_mode = 'scale_width'

curdoc().add_root(c)
curdoc().title = "COVID Simulation"

