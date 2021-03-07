import random
import io
import base64
import logging
import json

from flask import Flask, make_response, render_template, request, session, jsonify
from flask_session import Session

from wtforms import Form, BooleanField, StringField, IntegerField, validators

import math
import numpy as np
import pandas as pd

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bokeh.embed import json_item
from bokeh.resources import CDN

from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts  import column
from bokeh.models import LinearAxis, Range1d
from bokeh.models import DatetimeTickFormatter, MonthsTicker, NumeralTickFormatter, Legend
from bokeh.embed import components , json_item
from bokeh.palettes import Category10, Category20, Category20b, Category20c 


from data import Data
from object_dict import objdict

import SISV as sisv
from SISV_calib import SISV_lmfit

#from report import Report
#from piecewiseexp_study import piecewiseexp_study

from flask import Flask, session
from flask_session import Session

'''
https://flask.palletsprojects.com/en/1.1.x/quickstart/
cd ~/web
export FLASK_APP=hello.py
flask run --host=0.0.0.0
'''


#==============================================
#==============================================
#==============================================
app = Flask(__name__,
            static_url_path='/static')
            
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

#available data/results are stored here
sources = {}
series = {}

#==============================================
#==============================================
#==============================================

class FormStudy2(Form):
    region = StringField('Region', default='US')
    state  = StringField('State', default='New York')
    county = StringField('County', default='')
    positives_breaks = IntegerField('Positives Breaks', default=4)
    fatalities_breaks = IntegerField('Fatalities Breaks', default=4)
    minwindow = IntegerField('Minimum Window', default=14)
    absolute = BooleanField('Absolute', default=False)


#==============================================

def validate_string(param, params, default_value=''):
    if param in params:
        return params[param]
    else:
        return default_value
 
def validate_boolean(param, params, default_value=False):
    if param in params:
         p = params[param]
         return True if p=='true' else False
    else:
        return default_value
        
        
#==============================================
@app.route('/sources')
def get():
    #return session.get('sources', {})
    return sources


#==============================================

def dict_to_table(d, f):
    r = '<table class="table table-striped table-bordered table-sm">'
    for i, (k,v) in enumerate(d.items()):
        ff = f[k] #formatter, ff() is a function
        r += "<tr><td>"+k+"</td><td>"+ff(v)+"</td></tr>"
    r += "</table>"
    return r
    
def summary(d, IFR=0.5e-2):
    
    r = {}
    f = {}
    
    
    r['Cumul Fatalities'] = d.fatalities[-1]
    f['Cumul Fatalities'] = lambda x:'{:,.0f}'.format(x)
    
    r['Cumul Fatalities per 100k'] = d.fatalities[-1] * 100e3 / d.population
    f['Cumul Fatalities per 100k'] = lambda x:'{:,.0f}'.format(x)

    r['Population (million)'] = d.population / 1e6
    f['Population (million)'] = lambda x:'{:,.1f}'.format(x)
    
    r['Fatality Rate (Assumption)'] = IFR
    f['Fatality Rate (Assumption)'] = lambda x:'{:.2%}'.format(x)
    
    r['Recovered (Estimated)'] = d.fatalities[-1] / IFR / d.population
    f['Recovered (Estimated)'] = lambda x:'{:.0%}'.format(x)

    r['Fatalities last week'] = np.sum(d.dfatalities[-7:])
    f['Fatalities last week'] = lambda x:'{:,.0f}'.format(x)

    r['Fatalities last week per 100k'] = np.sum(d.dfatalities[-7:]) * 100e3 / d.population
    f['Fatalities last week per 100k'] = lambda x:'{:,.0f}'.format(x)

    r['New Positives last week'] = np.sum(d.dpositives[-7:])
    f['New Positives last week'] = lambda x:'{:,.0f}'.format(x)

    r['New Positive last week per 100k'] = np.sum(d.dpositives[-7:]) * 100e3 / d.population
    f['New Positive last week per 100k'] = lambda x:'{:,.0f}'.format(x)
    
    r['Detection Rate (Estimated)'] = r['New Positives last week'] * IFR /  r['Fatalities last week']
    f['Detection Rate (Estimated)'] = lambda x:'{:.0%}'.format(x)
    
    
    
    #pr = pd.DataFrame(list(r.items()))
    #return pr.to_html(index=False, header=False, formatters=f)
    
    r = dict_to_table(r, f)
    print(r)
    return r
    
    #r = json.dumps(r, cls=NpEncoder)
    #print(r)
    #return r


@app.route('/load', methods=["POST"])
def load():
    
    params = request.json #.form.to_dict(flat=True)
    print('-----------')
    print('load():', params)

    source = 'Johns Hopkins'
    cutoff_positive = 1
    cutoff_death = 1
    truncate = 0

    region = validate_string('region', params)
    state = validate_string('state', params)
    county = validate_string('county', params)
        
    d = Data(source=source, region=region, state=state, county=county, cutoff_positive=cutoff_positive, cutoff_death=cutoff_death, truncate=truncate) 

    sourceId = region + '-' + state + county
    
    #store data server-side
    #sources = session.get('sources', {})
    sources[sourceId] = {
        "id"     : sourceId,
        "type"   : 'data',
        "region" : region,
        "state"  : state,
        "county" : county,
        "data"   : d
        
    }
    
    #list of data series that are made available for plotting in the front-end
    r = { 
        "source": sourceId,
        "series": []
    }

    def register_series(name):
        seriesId = sourceId + '-' + name
        
        s = {
            'id': seriesId, 
            'name': name,
            'type': 'data'
        }
        r['series'].append(s.copy())    #this will be returned to the front-end
        
        s['data'] = d
        series[seriesId]  = s #this is stored server-side, to handle /plot
        

    NAMES = ['Cumul Fatalities', 'Cumul Positives', 'Daily Fatalities', 'Daily Positives']
    for n in NAMES:
        register_series(n)
    
    print(r)
    
    s = summary(d)
    #print(s)
    
    return {"series": r, "summary": {"id": sourceId, "statistics":s}}
             

#==============================================

#recast the model parameters from the front-end format to the format expected by SISV
def recast_params(p):
    p1 = p.copy()
    
    p1['population'] = 1e6 * p['population']
    
    p1['death_rate'] = p['death_rate'] / 100.0
    
    p1['gamma_exp']  = 0 if p['gamma_exp']==0 else 1/p['gamma_exp']  #from number of days to frequency
    p1['gamma']      = 1/p['gamma']  #from number of days to frequency
    p1['gamma_crit'] = 1/p['gamma_crit']  #from number of days to frequency
    p1['gamma_pos']  = 1/p['gamma_pos']  #from number of days to frequency

    p1['detection_rate'] = p['detection_rate'] / 100.0

    p1['immun'] = p['immun']
    p1['vacc_start'] = p['vacc_start']
    p1['vacc_rate'] = p['vacc_rate'] / 100.0 /365  #annual rate of vaccination (of unvaccinated pop)
    p1['vacc_immun'] = p['vacc_immun']

    gamma = p1['gamma']    
    
    p1['beta0'] = float(p['contact_rates'][0]['r0']) * gamma
    p1['beta1'] = float(p['contact_rates'][1]['r0']) * gamma
    
    p1['t1'] = int(p['contact_rates'][1]['t'])

    #exp_stages          = params['exp_stages']              #number of incubation stages [0,nE]
    #inf_stages          = params['inf_stages']              #number of infectious stages [0,nI]
    #crit_stages         = params['crit_stages']             #number of critical stages [0,nC]
    #test_stages         = params['test_stages']             #number of testing stages [0,nT]
    
    return p1

@app.route('/simul', methods=["POST"])
def simul():
    
    params = request.json #.form.to_dict(flat=True)
    print('-----------')
    print('simul():', params)
    
    #run a simulation
    n      = params['n']
    xd_min = params['xd_min']  #needs to be a pandas datetime like value
    x      = np.arange(n)
    xd     = pd.date_range(start=xd_min, periods=n, freq='D')

    p      = params['params']
    p1     = recast_params(p)
    print(p1)

    y      = sisv.SISV_J(x, p1)

    
    #store data server-side
    #sources = session.get('sources', {})
    sourceId = params['scenario']
    sources[sourceId] = {
        "id"     : sourceId,
        "type"   : 'simul',
        "n"      : n,
        "xd_min" : xd_min,
        "params" : p,
        "x"      : x,
        "xd"     : xd,
        "y"      : y
    }
    
    #list of data series that are made available for plotting in the front-end
    r = { 
        "source": sourceId,
        "series": []
    }

    def register_series(name):
        seriesId = sourceId + '-' + name
        
        s = {
            'id': seriesId,
            'name': name,
            'type': 'simul'
        }
        r['series'].append(s.copy())    #this will be returned to the front-end

        s['sourceId'] = sourceId #used by the back-end /plot() function to retrieve the data for this simulation
        series[seriesId]  = s #this is stored server-side, to handle /plot
        

    SIMUL_NAMES = ['Susceptible', 'Infectious', 'Fatalities']
    for n in SIMUL_NAMES:
        register_series(n)

    return {"series": r}
        


#==============================================

def format_fig(f, ylabel="", legend=None):
    
    if legend is not None:
        f.add_layout(Legend(items=legend, location='center'), 'below')
        
    #f.y_range.start = 1
    #f.y_range.range_padding = 0                

    f.yaxis[0].formatter = NumeralTickFormatter(format="0,0")
    
    f.xaxis[0].ticker = MonthsTicker(months=list(range(1,13)))  #[0] needed because we added a second x-axis
    f.xaxis[0].formatter=DatetimeTickFormatter(
            hours=["%d %B %Y"],
            days=["%d %B %Y"],
            months=["%d %B %Y"],
            years=["%d %B %Y"],
        )
    
    f.xgrid.ticker = f.xaxis[0].ticker   
    f.xaxis[0].major_label_orientation = math.pi/4
    
    f.ygrid.grid_line_color = 'navy'
    f.ygrid.grid_line_alpha = 0.3
    f.ygrid.minor_grid_line_color = 'navy'
    f.ygrid.minor_grid_line_alpha = 0.1
    
    f.yaxis[0].axis_label = ylabel
    
    
    #add a second x-axis to count the number of days
    #f.extra_x_ranges['x2'] = Range1d(0,100)   #<<<<<NEED TO FIND A WAY TO GET THE RANGE OF THE PRIMARY AXIS!!!!!
    #ax2 = LinearAxis(x_range_name="x2", axis_label="days")
    #f.add_layout(ax2, 'below')


        
@app.route('/plot', methods=["POST"])
def plot():

    params = request.json #.form.to_dict(flat=True)

    print('-----------')
    print('plot():', params)

    axis_type = "log" if validate_boolean('log', params, False) else "linear"
    relative = validate_boolean('relative', params, False)

    if 'series' in params and len(params['series'])>0:
        
            
        f1 = figure(title='Plot', plot_height=400, plot_width=400, sizing_mode='scale_both', y_axis_type=axis_type)
        legend = []
        
        palette = Category10[10]
        color = 0
        
        for s in params['series']:
            
            id = s['id']
            name = s['name']
            
            if id in series:
                
                if series[id]['type'] == 'data':
                    
                    d = series[id]['data']
                    
                    population = 1 if d.population <=0 else d.population
                    scale_factor = 100000/population if relative else 1.0
                    
                    if name == 'Daily Fatalities':
                        r0 = f1.line(d.xd[d.minD+1:], d.dfatalities * scale_factor, line_width=1, line_color=palette[color], line_dash='dotted', alpha=0.3)
                        r1 = f1.circle(d.xd[d.minD+1:], d.dfatalities * scale_factor, size=5, color=palette[color], alpha=0.3)
                        legend.append((id , [r0, r1]))

                    if name == 'Daily Positives':
                        r0 = f1.line(d.xd[d.minD+1:], d.dfatalities * scale_factor, line_width=1, line_color=palette[color], line_dash='dotted', alpha=0.3)
                        r1 = f1.circle(d.xd[d.minD+1:], d.dfatalities * scale_factor, size=5, color=palette[color], alpha=0.3)
                        legend.append((id   , [r0, r1]))
        
                    if name == 'Cumul Fatalities':
                        r0 = f1.line(d.xd, d.fatalities * scale_factor, line_width=3, line_color=palette[color], line_dash='solid', alpha=1)
                        legend.append((id   , [r0]))
        
                    if name == 'Cumul Positives':
                        r0 = f1.line(d.xd, d.positives * scale_factor, line_width=3, line_color=palette[color], line_dash='solid', alpha=1)
                        legend.append((id   , [r0]))

                if series[id]['type'] == 'simul':
                    
                    sourceId = series[id]['sourceId']
                    source = sources[sourceId] 
                    xd = source['xd']
                    y = source['y']
                    
                    name_to_col = {'Susceptible': sisv.cS, 'Infectious': sisv.cI, 'Fatalities': sisv.cF}

                    col = name_to_col[name]

                    population = source['params']['population']
                    scale_factor = 100000/population if relative else 1.0
                    
                    r0 = f1.line(xd, y[:,col] * scale_factor, line_width=1, line_color=palette[color], line_dash='solid', alpha=1)
                    legend.append((id   , [r0]))

                color = color+1
                if color>=len(palette):
                    color=0
                
                
                
                
        format_fig(f1, 'COVID Evolution', legend)
        r = {'status':'OK', 'fig': json_item(f1)}
        return r
        #print(r)
    else:
        return {'status':'KO', 'msg':'Please select the series to be plotted'}

    


@app.route('/calc2_old', methods=["POST"])
def route_calc2_old():
    form = FormStudy2(request.form)
    if request.method == 'POST' and form.validate():
        params = request.form.to_dict(flat=True)
        print(request)
        print(params)
        
        source = 'Johns Hopkins'
        cutoff_positive = 1
        cutoff_death = 1
        truncate = 0
    
        d = Data(source=source, region=params['region'], state=params['state'], county=params['county'], cutoff_positive=cutoff_positive, cutoff_death=cutoff_death, truncate=truncate) 
        f1 = figure(title='Excess Fatalities', plot_width=800, plot_height=600 )#, y_axis_type="log")
        r0 = f1.line(d.xd[d.minD+1:], d.dfatalities, line_width=1, line_color='red', line_dash='dotted', alpha=0.3)
        r1 = f1.circle(d.xd[d.minD+1:], d.dfatalities, size=5, color="red", alpha=0.3)
        legend = [
            ("COVID fatalities"   , [r0, r1]),
        ]
        format_fig(f1, 'Number of deaths per day', legend)

        return json.dumps(json_item(f1))
        #return jsonify({'script':fig_script, 'div':fig_div})

@app.route('/calc2', methods=["POST"])
def route_calc2():
    
    if 'data' in session and session['data'] is not None:

        d = session['data']
        f1 = figure(title='Excess Fatalities', plot_width=800, plot_height=600 )#, y_axis_type="log")
        r0 = f1.line(d.xd[d.minD+1:], d.dfatalities, line_width=1, line_color='red', line_dash='dotted', alpha=0.3)
        r1 = f1.circle(d.xd[d.minD+1:], d.dfatalities, size=5, color="red", alpha=0.3)
        legend = [
            ("COVID fatalities"   , [r0, r1]),
        ]
        format_fig(f1, 'Number of deaths per day', legend)

        return json.dumps(json_item(f1))
        #return jsonify({'script':fig_script, 'div':fig_div})        
    
@app.route('/study3', methods=["GET","POST"])
def route_study3():
    
    
    form = FormStudy2(request.form)
    if request.method == 'POST' and form.validate():

        params = request.form.to_dict(flat=True)
        
        session['params'] = params
        
        source = 'Johns Hopkins'
        cutoff_positive = 1
        cutoff_death = 1
        truncate = 0
        
        
        d = Data(source=source, region=params['region'], state=params['state'], county=params['county'], cutoff_positive=cutoff_positive, cutoff_death=cutoff_death, truncate=truncate) 
        ed = d.excess_deaths()
        h = d.hospitalization()

        scale = 1 if 'absolute' in params and params['absolute'] else 1e5/d.population
        
        session['data'] = d

        f1 = figure(title='Excess Fatalities', plot_width=800, plot_height=600 )#, y_axis_type="log")
        
        r0 = f1.line(d.xd[d.minD+1:], scale*d.dfatalities, line_width=1, line_color='red', line_dash='dotted', alpha=0.3)
        r1 = f1.circle(d.xd[d.minD+1:], scale*d.dfatalities, size=5, color="red", alpha=0.3)
        
        #plot 7-day rolling average
        rolling = pd.DataFrame(data = scale*d.dfatalities).interpolate().rolling(7).mean()
        r2 = f1.line(d.xd[d.minD+1:], rolling.loc[:,0].values, line_width=1, line_color='red')
        
        legend = [
            ("COVID fatalities"   , [r0, r1]),
            ("COVID 7-day average"   , [r2]),
        ]
        
        if ed is not None:
            r3 = f1.line(ed['date'], scale*ed['total_deaths']/7, line_width=1, line_color='black')
            r4 = f1.line(ed['date'], scale*ed['excess_deaths']/7, line_width=1, line_color='grey')
        
            legend.append(("Total Deaths", [r3]))
            legend.append(("Excess Deaths", [r4]))
            
        format_fig(f1, 'Number of deaths per day', legend)

        #----------------------------
        f2 = figure(title='Positives and Hospitalizations', plot_width=800, plot_height=600 )#, y_axis_type="log")
        
        r0 = f2.line(d.xd[d.minP+1:], scale*d.dpositives, line_width=1, line_color='black', line_dash='dotted', alpha=0.3)
        r1 = f2.circle(d.xd[d.minP+1:], scale*d.dpositives, size=5, color="black", alpha=0.3)
        
        #plot 7-day rolling average
        rolling = pd.DataFrame(data = scale*d.dpositives).interpolate().rolling(7).mean()
        r2 = f2.line(d.xd[d.minP+1:], rolling.loc[:,0].values, line_width=1, line_color='black')
        
        if h is not None:
            h = h[h['hospitalizedCurrently']>1]
            r3 = f2.line(h['date'], scale*h['hospitalizedCurrently'], line_width=1, line_color='orange')

        legend = [
            ("COVID Daily Positive Test Results"   , [r0, r1]),
            ("COVID Tests 7-day average"   , [r2]),
            ("COVID In Hospital"   , [r3]),
        ]
        
        format_fig(f2, 'Daily New Positive Test Results and Currently Hospitalized', legend)


        #p = column(f1,f2)

        fig_script , (fig_div1, fig_div2) = components((f1,f2))

        # following above points: 
        #  + pass plot object 'p' into json_item
        #  + wrap the result in json.dumps and return to frontend
        #return json.dumps(json_item(p, "myplot"))
    
        return render_template('study3.html', form=form, bokeh_script=fig_script, bokeh_div1=fig_div1, bokeh_div2=fig_div2)
        

    else:
        
        session['params'] = None
        session['data'] = None
        
        return render_template('study3.html', form=form)    
        
        
#==============================================
'''
@app.route('/study2', methods=["GET","POST"])
def route_study2():
    
    form = FormStudy2(request.form)
    if request.method == 'POST' and form.validate():

        params = request.form.to_dict(flat=True)
        
        source = 'Johns Hopkins'
        cutoff_positive = 1
        cutoff_death = 1
        truncate = 0

        d = Data(source=source, region=params['region'], state=params['state'], county=params['county'], cutoff_positive=cutoff_positive, cutoff_death=cutoff_death, truncate=truncate) 

        o = Report()
        o = piecewiseexp_study(d, o, positives_breaks=int(params['positives_breaks']), fatalities_breaks=int(params['fatalities_breaks']), minwindow=int(params['minwindow']))

        return render_template('study2.html', form=form, 
                                positives_doubling=o['Positives_Doubling_Time'], 
                                fatalities_doubling=o['Fatalities_Doubling_Time'], 
                                chart=o.html_chart('PieceWise_Exponential_Growth_model'))

    else:

        return render_template('study2.html', form=form, positives_doubling=None, fatalities_doubling=None, chart=None)

'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)