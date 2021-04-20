import random
import io
import base64
import logging
import json
from datetime import datetime
from collections import OrderedDict

from flask import Flask, make_response, render_template, request, session, jsonify, redirect, url_for
from flask_session import Session
from flask_executor import Executor

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

from maxlik import piecewiseexp_diffevol
import SISV as sisv
from SISV_calib import SISV_lmfit

#from report import Report
#from piecewiseexp_study import piecewiseexp_study


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

executor = Executor(app)
app.config['EXECUTOR_TYPE'] = 'thread'
app.config['EXECUTOR_PROPAGATE_EXCEPTIONS'] = True
            
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
@app.route('/')
def home():
    return redirect(url_for('static/index.html'))

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
    
    r = dict_to_table(r, f)
    #app.logger.debug(r)
    return r


@app.route('/load', methods=["POST"])
def load():
    try:
        params = request.json #.form.to_dict(flat=True)

        app.logger.debug('----------')
        app.logger.debug('load(): %s', params)
    
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
        
        s = summary(d)
        app.logger.debug(s)
        
        return {"status":"OK", "msg":"Data loaded successfully; select the results to plot or click the location's button to see the statistics", "series": r, "summary": {"id": sourceId, "statistics":s}}
        
    except:
        app.logger.exception('')
        return{"status":"KO", 'msg':''}
        

@app.route('/regions', methods=["POST"])
def regions():
    try:
        
        app.logger.debug('----------')
        app.logger.debug('regions()')
        
        r = Data().regions()
        return {'status':'OK', 'regions':r}
        

    except:
        app.logger.exception('')
        return{"status":"KO", 'msg':'Something went horribly wrong'}

@app.route('/states', methods=["POST"])
def states():
    try:
        params = request.json #.form.to_dict(flat=True)

        app.logger.debug('----------')
        app.logger.debug('states(): %s', params)
        
        r = Data().states(params['region'])
        return {'status':'OK', 'states': r}
        

    except:
        app.logger.exception('')
        return{"status":"KO", 'msg':'Something went horribly wrong'}

#==============================================

#recast the model parameters from the front-end format to the format expected by SISV
def recast_params_inv(p):
    
    rp = p.copy()
    
    rp['population'] = float(p['population']) / 1e6

    rp['i0'] = float(p['i0'])
    rp['f0'] = float(p['f0'])
    rp['p0'] = float(p['p0'])

    
    rp['death_rate'] = float(p['death_rate']) * 100.0
    
    rp['gamma_exp']  = 0 if float(p['gamma_exp'])==0 else 1/float(p['gamma_exp'])  #from number of days to frequency
    rp['gamma']      = 1/float(p['gamma'])  #from number of days to frequency
    rp['gamma_crit'] = 1/float(p['gamma_crit'])  #from number of days to frequency
    rp['gamma_pos']  = 1/float(p['gamma_pos'])  #from number of days to frequency

    rp['detection_rate'] = float(p['detection_rate']) * 100.0

    rp['immun'] = int(p['immun']) 
    rp['vacc_start'] = int(p['vacc_start']) 
    rp['vacc_rate'] = float(p['vacc_rate']) * 100.0 *365  #annual rate of vaccination (of unvaccinated pop)
    rp['vacc_immun'] = int(p['vacc_immun']) 

    rp['exp_stages'] = int(p['exp_stages']) 
    rp['inf_stages'] = int(p['inf_stages']) 
    rp['crit_stages'] = int(p['crit_stages']) 
    rp['test_stages'] = int(p['test_stages']) 

    gamma = p['gamma']    
    segments = p['segments']
    
#    cr = [ {'t':'0', 'r0': p['beta0'] / gamma} ]
#    for i in range(1,segments+1):
#        ti = float(p['t{}'.format(i)])
#        ri = float(p['beta{}'.format(i)]) / gamma
#        cr.append({'t':ti,'r0':ri})

    cr = [ OrderedDict([('t', '0'), ('r0', p['beta0'] / gamma)]) ]
    for i in range(1,segments+1):
        ti = float(p['t{}'.format(i)])
        ri = float(p['beta{}'.format(i)]) / gamma
        cr.append(OrderedDict([('t',ti),('r0',ri)]))
    
    rp['contact_rates'] = cr

    for i in range(0, segments+1):
        rp.pop(p['beta{}'.format(i)], None)
        if i<segments: rp.pop(p['aux{}'.format(i)], None)
        if i>0: rp.pop(p['t{}'.format(i)], None)
    
    return rp


#recast the model parameters from the front-end format to the format expected by SISV
def recast_params(p):
    p1 = p.copy()
    
    p1['population'] = 1e6 * float(p['population'])
    
    p1['i0'] = float(p['i0'])
    
    p1['death_rate'] = float(p['death_rate']) / 100.0
    
    p1['gamma_exp']  = 0 if float(p['gamma_exp'])==0 else 1/float(p['gamma_exp'])  #from number of days to frequency
    p1['gamma']      = 1/float(p['gamma'])  #from number of days to frequency
    p1['gamma_crit'] = 1/float(p['gamma_crit'])  #from number of days to frequency
    p1['gamma_pos']  = 1/float(p['gamma_pos'])  #from number of days to frequency

    p1['detection_rate'] = float(p['detection_rate']) / 100.0

    p1['immun'] = int(p['immun'])
    p1['vacc_start'] = int(p['vacc_start'])
    p1['vacc_rate'] = float(p['vacc_rate']) / 100.0 /365  #annual rate of vaccination (of unvaccinated pop)
    p1['vacc_immun'] = int(p['vacc_immun'])

    p1['exp_stages'] = int(p['exp_stages']) 
    p1['inf_stages'] = int(p['inf_stages']) 
    p1['crit_stages'] = int(p['crit_stages']) 
    p1['test_stages'] = int(p['test_stages']) 

    gamma = p1['gamma']    
    
    p1['beta0'] = float(p['contact_rates'][0]['r0']) * gamma
    
    cr = p['contact_rates'] #table of contact rates
    for i, r in enumerate(cr):
        if i>0: #first row is beta0, already done
            p1['beta{}'.format(i)] = float(r['r0']) * gamma
            p1['t{}'.format(i)] = int(r['t'])
    
    p1['segments'] = len(cr) - 1
    
    #exp_stages          = params['exp_stages']              #number of incubation stages [0,nE]
    #inf_stages          = params['inf_stages']              #number of infectious stages [0,nI]
    #crit_stages         = params['crit_stages']             #number of critical stages [0,nC]
    #test_stages         = params['test_stages']             #number of testing stages [0,nT]
    
    return p1

@app.route('/simul', methods=["POST"])
def simul():
    try:
    
        params = request.json #.form.to_dict(flat=True)
        app.logger.debug('-----------')
        app.logger.debug('simul():', params)
        
        #run a simulation
        n      = int(params['n'])
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
            
    
        SIMUL_NAMES = ['Susceptible', 'Infectious', 'Fatalities', 'Daily Fatalities']
        for n in SIMUL_NAMES:
            register_series(n)
    
        return {"status":"OK", "msg":"Simulation ran successfully; select the results to plot", "series": r}
        
    except:
        app.logger.exception('')
        return{"status":"KO", "msg": "Something went horribly wrong"}        


#==============================================

def format_fig(f, ylabel="", legend=None):
    
    if legend is not None:
        f.add_layout(Legend(items=legend, location='center'), 'above')
        
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
    try:

        params = request.json #.form.to_dict(flat=True)
    
        app.logger.debug('-----------')
        app.logger.debug('plot():', params)
    
        #axis_type = "log" if validate_boolean('log', params, False) else "linear"
        relative = validate_boolean('relative', params, False)
    
        if 'series' in params and len(params['series'])>0:
            
                
#            f1 = figure(title='Plot', plot_height=400, plot_width=400, sizing_mode='scale_both', y_axis_type=axis_type)
            f_lin = figure(title='Linear', y_axis_type='linear', plot_height=200, plot_width=300, sizing_mode='scale_both')
            f_log = figure(title='Log', y_axis_type='log', plot_height=200, plot_width=300, sizing_mode='scale_both')
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
                            r0 = f_lin.line(d.xd[d.minD+1:], np.nan_to_num(d.dfatalities) * scale_factor, line_width=1, line_color=palette[color], line_dash='dotted', alpha=0.3)
                            r1 = f_lin.circle(d.xd[d.minD+1:], np.nan_to_num(d.dfatalities) * scale_factor, size=5, color=palette[color], alpha=0.3)
                            legend.append((id , [r0, r1]))

                            f_log.line(d.xd[d.minD+1:], np.nan_to_num(d.dfatalities) * scale_factor, line_width=1, line_color=palette[color], line_dash='dotted', alpha=0.3)
                            f_log.circle(d.xd[d.minD+1:], np.nan_to_num(d.dfatalities) * scale_factor, size=5, color=palette[color], alpha=0.3)
    
                        if name == 'Daily Positives':
                            r0 = f_lin.line(d.xd[d.minP+1:], np.nan_to_num(d.dpositives) * scale_factor, line_width=1, line_color=palette[color], line_dash='dotted', alpha=0.3)
                            r1 = f_lin.circle(d.xd[d.minP+1:], np.nan_to_num(d.dpositives) * scale_factor, size=5, color=palette[color], alpha=0.3)
                            legend.append((id   , [r0, r1]))

                            f_log.line(d.xd[d.minP+1:], np.nan_to_num(d.dpositives) * scale_factor, line_width=1, line_color=palette[color], line_dash='dotted', alpha=0.3)
                            f_log.circle(d.xd[d.minP+1:], np.nan_to_num(d.dpositives) * scale_factor, size=5, color=palette[color], alpha=0.3)
            
                        if name == 'Cumul Fatalities':
                            r0 = f_lin.line(d.xd, d.fatalities * scale_factor, line_width=3, line_color=palette[color], line_dash='solid', alpha=1)
                            legend.append((id   , [r0]))
                            f_log.line(d.xd, d.fatalities * scale_factor, line_width=3, line_color=palette[color], line_dash='solid', alpha=1)
            
                        if name == 'Cumul Positives':
                            r0 = f_lin.line(d.xd, d.positives * scale_factor, line_width=3, line_color=palette[color], line_dash='solid', alpha=1)
                            legend.append((id   , [r0]))
                            f_log.line(d.xd, d.positives * scale_factor, line_width=3, line_color=palette[color], line_dash='solid', alpha=1)
    
                    if series[id]['type'] == 'piecewiseexp':
                        
                        sourceId = series[id]['sourceId']
                        s = sources[sourceId]
                        d = s['data']
                        eg = s['piecewiseexp']

                        population = 1 if d.population <=0 else d.population
                        scale_factor = 100000/population if relative else 1.0
                        
                        if name == 'Fatalities Exp. Growth':
                            r0 = f_lin.line(eg['xd'], eg['y'] * scale_factor, line_width=1, line_color=palette[color], line_dash='solid', alpha=1)
                            legend.append((id , [r0]))
                            f_log.line(eg['xd'], eg['y'] * scale_factor, line_width=1, line_color=palette[color], line_dash='solid', alpha=1)

                    if series[id]['type'] == 'simul':
                        
                        sourceId = series[id]['sourceId']
                        source = sources[sourceId] 
                        xd = source['xd']
                        y = source['y']
                        
                        name_to_col = {'Susceptible': sisv.cS, 'Infectious': sisv.cI, 'Fatalities': sisv.cF, 'Daily Fatalities': sisv.cF}
    
                        col = name_to_col[name]
    
                        population = source['params']['population'] * 1e6
                        scale_factor = 100000/population if relative else 1.0
                        print(population)
                        
                        if name == 'Daily Fatalities':
                            r0 = f_lin.line(xd[1:], np.diff(y[:,col]) * scale_factor, line_width=1, line_color=palette[color], line_dash='solid', alpha=1)
                            f_log.line(xd[1:], np.diff(y[:,col]) * scale_factor, line_width=1, line_color=palette[color], line_dash='solid', alpha=1)
                        else:
                            r0 = f_lin.line(xd, y[:,col] * scale_factor, line_width=1, line_color=palette[color], line_dash='solid', alpha=1)
                            f_log.line(xd, y[:,col] * scale_factor, line_width=1, line_color=palette[color], line_dash='solid', alpha=1)
                        legend.append((id   , [r0]))
    
                    color = color+1
                    if color>=len(palette):
                        color=0

            format_fig(f_lin, 'COVID Evolution', legend)
            format_fig(f_log, 'COVID Evolution')
            r = {'status':'OK', 'msg':'', 'fig_lin': json_item(f_lin), 'fig_log': json_item(f_log)}
            return r
            
        else:
            return {'status':'KO', 'msg':'Please select the series to be plotted'}
            
    except:
        app.logger.exception('')
        return{"status":"KO", 'msg':'Something went horribly wrong'}        

#-----------------------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
@executor.job
def calib_piecewiseexp_job(params):

    app.logger.debug(params['jobId'], ' starting')

    p, breakpoints, likelihood, fit = piecewiseexp_diffevol(params['x'], params['y'], breaks=params['breaks'], minwindow=7)

    app.logger.debug(params['jobId'], ' finished')

    return p, breakpoints, likelihood, fit, params

#-------------------------------------------------------------
@app.route('/calib_piecewiseexp_get', methods=["POST"])
def calib_piecewiseexp_get():
        req = request.json #.form.to_dict(flat=True)
    
        app.logger.debug('-----------')
        app.logger.debug('calibrate_piecewiseexp_get():', req)
        
        jobId = req['jobId']

        if not executor.futures.done(jobId):
            return {'status':'running', 'msg':executor.futures._state(jobId)}
        
        future = executor.futures.pop(jobId)
        p, breakpoints, likelihood, fit, params = future.result()

        sourceId = params['sourceId']
        s = sources[sourceId]
        app.logger.debug(s)
        
        eg = {
            'x': params['x'],
            'xd' : params['xd'],
            'y': fit,
            'breakpoints': [],
            'f0': p[0],
        }
        
        eg['breakpoints'].append({'t':0, 'd': math.log(2)/p[1]}) #initial doubling period
        for i in range(len(breakpoints)):
            eg['breakpoints'].append({'t':breakpoints[i], 'd': math.log(2)/p[i+2]}) #successive doubling periods

        s['piecewiseexp'] = eg.copy() #store a copy of the results server side, in the dictionary of sources

        app.logger.debug(eg)
        
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
                'type': 'piecewiseexp'
            }
            r['series'].append(s.copy())    #this will be returned to the front-end
    
            s['sourceId'] = sourceId #used by the back-end /plot() function to retrieve the data for this simulation
            series[seriesId]  = s #this is stored server-side, to handle /plot
            
    
        SIMUL_NAMES = ['Fatalities Exp. Growth']
        for n in SIMUL_NAMES:
            register_series(n)

        app.logger.debug(r)
        
        app.logger.debug(sources[sourceId]['piecewiseexp'])
    
        return {"status":"OK", "msg":"Calibration ran successfully; select the results to plot", "series": r}


        


#-------------------------------------------------------------
#https://testdriven.io/blog/flask-contexts-advanced/
@app.route('/calib_piecewiseexp_start', methods=["POST"])
def calib_piecewiseexp_start():
    try:

        req = request.json #.form.to_dict(flat=True)
    
        app.logger.debug('-----------')
        app.logger.debug('calibrate_piecewiseexp_start():', req)
        
        region = req['region']
        state = req['state']
        county = req['county']
        breaks = int(req['breaks'])
    
        sourceId = region + '-' + state + county
    
        if sourceId in sources:    
            
            jobId = 'EXP-{}-{}'.format(sourceId,breaks)
            s = sources[sourceId]
            
            d = s['data']
            params = {
                'sourceId' : sourceId,
                'jobId'    : jobId,
                'x'        : d.x[d.minD+1:],
                'xd'       : d.xd[d.minD+1:],
                'y'        : d.dfatalities,
                'breaks'   : breaks
            }
            
            #p, breakpoints, likelihood, fit = piecewiseexp_diffevol(x, y, breaks=breaks, minwindow=7)
            calib_piecewiseexp_job.submit_stored(jobId, params)
            app.logger.debug('launched ' + jobId)
        
            return {"status":"OK", "msg":"Calibration process has started", "jobId": jobId}

    except:
        app.logger.exception('')
        return{"status":"KO", 'msg':'Something went horribly wrong'}        


#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
@executor.job
def calib_sir_job(params):

    app.logger.debug('starting '+ params['jobId'])

    d = params['d']
    o = params['overrides']
    
    o['population'] = d.population

    #p, breakpoints, likelihood, fit = piecewiseexp_diffevol(params['x'], params['y'], breaks=params['breaks'], minwindow=7)
    p = SISV_lmfit(d, overrides=o, solver='leastsq')

    app.logger.debug(params['jobId'] + ' finished')

    return p, d.minDate

#-------------------------------------------------------------

def print_types(r):
    for k,(v1,v2) in enumerate(r.items()):
        print(v1,type(v2),v2)

@app.route('/calib_sir_get', methods=["POST"])
def calib_sir_get():
        req = request.json #.form.to_dict(flat=True)
    
        app.logger.debug('-----------')
        app.logger.debug('calibrate_sir_get():', req)
        
        jobId = req['jobId']

        if not executor.futures.done(jobId):
            return {'status':'running', 'msg':executor.futures._state(jobId)}
        
        future = executor.futures.pop(jobId)
        p, xd_min = future.result()
        
        rp =recast_params_inv(p)

        r = {
            'xd_min': xd_min.strftime('%d-%b-%Y'),
            'params': rp
        }
        
        print('xd_min', type(r['xd_min']))
        print_types(r['params'])
        
        return {"status":"OK", "msg":"Calibration ran successfully; select the results to plot", 'res':r}


#https://testdriven.io/blog/flask-contexts-advanced/
@app.route('/calib_sir_start', methods=["POST"])
def calib_sir_start():
    try:

        req = request.json #.form.to_dict(flat=True)
    
        app.logger.debug('-----------')
        app.logger.debug('calibrate_sir_start():', req)

        p      = req['params']
        print(p)
        p1     = recast_params(p['params'])
        segments = p1['segments']
        print(p1)
    
        sourceId = req['sourceId']
        if sourceId in sources:    
            
            jobId = 'SIR-{}-{}'.format(sourceId, segments)
            s = sources[sourceId]
            d = s['data']
            

            params = {
                'sourceId' : sourceId,
                'jobId'    : jobId,
                'd'        : d,
                'overrides': p1 #{
                #    'segments': segments
                #}
            }
            
            #p, breakpoints, likelihood, fit = piecewiseexp_diffevol(x, y, breaks=breaks, minwindow=7)
            calib_sir_job.submit_stored(jobId, params)
            app.logger.debug('launched ' + jobId)
        
            return {"status":"OK", "msg":"Calibration process has started", "jobId": jobId}

    except:
        app.logger.exception('')
        return{"status":"KO", 'msg':'Something went horribly wrong'}        





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