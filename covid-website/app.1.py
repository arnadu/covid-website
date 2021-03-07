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
from bokeh.models import DatetimeTickFormatter, MonthsTicker, NumeralTickFormatter, Legend
from bokeh.embed import components , json_item

from data import Data
from object_dict import objdict

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
        
#==============================================
@app.route('/sources')
def get():
    #return session.get('sources', {})
    return sources

#==============================================
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
    
    #list of data series that are made available for plotting in the front-end
    series = { 
        "source": sourceId,
        "series": [
            {"id": sourceId+"-Fatalities", "name":"Fatalities"},
            {"id": sourceId+"-Positives", "name":"Positives"},
        ]
    }
    
    #sources = session.get('sources', {})
    sources[sourceId] = {
        "id" : sourceId,
        "region" : region,
        "state" : state,
        "county" : county,
        "data" : d
        
    }
    
    return series
             

#==============================================

def format_fig(f, ylabel="", legend=None):
    
    if legend is not None:
        f.add_layout(Legend(items=legend, location='center'), 'below')
        
       
    f.yaxis[0].formatter = NumeralTickFormatter(format="0,0")
    
    f.xaxis.ticker = MonthsTicker(months=list(range(1,13)))
    
    f.xaxis.formatter=DatetimeTickFormatter(
            hours=["%d %B %Y"],
            days=["%d %B %Y"],
            months=["%d %B %Y"],
            years=["%d %B %Y"],
        )
    
    f.xgrid.ticker = f.xaxis.ticker
    
    f.xaxis.major_label_orientation = math.pi/4
    
    f.ygrid.grid_line_color = 'navy'
    f.ygrid.grid_line_alpha = 0.3
    f.ygrid.minor_grid_line_color = 'navy'
    f.ygrid.minor_grid_line_alpha = 0.1
    
    f.yaxis[0].axis_label = ylabel
    
        
@app.route('/plot', methods=["POST"])
def plot():

    params = request.json #.form.to_dict(flat=True)

    print('-----------')
    print('plot():', params)

    if len(sources) == 0:
        return 'no data'
        
    key = list(sources.keys())[0]
    d = sources[key]['data']
    print(d)
    
    f1 = figure(title='Plot', plot_height=400, plot_width=400, sizing_mode='scale_both') #, y_axis_type="log")
    
    r0 = f1.line(d.xd[d.minD+1:], d.dfatalities, line_width=1, line_color='red', line_dash='dotted', alpha=0.3)
    r1 = f1.circle(d.xd[d.minD+1:], d.dfatalities, size=5, color="red", alpha=0.3)
    
    legend = [
        ("COVID fatalities"   , [r0, r1]),
    ]
    format_fig(f1, 'Number of deaths per day', legend)

    r = json.dumps(json_item(f1))
    #print(r)
    return r


    


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