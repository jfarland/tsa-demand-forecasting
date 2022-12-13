from h2o_wave import main, app, Q, ui, on, handle_on, data
from typing import Optional, List
import pandas as pd
import plotly.express as px
from plotly import io as pio

import datatable as dt
import requests
import json

model_dict = {
    'one-week':'https://model.internal.dedicated.h2o.ai/806c196d-db50-4068-b1ab-71f07b3c00bb/model/score',
    'two-week':'https://model.internal.dedicated.h2o.ai/101ff2f1-2e2b-4ab9-b766-adfa1588d08e/model/score'
}


#Start data manipulations and calculations for Page 2 here:

avsp=pd.read_csv('data/prepared-data/Flclt_tsa_dash.csv',
                 parse_dates=['Date', 'date_time'])
avsp['Month']=avsp['Date'].dt.to_period('M')
avsp_sort = avsp.sort_values(by=['Month', 'Airport'])
avsp_sort['Month']=avsp_sort["Month"].astype(str)
mytup = avsp_sort[['ID', "Airport", 'Month','Total']]
mydict = mytup.set_index('ID').T.to_dict('list')
mydict=dict(sorted(mydict.items(), key=lambda item: item[0]))
avsp_mnth=avsp.groupby(['Airport','Month'])[['Airport','Total','Flights','Booths','Month','Max_wait_total','0-15','16-30','31-45',
                                   '46-60']].agg({'Airport': 'first', 'Month': 'first', 'Booths': 'median',
                                                  'Max_wait_total': 'median', 'Flights': 'sum', 'Total': 'sum',
                                                  '0-15': 'sum', '16-30': 'sum', '31-45': 'sum', '46-60': 'sum'})
avsp_mnth['Month'] = avsp_mnth['Month'].astype(str)
avsp_mnth['Month'] = pd.to_datetime(avsp_mnth['Month'])
pl_fig = px.line(avsp_mnth, x='Month', y='Total', color='Airport')
p_mt = pio.to_html(pl_fig, validate=False, include_plotlyjs='cdn',
                  # config=confi
       )
val_list=avsp_mnth.values.tolist()
var_list=avsp_mnth.columns.tolist()
df_fore_m=pd.read_csv('data/prepared-data/'
                    'h2oai_experiment_flclt_tsa_monthly_train_dataset_FLCLT_tsa_train_pred.csv',
                 parse_dates=['Date', 'date_time'])

df_fore_w=pd.read_csv('data/prepared-data/'
                    'h2oai_experiment_nokabaki_train_dataset_FLCLT_tsa_train_predictions.csv',
                 parse_dates=['Date', 'date_time'])
column_days=df_fore_w['Date']


# Start forecast data manipulation
actual=["Actual"]
predict=["Predicted"]

# Predicted Data sets (bi-weekly and weekly)
pred_df_m=df_fore_m[df_fore_m['type'].isin(predict)]
pred_df_w=df_fore_w[df_fore_w['type'].isin(predict)]

# Actual Data sets (bi-weekly and weekly)
df_real_m=df_fore_m[df_fore_m['type'].isin(actual)]
df_real_w=df_fore_w[df_fore_w['type'].isin(actual)]

# Use for page cards that should be removed when navigating away.
# For pages that should be always present on screen use q.page[key] = ...
def add_card(q, name, card) -> None:
    q.client.cards.add(name)
    q.page[name] = card

# Remove all the cards related to navigation.
def clear_cards(q, ignore: Optional[List[str]] = []) -> None:
    if not q.client.cards:
        return
    for name in q.client.cards.copy():
        if name not in ignore:
            del q.page[name]
            q.client.cards.remove(name)

@on('#home')
async def handle_home(q: Q):
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    add_card(q, 'article', ui.tall_article_preview_card(
        box=ui.box('horizontal2', height='900px'), 
        title='Mission Readiness through Optimization',
        subtitle='Powered by Artificial Intelligence',
        name='main_button',
        #image='https://images.pexels.com/photos/624015/pexels-photo-624015.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
        #image='https://images.pexels.com/photos/4671912/pexels-photo-4671912.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750',
        image='https://i.imgur.com/IQoObId.jpg',
        content='''
This tool is designed to provide high-performance, short-range predictions of required TSA resources across a national grid of airports and terminals. 
Behind the scenes, a state of the art predictive model is trained using H2O's AI Engines. The model is then deployed and monitored 
for performance degradation within H2O's Machine Learning Operations (ML Ops)
        '''
    ))

@on('main_button')
@on('#data')
async def handle_data(q: Q):
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    add_card(q, 'chart2', ui.frame_card(
        box='lhs',
        title='Total Passengers by Airport & Month',
        content=p_mt))

    add_card(q, 'chart1', ui.plot_card(
        box='rhs',
        title='Daily Total Passengers by Month',
        data=data('Airport Month Total', rows=mydict),
        plot=ui.plot([ui.mark(type='point', x_scale='time-category', x='=Month', y='=Total', color='=Airport',
                              dodge='auto'
                              ,
                              # y_min=50
                              )])
    ))

    add_card(q, 'table', ui.form_card(
        box=ui.box('data_table'),
        items = [
            table_from_df(q.client.df_train.to_pandas(), name='agg_data', 
                searchables=['Airport'],
                downloadable=True, 
                sortables=['date_time'],
                tags={
                    'Airport': {
                        'CLT': '$blue',
                        'TPA': '$red'
                    }
                }
            )]
    ))


@on('#forecast')
async def handle_forecast(q: Q):
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta). 
    
    # number of days to lookback and visulize
    lookback = 60

    models = ['one-week', 'two-week']
    model_choices = [ui.choice(x, x) for x in models]

    items_list = [
        ui.text_l(content='Select Forecasting Horizon and Model'),
        ui.dropdown(name='model_selector', label='Model', value=str(models[0]), choices=model_choices, trigger=True),
        ui.button(name='predict', label='Predict', primary=True),
        ui.text_l(content=''),
        ui.text_l(content=''),
        ui.text_l(content=''),
        ui.text_l(content='')
    ]
    
    add_card(q, "inputs", ui.form_card(box=ui.box('inputs', height='600px'), items=items_list))

    if q.args.predict:


        df_test = q.client.df_test.to_pandas()

        print(model_dict['one-week'])
        print(model_dict[q.args.model_selector])


        # forecast
        fcst = get_mlops_preds(
            endpoint = model_dict[q.args.model_selector],
            #endpoint = model_dict['one-week'],
            df = df_test.drop(['Total'], axis=1))


        print(f'FORECAST RETRIEVED {pd.DataFrame(fcst[0])}')

        df_test = pd.concat([df_test.drop(['Total'], axis=1), pd.DataFrame(fcst[0])], axis = 1)
        df_test['Total'] = round(df_test['Total'].astype(float), 2)
        df_test = df_test.dropna()
        df_test['Timestamp'] = pd.to_datetime(df_test['date_time']).dt.date
        df_test['Period'] = 'forecast'
        df_test = df_test[['Timestamp', 'Period', 'Total']]
        df_test = df_test.groupby(['Timestamp', 'Period']).agg({'Total':'sum'})
        df_test = df_test.reset_index()
            
        print(df_test.tail())

        df_train = q.client.df_train.to_pandas()
        df_train = df_train.dropna()
        df_train['Period'] = 'historical'
        df_train['Total'] = round(df_train['Total'].astype(float), 2)
        df_train['Timestamp'] = pd.to_datetime(df_train['date_time']).dt.date
        df_train['Timestamp'] = df_train['Timestamp'].astype(str)
        df_train = df_train[['Timestamp', 'Period', 'Total']]
        df_train = df_train.groupby(['Timestamp', 'Period']).agg({'Total':'sum'})
        df_train = df_train.tail(lookback)
        df_train = df_train.reset_index()

        df_full = pd.concat([df_train, df_test], axis = 0)

        df_full['Timestamp'] = df_full['Timestamp'].astype(str)
        df_full = df_full[df_full['Total'] > 1000]
        print(df_full.dtypes)

        print(df_full.head())
        print(df_full.tail())

        # create plot data
        ts_plot_rows = [tuple(x) for x in df_full.to_numpy()]

        # Create data buffer
        ts_plot_data = data('timestamp period demand', rows = ts_plot_rows)

        # Reference: https://wave.h2o.ai/docs/examples/plot-line-groups
        
        add_card(q, 'timeseries_data_viz', ui.plot_card(
            box = ui.box('timeseries', height='600px'), 
            title = 'Time Series Visualization',
            data = ts_plot_data,
            plot = ui.plot([
                ui.mark(
                    type='path', x='=timestamp', y='=demand', color='=period', 
                    y_title="Demand", x_title='Time') #color_range=random.choice(colors))
            ])
        ))

    else:
            

        df_full = q.client.df_train.to_pandas()
        df_full['Period'] = 'historical'
        df_full['Timestamp'] = pd.to_datetime(df_full['date_time']).dt.date
        df_full['Timestamp'] = df_full['Timestamp'].astype(str)
        df_full = df_full[['Timestamp', 'Period', 'Total']]
        df_full = df_full.groupby(['Timestamp', 'Period']).agg({'Total':'sum'})
        df_full = df_full.tail(lookback)
        df_full = df_full.reset_index()

        df_full = df_full[df_full['Total'] > 1000]


        # create plot data
        ts_plot_rows = [tuple(x) for x in df_full.to_numpy()]

        # Create data buffer
        ts_plot_data = data('timestamp period demand', rows = ts_plot_rows)

        # Reference: https://wave.h2o.ai/docs/examples/plot-line-groups
        
        add_card(q, 'timeseries_data_viz', ui.plot_card(
            box = ui.box('timeseries', height='600px'), 
            title = 'Time Series Visualization',
            data = ts_plot_data,
            plot = ui.plot([
                ui.mark(
                    type='path', x='=timestamp', y='=demand', color='=period', 
                    y_title="Demand", x_title='Time') #color_range=random.choice(colors))
            ])
        ))

    # add_card(q, 'experiment_output', ui.form_card(
    #     box=ui.box('horizontal2'),
    #     items = [
    #         table_from_df(df_full, name='agg_data', 
    #             downloadable=True, 
    #             sortables=['date_time'],
    #             tags={
    #                 'Period': {
    #                     'historical': '$blue',
    #                     'forecast': '$orange'
    #                 }
    #             })]
    # ))




async def init(q: Q) -> None:
    q.page['meta'] = ui.meta_card(box='', layouts=[ui.layout(breakpoint='xs', min_height='100vh', zones=[
        ui.zone('header'),
        ui.zone('content', zones=[
            # Specify various zones and use the one that is currently needed. Empty zones are ignored.
            ui.zone('horizontal', direction=ui.ZoneDirection.ROW,
                zones=[ui.zone('lhs', size='45%'), ui.zone('rhs', size='55%')]),
            ui.zone('horizontal2', direction=ui.ZoneDirection.ROW),
            ui.zone('data_table')
        ]),
        ui.zone('forecasting', direction=ui.ZoneDirection.ROW,
                zones=[ui.zone('inputs', size='25%'), ui.zone('timeseries', size='75%')]),
        ui.zone('footer')]),
    ],
    theme='lighting')

    q.page['header'] = ui.header_card(
        box='header', title='TSA Demand Forecasting', subtitle="AI-powered Optimization and Forecasting",
        image="https://www.tsa.gov/sites/default/files/styles/news_width_300/public/tsa_insignia_rgb.jpg?itok=U15GtY-Y",
        secondary_items=[
            ui.tabs(name='tabs', value=f'#{q.args["#"]}' if q.args['#'] else '#home', link=True, items=[
                ui.tab(name='#home', label='Home'),
                ui.tab(name='#data', label='Data Dashboard'),
                ui.tab(name='#forecast', label='Predictive Analytics')
            ]),
        ],
        items=[
            ui.persona(title='Jane Doe', subtitle='TSA Agent Coordinator', size='l',
                       image='https://images.pexels.com/photos/1181424/pexels-photo-1181424.jpeg?auto=compress&h=750&w=1260'),
        ]
    )

     # Footer Card
    q.page["footer"] = ui.footer_card(
        box="footer", caption="(c) 2022 US Federal Government. All rights reserved.", 
        items = [
            ui.inline(justify="end", items = [
                ui.links(label = "About the TSA", width='200px', items = [
                    ui.link(label="Mission Statement", path='https://www.tsa.gov/about', target="_blank"),
                    ui.link(label="Strategy", path="https://www.tsa.gov/about/strategy", target="_blank"),
                    ui.link(label="Leadership", path="https://www.tsa.gov/about/tsa-leadership", target="_blank"),
                    ui.link(label="Opportunities", path="https://www.tsa.gov/about/jobs-at-tsa", target="_blank")
                ]), 
                ui.links(label = "Artificial Intelligence", width='200px', items = [
                    ui.link(label="Driverless AI", path='https://h2o.ai/platform/ai-cloud/make/h2o-driverless-ai/', target="_blank"),
                    ui.link(label="AI Operations", path="https://h2o.ai/platform/ai-cloud/operate/", target="_blank"),
                    ui.link(label="Document AI", path="https://h2o.ai/platform/ai-cloud/make/document-ai/", target="_blank")
                ])
            ])
        ]
    )

    q.client.df_train = dt.fread('data/prepared-data/tsa_demand_train.csv')
    q.client.df_test = dt.fread('data/prepared-data/tsa_demand_test.csv')


    # If no active hash present, render page1.
    if q.args['#'] is None:
        await handle_home(q)

@app('/')
async def serve(q: Q):
    # Run only once per client connection.
    if not q.client.initialized:
        q.client.cards = set()
        await init(q)
        q.client.initialized = True

    # Handle routing.
    await handle_on(q)
    await q.page.save()

### UTILITY FUNCTIONS ####

def table_from_df(
    df: pd.DataFrame,
    name: str,
    sortables: list = None,
    filterables: list = None,
    searchables: list = None,
    numerics: list = None,
    times: list = None,
    icons: dict = None,
    tags: dict = None, 
    progresses: dict = None,
    min_widths: dict = None,
    max_widths: dict = None,
    link_col: str = None,
    multiple: bool = False,
    groupable: bool = False,
    downloadable: bool = False,
    resettable: bool = False,
    height: str = None,
    checkbox_visibility: str = None
) -> ui.table:
    """
    Convert a Pandas dataframe into Wave ui.table format.
    """

    if not sortables:
        sortables = []
    if not filterables:
        filterables = []
    if not searchables:
        searchables = []
    if not numerics:
        numerics = []
    if not times:
        times = []
    if not icons:
        icons = {}
    if not tags:
        tags = {}
    if not progresses:
        progresses = {}
    if not min_widths:
        min_widths = {}
    if not max_widths:
        max_widths = {}

    cell_types = {}
    for col in icons.keys():
        cell_types[col] = ui.icon_table_cell_type(color=icons[col]['color'])
    for col in progresses.keys():
        cell_types[col] = ui.progress_table_cell_type(color=progresses[col]['color'])
    for col in tags.keys():
        cell_types[col] = ui.tag_table_cell_type(name='',tags=[ui.tag(label=str(key), color=str(value)) for key, value in tags[col].items()])

    columns = [ui.table_column(
        name=str(x),
        label=str(x),
        sortable=True if x in sortables else False,
        filterable=True if x in filterables else False,
        searchable=True if x in searchables else False,
        data_type='number' if x in numerics else ('time' if x in times else 'string'),
        cell_type=cell_types[x] if x in cell_types.keys() else None,
        min_width=min_widths[x] if x in min_widths.keys() else None,
        max_width=max_widths[x] if x in max_widths.keys() else None,
        link=True if x == link_col else False
    ) for x in df.columns.values]

    rows = [ui.table_row(name=str(i), cells=[str(cell) for cell in row]) for i, row in df.iterrows()]

    table = ui.table(
        name=name,
        columns=columns,
        rows=rows,
        multiple=multiple,
        groupable=groupable,
        downloadable=downloadable,
        resettable=resettable,
        height=height,
        checkbox_visibility=checkbox_visibility
    )

    return table

def get_mlops_preds(endpoint: str, df: pd.DataFrame, get_shap=None, apply_drift=False):
    df = df.replace(to_replace=r'"|\'', value='', regex=True)

    rows = df

    vals = rows.values.tolist()
    for i in range(len(vals)):
        vals[i] = [str(x) if not pd.isna(x) else '' for x in vals[i]]
    
    dictionary = {'fields': df.columns.tolist(), 'rows': vals}
    if get_shap == 'original':
        dictionary['requestShapleyValueType'] = 'ORIGINAL'
    elif get_shap == 'transformed':
        dictionary['requestShapleyValueType'] = 'TRANSFORMED'
    elif get_shap == 'both':
        dictionary['requestShapleyValueType'] = 'BOTH'

    response = requests.post(url=endpoint, json=dictionary)

    if response.status_code == 200:
        response =  json.loads(response.text)
    else:
        return None, response

    preds = pd.DataFrame(data=response['score'], columns=response['fields'])

    if get_shap is not None:
        data = response['featureShapleyContributions']['contributionGroups'][0]['contributions']
        fields = response['featureShapleyContributions']['features']
        explanations = pd.DataFrame(data=data, columns=fields)
    else:
        explanations = None

    return preds, explanations