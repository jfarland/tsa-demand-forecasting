from h2o_wave import main, app, Q, ui, on, handle_on, data
from typing import Optional, List
import pandas as pd
import plotly.express as px
from plotly import io as pio

import datatable as dt


#Start data manipulations and calculations for Page 2 here:

avsp=pd.read_csv('./data/prepared-data/flclt_tsa_dash.csv',
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
df_fore_m=pd.read_csv('./data/prepared-data/'
                    'h2oai_experiment_flclt_tsa_monthly_train_dataset_FLCLT_tsa_train_pred.csv',
                 parse_dates=['Date', 'date_time'])

df_fore_w=pd.read_csv('./data/prepared-data/'
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

def make_markdown_row(values):
    return f"| {' | '.join([str(x) for x in values])} |"

def make_markdown_table(fields, rows):
    return '\n'.join([
        make_markdown_row(fields),
        make_markdown_row('-' * len(fields)),
        '\n'.join([make_markdown_row(row) for row in rows]),
    ])

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

    for i in range(3):
        add_card(q, f'info{i}', ui.tall_info_card(box='horizontal', name='', title='Speed',
                                                  caption='The models are performant thanks to...', icon='SpeedHigh'))
    add_card(q, 'article', ui.tall_article_preview_card(
        box=ui.box('vertical', height='600px'), title='How does magic work',
        #image='https://images.pexels.com/photos/624015/pexels-photo-624015.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
        image='https://images.pexels.com/photos/4671912/pexels-photo-4671912.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750',
        content='''
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum ac sodales felis. Duis orci enim, iaculis at augue vel, mattis imperdiet ligula. Sed a placerat lacus, vitae viverra ante. Duis laoreet purus sit amet orci lacinia, non facilisis ipsum venenatis. Duis bibendum malesuada urna. Praesent vehicula tempor volutpat. In sem augue, blandit a tempus sit amet, tristique vehicula nisl. Duis molestie vel nisl a blandit. Nunc mollis ullamcorper elementum.
Donec in erat augue. Nullam mollis ligula nec massa semper, laoreet pellentesque nulla ullamcorper. In ante ex, tristique et mollis id, facilisis non metus. Aliquam neque eros, semper id finibus eu, pellentesque ac magna. Aliquam convallis eros ut erat mollis, sit amet scelerisque ex pretium. Nulla sodales lacus a tellus molestie blandit. Praesent molestie elit viverra, congue purus vel, cursus sem. Donec malesuada libero ut nulla bibendum, in condimentum massa pretium. Aliquam erat volutpat. Interdum et malesuada fames ac ante ipsum primis in faucibus. Integer vel tincidunt purus, congue suscipit neque. Fusce eget lacus nibh. Sed vestibulum neque id erat accumsan, a faucibus leo malesuada. Curabitur varius ligula a velit aliquet tincidunt. Donec vehicula ligula sit amet nunc tempus, non fermentum odio rhoncus.
Vestibulum condimentum consectetur aliquet. Phasellus mollis at nulla vel blandit. Praesent at ligula nulla. Curabitur enim tellus, congue id tempor at, malesuada sed augue. Nulla in justo in libero condimentum euismod. Integer aliquet, velit id convallis maximus, nisl dui porta velit, et pellentesque ligula lorem non nunc. Sed tincidunt purus non elit ultrices egestas quis eu mauris. Sed molestie vulputate enim, a vehicula nibh pulvinar sit amet. Nullam auctor sapien est, et aliquet dui congue ornare. Donec pulvinar scelerisque justo, nec scelerisque velit maximus eget. Ut ac lectus velit. Pellentesque bibendum ex sit amet cursus commodo. Fusce congue metus at elementum ultricies. Suspendisse non rhoncus risus. In hac habitasse platea dictumst.
        '''
    ))

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

    # add_card(q, 'table', ui.form_card(box='vertical',
    #                                   items=[ui.text(make_markdown_table(
    #                                       fields=var_list,
    #                                       rows=val_list
    #                                   ))],
    #                                   ))
    add_card(q, 'table', ui.form_card(
                box=ui.box('data_table'),
                items = [
                    table_from_df(q.client.df.to_pandas(), name='agg_data', 
                        downloadable=True, 
                        sortables=['Month'],
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

    for i in range(12):
        add_card(q, f'item{i}', ui.wide_info_card(box=ui.box('grid', width='400px'), name='', title='Tile',
                                                  caption='Lorem ipsum dolor sit amet'))

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
    ])],
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
            ui.persona(title='Jane Doe', subtitle='Developer', size='xs',
                       image='https://images.pexels.com/photos/1181424/pexels-photo-1181424.jpeg?auto=compress&h=750&w=1260'),
        ]
    )

    q.client.df = dt.fread('./data/prepared-data/Flclt_tsa.csv')


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