# app/taipy_app.py  (FIXED - Flask + Plotly Dash version)
# Works perfectly on Windows — no socket/gevent issues
# Run: python app/taipy_app.py
# Open: http://localhost:8050

import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objects as go

# ── Load data ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

predictions = pd.read_csv(os.path.join(BASE, "data/outputs/predictions.csv"),    parse_dates=['date'])
inventory   = pd.read_csv(os.path.join(BASE, "data/outputs/inventory_recommendations.csv"))
cleaned     = pd.read_csv(os.path.join(BASE, "data/processed/cleaned_data.csv"), parse_dates=['date'])

product_list = sorted(predictions['product'].unique().tolist())
store_list   = sorted(predictions['store'].unique().tolist())

SERVICE_LEVEL_Z = {0.90: 1.28, 0.92: 1.41, 0.95: 1.65, 0.97: 1.88, 0.98: 2.05, 0.99: 2.33}

# ── KPI helpers ────────────────────────────────────────────────────────────────
total_revenue  = f"\u20b9{cleaned['revenue'].sum()/1e6:.1f}M"
total_products = predictions['product'].nunique()
critical_items = len(inventory[inventory['urgency_flag'] == 'CRITICAL'])
avg_mape       = abs((predictions['units_sold'] - predictions['predicted_sales']) /
                     (predictions['units_sold'] + 1)).mean() * 100


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS  — must be defined BEFORE the layout uses them
# ══════════════════════════════════════════════════════════════════════════════

def _card_style():
    return {
        'background':   '#161926',
        'borderRadius': '12px',
        'padding':      '24px',
        'border':       '1px solid #2a2d3e',
    }

def _label_style():
    return {
        'fontSize':      '12px',
        'color':         '#7c8db5',
        'fontWeight':    '600',
        'textTransform': 'uppercase',
        'letterSpacing': '0.8px',
        'marginBottom':  '8px',
        'display':       'block',
    }

def _kpi_card(title, value, sub, accent):
    return html.Div(style={
        'background':    '#161926',
        'borderRadius':  '12px',
        'padding':       '20px 24px',
        'border':        '1px solid #2a2d3e',
        'borderLeft':    f'3px solid {accent}',
    }, children=[
        html.P(title, style={
            'margin': '0 0 4px', 'fontSize': '11px',
            'color': '#7c8db5', 'fontWeight': '600',
            'textTransform': 'uppercase', 'letterSpacing': '0.8px',
        }),
        html.H2(value, style={
            'margin': '0 0 4px', 'fontSize': '28px',
            'color': accent, 'fontWeight': '700',
        }),
        html.P(sub, style={'margin': 0, 'fontSize': '12px', 'color': '#5a6480'}),
    ])

def _control_card(label, component):
    return html.Div(style=_card_style(), children=[
        html.Label(label, style=_label_style()),
        component,
    ])

def _metric_row(label, value, color):
    return html.Div(style={
        'display':        'flex',
        'justifyContent': 'space-between',
        'alignItems':     'center',
        'padding':        '6px 0',
        'borderBottom':   '1px solid #1e2130',
    }, children=[
        html.Span(label, style={'fontSize': '12px', 'color': '#7c8db5'}),
        html.Span(value, style={'fontSize': '13px', 'color': color, 'fontWeight': '700'}),
    ])

# Shared Plotly dark theme applied to every chart
DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c8cdd8', family='"Segoe UI", sans-serif', size=12),
    margin=dict(l=40, r=20, t=20, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#2a2d3e', borderwidth=1),
    xaxis=dict(gridcolor='#1e2130', linecolor='#2a2d3e', tickcolor='#5a6480'),
    yaxis=dict(gridcolor='#1e2130', linecolor='#2a2d3e', tickcolor='#5a6480'),
)


# ══════════════════════════════════════════════════════════════════════════════
#  APP + LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app = Dash(__name__, title="Retail Forecasting Dashboard")

# ── Fix slider tooltip styles ──────────────────────────────────────────────────
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Slider tooltip box */
            .rc-slider-tooltip-inner {
                background-color: #1e2130 !important;
                color: #60a5fa !important;
                border: 1px solid #2a2d3e !important;
                font-weight: 700 !important;
                font-size: 12px !important;
                padding: 4px 10px !important;
                border-radius: 6px !important;
                box-shadow: none !important;
            }
            /* Tooltip arrow */
            .rc-slider-tooltip-placement-bottom .rc-slider-tooltip-arrow {
                border-bottom-color: #2a2d3e !important;
            }
            /* Slider track */
            .rc-slider-track {
                background-color: #60a5fa !important;
            }
            /* Slider handle */
            .rc-slider-handle {
                border-color: #60a5fa !important;
                background-color: #60a5fa !important;
            }
            .rc-slider-handle:hover {
                border-color: #93c5fd !important;
            }
            /* Slider rail */
            .rc-slider-rail {
                background-color: #2a2d3e !important;
            }
            /* Scrollbar */
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: #0f1117; }
            ::-webkit-scrollbar-thumb { background: #2a2d3e; border-radius: 3px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div(style={
    'fontFamily': '"Segoe UI", sans-serif',
    'background':  '#0f1117',
    'minHeight':   '100vh',
    'color':       '#e8eaf0',
    'padding':     '0',
}, children=[

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div(style={
        'background':    'linear-gradient(135deg, #1a1d2e 0%, #16213e 100%)',
        'borderBottom':  '1px solid #2a2d3e',
        'padding':       '24px 40px',
        'display':       'flex',
        'alignItems':    'center',
        'justifyContent':'space-between',
    }, children=[
        html.Div([
            html.H1("\U0001f6d2 Retail Intelligence Dashboard", style={
                'margin': 0, 'fontSize': '24px', 'fontWeight': '700',
                'color': '#ffffff', 'letterSpacing': '-0.5px',
            }),
            html.P("Sales Forecasting & Inventory Optimization System", style={
                'margin': '4px 0 0', 'fontSize': '13px', 'color': '#7c8db5',
            }),
        ]),
        html.Span("\u25cf LIVE", style={
            'color': '#4ade80', 'fontSize': '12px',
            'fontWeight': '600', 'letterSpacing': '1px',
        }),
    ]),

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    html.Div(style={
        'display':              'grid',
        'gridTemplateColumns':  'repeat(4, 1fr)',
        'gap':                  '16px',
        'padding':              '24px 40px 0',
    }, children=[
        _kpi_card("Total Revenue",       total_revenue,          "3-year simulated sales", "#4ade80"),
        _kpi_card("Products Tracked",    str(total_products),    "across 3 stores",        "#60a5fa"),
        _kpi_card("Critical Alerts",     str(critical_items),    "need immediate reorder", "#f87171"),
        _kpi_card("Forecast Accuracy",   f"{100-avg_mape:.1f}%", f"MAPE: {avg_mape:.1f}%","#facc15"),
    ]),

    # ── Dropdowns ─────────────────────────────────────────────────────────────
    html.Div(style={
        'display':             'grid',
        'gridTemplateColumns': '1fr 1fr',
        'gap':                 '16px',
        'padding':             '24px 40px 0',
    }, children=[
        _control_card("Product", dcc.Dropdown(
            id='product-dd',
            options=[{'label': p, 'value': p} for p in product_list],
            value=product_list[0],
            clearable=False,
            style={'background': '#1e2130', 'color': '#000'},
        )),
        _control_card("Store", dcc.Dropdown(
            id='store-dd',
            options=[{'label': s, 'value': s} for s in store_list],
            value=store_list[0],
            clearable=False,
            style={'background': '#1e2130', 'color': '#000'},
        )),
    ]),

    # ── Forecast Chart ────────────────────────────────────────────────────────
    html.Div(style={'padding': '24px 40px 0'}, children=[
        html.Div(style=_card_style(), children=[
            html.H3("Sales Forecast — Actual vs Predicted", style={
                'margin': '0 0 16px', 'fontSize': '15px',
                'color': '#e8eaf0', 'fontWeight': '600',
            }),
            dcc.Graph(id='forecast-chart',
                      config={'displayModeBar': False},
                      style={'height': '320px'}),
        ]),
    ]),

    # ── What-If Sliders + Inventory Chart ─────────────────────────────────────
    html.Div(style={
        'display':             'grid',
        'gridTemplateColumns': '320px 1fr',
        'gap':                 '16px',
        'padding':             '24px 40px 0',
    }, children=[

        # Sliders panel
        html.Div(style=_card_style(), children=[
            html.H3("\u2699 What-If Simulation", style={
                'margin': '0 0 20px', 'fontSize': '15px',
                'color': '#e8eaf0', 'fontWeight': '600',
            }),

            html.Label("Service Level", style=_label_style()),
            dcc.Slider(
                id='sl-slider', min=0.90, max=0.99, step=0.01, value=0.95,
                marks={
                    0.90: {'label': '90%', 'style': {'color': '#7c8db5'}},
                    0.95: {'label': '95%', 'style': {'color': '#60a5fa'}},
                    0.99: {'label': '99%', 'style': {'color': '#7c8db5'}},
                },
                tooltip={'placement': 'bottom', 'always_visible': False},
            ),
            html.Div(id='sl-display', style={
                'color': '#60a5fa', 'fontSize': '13px',
                'margin': '8px 0 20px', 'textAlign': 'center',
            }),

            html.Label("Lead Time (days)", style=_label_style()),
            dcc.Slider(
                id='lt-slider', min=1, max=14, step=1, value=5,
                marks={
                    1:  {'label': '1d',  'style': {'color': '#7c8db5'}},
                    7:  {'label': '7d',  'style': {'color': '#60a5fa'}},
                    14: {'label': '14d', 'style': {'color': '#7c8db5'}},
                },
                tooltip={'placement': 'bottom', 'always_visible': False},
            ),
            html.Div(id='lt-display', style={
                'color': '#60a5fa', 'fontSize': '13px',
                'margin': '8px 0 20px', 'textAlign': 'center',
            }),

            html.Div(id='metrics-box', style={
                'background':   '#0f1117',
                'borderRadius': '8px',
                'padding':      '16px',
                'marginTop':    '8px',
            }),
        ]),

        # Inventory simulation chart
        html.Div(style=_card_style(), children=[
            html.H3("Inventory Stock Simulation", style={
                'margin': '0 0 16px', 'fontSize': '15px',
                'color': '#e8eaf0', 'fontWeight': '600',
            }),
            dcc.Graph(id='inv-chart',
                      config={'displayModeBar': False},
                      style={'height': '340px'}),
        ]),
    ]),

    # ── Category + Store Charts ───────────────────────────────────────────────
    html.Div(style={
        'display':             'grid',
        'gridTemplateColumns': '1fr 1fr',
        'gap':                 '16px',
        'padding':             '24px 40px 0',
    }, children=[
        html.Div(style=_card_style(), children=[
            html.H3("Revenue by Category", style={
                'margin': '0 0 16px', 'fontSize': '15px',
                'color': '#e8eaf0', 'fontWeight': '600',
            }),
            dcc.Graph(id='cat-chart',
                      config={'displayModeBar': False},
                      style={'height': '280px'}),
        ]),
        html.Div(style=_card_style(), children=[
            html.H3("Monthly Sales by Store", style={
                'margin': '0 0 16px', 'fontSize': '15px',
                'color': '#e8eaf0', 'fontWeight': '600',
            }),
            dcc.Graph(id='store-chart',
                      config={'displayModeBar': False},
                      style={'height': '280px'}),
        ]),
    ]),

    # ── Reorder Table ─────────────────────────────────────────────────────────
    html.Div(style={'padding': '24px 40px 40px'}, children=[
        html.Div(style=_card_style(), children=[
            html.H3("\U0001f6a8 Inventory Reorder Recommendations", style={
                'margin': '0 0 16px', 'fontSize': '15px',
                'color': '#e8eaf0', 'fontWeight': '600',
            }),
            dash_table.DataTable(
                id='inv-table',
                columns=[
                    {'name': 'Product',          'id': 'product'},
                    {'name': 'Store',            'id': 'store'},
                    {'name': 'Avg Daily Demand', 'id': 'avg_daily_demand'},
                    {'name': 'Safety Stock',     'id': 'safety_stock_units'},
                    {'name': 'Reorder Point',    'id': 'reorder_point_units'},
                    {'name': 'Current Stock',    'id': 'current_stock'},
                    {'name': 'Days Left',        'id': 'days_of_stock_left'},
                    {'name': 'EOQ Units',        'id': 'eoq_units'},
                    {'name': 'Status',           'id': 'urgency_flag'},
                ],
                data=inventory.to_dict('records'),
                page_size=12,
                sort_action='native',
                filter_action='native',
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': '#1e2130',
                    'color':           '#7c8db5',
                    'fontWeight':      '600',
                    'fontSize':        '12px',
                    'border':          '1px solid #2a2d3e',
                    'textTransform':   'uppercase',
                    'letterSpacing':   '0.5px',
                },
                style_cell={
                    'backgroundColor': '#161926',
                    'color':           '#c8cdd8',
                    'border':          '1px solid #1e2130',
                    'fontSize':        '13px',
                    'padding':         '10px 14px',
                    'fontFamily':      '"Segoe UI", sans-serif',
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{urgency_flag} = "CRITICAL"'},
                        'backgroundColor': '#2d1515',
                        'color':           '#f87171',
                        'fontWeight':      '700',
                    },
                    {
                        'if': {'filter_query': '{urgency_flag} = "REORDER NOW"'},
                        'backgroundColor': '#2d2015',
                        'color':           '#facc15',
                    },
                    {
                        'if': {'filter_query': '{urgency_flag} = "OK"'},
                        'color': '#4ade80',
                    },
                ],
            ),
        ]),
    ]),
])


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output('forecast-chart', 'figure'),
    Input('product-dd', 'value'),
    Input('store-dd',   'value'),
)
def update_forecast(product, store):
    df = predictions[
        (predictions['product'] == product) &
        (predictions['store']   == store)
    ].sort_values('date')

    fig = go.Figure()

    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['units_sold'],
            name='Actual',
            line=dict(color='#60a5fa', width=1.5),
            mode='lines',
        ))
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['predicted_sales'],
            name='Forecast',
            line=dict(color='#f87171', width=1.5, dash='dot'),
            mode='lines',
        ))
        # Confidence band (fill between upper and lower bound)
        upper = df['predicted_sales'] * 1.10
        lower = df['predicted_sales'] * 0.90
        fig.add_trace(go.Scatter(
            x=pd.concat([df['date'], df['date'][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(248,113,113,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name='\u00b110% band',
            showlegend=True,
        ))

    fig.update_layout(**DARK_LAYOUT)
    return fig


@app.callback(
    Output('inv-chart',   'figure'),
    Output('metrics-box', 'children'),
    Output('sl-display',  'children'),
    Output('lt-display',  'children'),
    Input('product-dd',   'value'),
    Input('store-dd',     'value'),
    Input('sl-slider',    'value'),
    Input('lt-slider',    'value'),
)
def update_inventory(product, store, sl, lt):
    Z = SERVICE_LEVEL_Z.get(round(sl, 2), 1.65)

    prod_preds = predictions[
        (predictions['product'] == product) &
        (predictions['store']   == store)
    ]

    daily_mean = float(prod_preds['predicted_sales'].mean()) if not prod_preds.empty else 50.0
    daily_std  = float(prod_preds['predicted_sales'].std())  if not prod_preds.empty else 10.0

    safety_stock  = max(0, round(Z * daily_std * np.sqrt(lt)))
    reorder_point = round(daily_mean * lt + safety_stock)
    starting_stock = reorder_point + safety_stock + daily_mean * 5

    np.random.seed(42)
    days  = np.arange(45)
    stock = [starting_stock]
    for _ in days[1:]:
        nxt = stock[-1] - daily_mean + np.random.normal(0, daily_std * 0.25)
        stock.append(max(0.0, nxt))
    stock = np.array(stock)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=stock,
        name='Stock Level',
        line=dict(color='#4ade80', width=2),
        fill='tozeroy',
        fillcolor='rgba(74,222,128,0.06)',
        mode='lines',
    ))
    fig.add_hline(
        y=reorder_point,
        line=dict(color='#facc15', width=1.5, dash='dash'),
        annotation_text=f'Reorder Point: {reorder_point}u',
        annotation_font=dict(color='#facc15', size=11),
    )
    fig.add_hline(
        y=safety_stock,
        line=dict(color='#f87171', width=1.5, dash='dot'),
        annotation_text=f'Safety Stock: {safety_stock}u',
        annotation_font=dict(color='#f87171', size=11),
    )
    fig.add_hrect(y0=0, y1=safety_stock,
                  fillcolor='rgba(248,113,113,0.05)', line_width=0)

    layout = dict(DARK_LAYOUT)
    layout['xaxis'] = dict(DARK_LAYOUT['xaxis'], title='Days')
    layout['yaxis'] = dict(DARK_LAYOUT['yaxis'], title='Units')
    layout['margin'] = dict(l=40, r=160, t=20, b=40)
    fig.update_layout(**layout)

    metrics = html.Div([
        _metric_row("Safety Stock",   f"{safety_stock} units",   "#f87171"),
        _metric_row("Reorder Point",  f"{reorder_point} units",  "#facc15"),
        _metric_row("Daily Demand",   f"{daily_mean:.1f} units", "#60a5fa"),
        _metric_row("Demand Std Dev", f"{daily_std:.1f} units",  "#a78bfa"),
        _metric_row("Z-Score",        f"{Z}",                    "#4ade80"),
    ])

    return fig, metrics, f"Service Level: {sl*100:.0f}%", f"Lead Time: {lt} days"


@app.callback(
    Output('cat-chart', 'figure'),
    Input('product-dd', 'value'),
)
def update_category(_):
    cat_rev = cleaned.groupby('category')['revenue'].sum().sort_values()
    colors  = ['#60a5fa', '#4ade80', '#facc15', '#f87171', '#a78bfa']

    fig = go.Figure(go.Bar(
        x=cat_rev.values,
        y=cat_rev.index,
        orientation='h',
        marker=dict(color=colors[:len(cat_rev)]),
        text=[f'\u20b9{v/1e6:.1f}M' for v in cat_rev.values],
        textposition='outside',
        textfont=dict(color='#c8cdd8', size=11),
    ))
    fig.update_layout(**DARK_LAYOUT)
    return fig


@app.callback(
    Output('store-chart', 'figure'),
    Input('store-dd', 'value'),
)
def update_store(_):
    store_monthly = (
        cleaned.groupby(['month', 'store'])['units_sold']
        .sum()
        .reset_index()
    )
    palette = {'Store_A': '#60a5fa', 'Store_B': '#4ade80', 'Store_C': '#facc15'}

    fig = go.Figure()
    for s in store_list:
        d = store_monthly[store_monthly['store'] == s]
        fig.add_trace(go.Scatter(
            x=d['month'], y=d['units_sold'],
            name=s,
            line=dict(color=palette.get(s, '#a78bfa'), width=2),
            mode='lines+markers',
            marker=dict(size=5),
        ))

    layout = dict(DARK_LAYOUT)
    layout['xaxis'] = dict(
        DARK_LAYOUT['xaxis'],
        tickvals=list(range(1, 13)),
        ticktext=['J','F','M','A','M','J','J','A','S','O','N','D'],
    )
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  RETAIL INTELLIGENCE DASHBOARD")
    print("  Open your browser at: http://localhost:8050")
    print("="*55 + "\n")
    app.run(debug=False, host="127.0.0.1", port=8050)