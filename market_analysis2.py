import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go
from datetime import datetime, timedelta
import json

# Fonctions de calcul des indicateurs techniques
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bbands(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, sma, lower_band

def decision_achat_vente(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    last_close = data['Close'].iloc[-1]
    last_sma = data['SMA_20'].iloc[-1]
    
    if last_close > last_sma:
        explanation = "La moyenne mobile à 20 jours indique une tendance à la hausse."
        return "Recommandation : Acheter", explanation
    elif last_close < last_sma:
        explanation = "La moyenne mobile à 20 jours indique une tendance à la baisse."
        return "Recommandation : Vendre", explanation
    else:
        explanation = "La moyenne mobile à 20 jours montre une situation stable."
        return "Recommandation : Rester", explanation

# Initialisation de l'application Dash
app = Dash(__name__)

# Ajout de styles CSS personnalisés
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

# Layout de l'application Dash
app.layout = html.Div(style={'backgroundColor': '#f9f9f9', 'fontFamily': 'Arial'}, children=[
    html.Div(style={'textAlign': 'center', 'padding': '20px'}, children=[
        html.H1("Analyse du marché boursier avec IA", style={'color': '#003366'}),
        html.P("Utilisez les indicateurs techniques et les prévisions basées sur l'IA pour analyser le marché boursier.", style={'color': '#666666'})
    ]),
    html.Details([
        html.Summary('Qu\'est-ce que le Relative Strength Index (RSI) ?'),
        html.Div([
            html.P("Le Relative Strength Index (RSI) est un indicateur technique qui mesure la vitesse et le changement des mouvements de prix."),
            html.P("Il est utilisé pour identifier les conditions de surachat et de survente d'un actif financier."),
            html.P("Le RSI est calculé à l'aide de la formule : RSI = 100 - (100 / (1 + RS)), où RS est le rapport moyen des gains sur les pertes sur une période spécifique (généralement 14 jours)."),
            html.P("Un RSI supérieur à 70 est généralement interprété comme une indication que l'actif est suracheté et pourrait subir une correction à la baisse, tandis qu'un RSI inférieur à 30 indique souvent une condition de survente et pourrait signaler un potentiel de rebond.")
        ], style={'padding': '10px', 'backgroundColor': '#ffffff', 'border': '1px solid #ccc', 'borderRadius': '5px'})
    ]),
    html.Div([
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0'}, children=[
            html.Div(style={'width': '48%'}, children=[
                html.Label('Sélectionnez une action :'),
                dcc.Dropdown(
                    id='ticker',
                    options=[
                        {'label': 'Apple', 'value': 'AAPL'},
                        {'label': 'Microsoft', 'value': 'MSFT'},
                        {'label': 'Google', 'value': 'GOOGL'}
                    ],
                    value='AAPL',
                    style={'width': '100%'}
                )
            ]),
            html.Div(style={'width': '48%'}, children=[
                html.Label('Nombre d\'époques d\'entraînement'),
                dcc.Input(
                    id='training-epochs-input',
                    type='number',
                    value=10,
                    style={'width': '100%'}
                )
            ])
        ]),
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0'}, children=[
            html.Div(style={'width': '48%'}, children=[
                html.Label('Période RSI'),
                dcc.Slider(
                    id='rsi-period-slider',
                    min=5,
                    max=30,
                    step=1,
                    value=14,
                    marks={i: str(i) for i in range(5, 31)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),
            html.Div(style={'width': '48%'}, children=[
                html.Label('Période Bandes de Bollinger'),
                dcc.Slider(
                    id='bband-period-slider',
                    min=10,
                    max=50,
                    step=5,
                    value=20,
                    marks={i: str(i) for i in range(10, 51, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ]),
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0'}, children=[
            html.Div(style={'width': '100%'}, children=[
                html.Label('Nombre de neurones LSTM'),
                dcc.Slider(
                    id='lstm-neurons-slider',
                    min=20,
                    max=100,
                    step=10,
                    value=50,
                    marks={i: str(i) for i in range(20, 101, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ]),
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0'}, children=[
            dcc.Graph(id='price-graph', style={'width': '100%', 'height': '500px'})
        ]),
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0'}, children=[
            dcc.Graph(id='indicator-graph', style={'width': '100%', 'height': '500px'})
        ]),
        html.Div(id='decision-output', style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'backgroundColor': '#ffffff'}),
        html.Div(style={'marginTop': '20px'}, children=[
            html.Label('Entrez une note pour la prévision (1-10) :'),
            dcc.Input(
                id='rating-input',
                type='number',
                min=1,
                max=10,
                step=1,
                value=5,
                style={'width': '100%'}
            ),
            html.Button('Soumettre la note', id='submit-rating-button', n_clicks=0, style={'marginTop': '10px'}),
            html.Div(id='rating-output', style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'backgroundColor': '#ffffff'})
        ])
    ]),
    dcc.Interval(
        id='interval-component',
        interval=24 * 60 * 60 * 1000,  # Mise à jour toutes les 24 heures
        n_intervals=0
    ),
])

@app.callback(
    [Output('price-graph', 'figure'),
     Output('indicator-graph', 'figure'),
     Output('decision-output', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('ticker', 'value'),
     Input('rsi-period-slider', 'value'),
     Input('bband-period-slider', 'value'),
     Input('lstm-neurons-slider', 'value'),
     Input('training-epochs-input', 'value')]
)
def update_graph(n_intervals, ticker, rsi_period, bband_period, lstm_neurons, training_epochs):
    # Charger les données historiques jusqu'au jour actuel
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start_date, end=end_date)
    except Exception as e:
        return {}, {}, f"Erreur de chargement des données : {e}"

    # Calcul des indicateurs techniques
    data['RSI'] = calculate_rsi(data, window=rsi_period)
    data['Upper'], data['Middle'], data['Lower'] = calculate_bbands(data, window=bband_period)

    # Préparation des données pour LSTM
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    X = data[['Return', 'RSI', 'Upper', 'Middle', 'Lower']]
    y = data['Close']

    # Normalisation des données
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    # Division des données en ensemble d'entraînement et de test
    X_train, X_test = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):]
    y_train, y_test = y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):]

    # Reshape des données pour LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Création du modèle LSTM
    model = tf.keras.Sequential()
    model.add(layers.LSTM(units=lstm_neurons, input_shape=(1, X_train.shape[2])))
    model.add(layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entraînement du modèle
    early_stop = callbacks.EarlyStopping(monitor='loss', patience=5)
    model.fit(X_train, y_train, epochs=training_epochs, batch_size=16, callbacks=[early_stop])

    # Prédiction
    predicted_price = model.predict(X_test)
    predicted_price = predicted_price * data['Close'].std() + data['Close'].mean()

    # Création des graphiques
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Prix de clôture'))
    price_fig.add_trace(go.Scatter(x=data.index[-len(predicted_price):], y=predicted_price.flatten(), mode='lines', name='Prévision LSTM'))

    indicator_fig = go.Figure()
    indicator_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
    indicator_fig.add_trace(go.Scatter(x=data.index, y=data['Upper'], mode='lines', name='Bande supérieure'))
    indicator_fig.add_trace(go.Scatter(x=data.index, y=data['Middle'], mode='lines', name='Moyenne'))
    indicator_fig.add_trace(go.Scatter(x=data.index, y=data['Lower'], mode='lines', name='Bande inférieure'))

    # Ajout de styles
    price_fig.update_layout(
        title='Prix de clôture et prévision',
        xaxis_title='Date',
        yaxis_title='Prix',
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='#f9f9f9'
    )
    
    indicator_fig.update_layout(
        title='Indicateurs techniques',
        xaxis_title='Date',
        yaxis_title='Valeur',
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='#f9f9f9'
    )

    decision, explanation = decision_achat_vente(data)

    decision_output = html.Div([
        html.H4(decision, style={'color': '#333333'}),
        html.P(explanation, style={'color': '#666666'})
    ])

    return price_fig, indicator_fig, decision_output

@app.callback(
    Output('rating-output', 'children'),
    [Input('submit-rating-button', 'n_clicks')],
    [State('rating-input', 'value'),
     State('ticker', 'value'),
     State('training-epochs-input', 'value')]
)
def update_rating(n_clicks, rating, ticker, training_epochs):
    if n_clicks > 0:
        # Enregistrement de la note dans un fichier JSON
        rating_data = {
            'ticker': ticker,
            'rating': rating,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'training_epochs': training_epochs
        }
        try:
            with open('ratings.json', 'a') as f:
                json.dump(rating_data, f)
                f.write("\n")
            return f"Note enregistrée avec succès : {rating}/10"
        except Exception as e:
            return f"Erreur lors de l'enregistrement de la note : {e}"
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
