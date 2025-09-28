import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar datos
df = pd.read_csv("FINAL_USO.csv")
df["Date"] = pd.to_datetime(df["Date"])

# 2. Preparar modelo simple para predicción
X = df[["USDI_Price", "SP_close", "USO_Close"]]
y = df["Adj Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 3. Inicializar app
app = dash.Dash(__name__)

# 4. Layout del dashboard
app.layout = html.Div([
    html.H1("Dashboard - Análisis del Oro vs Indicadores", style={"textAlign": "center"}),

    html.Div([
        html.P(f"Error cuadrático medio (MSE): {mse:.2f}"),
        html.P(f"Coeficiente de determinación (R²): {r2:.3f}")
    ], style={"marginBottom": "20px"}),

    dcc.Dropdown(
        id="var-dropdown",
        options=[{"label": col, "value": col} for col in ["USDI_Price", "SP_close", "USO_Close"]],
        value="USDI_Price",
        clearable=False
    ),

    dcc.Graph(id="scatter-plot"),

    dcc.Graph(
        id="time-series",
        figure=px.line(df, x="Date", y="Adj Close", title="Serie temporal del Oro (Adj Close)")
    )
])

# 5. Callback para gráfico interactivo
@app.callback(
    dash.dependencies.Output("scatter-plot", "figure"),
    [dash.dependencies.Input("var-dropdown", "value")]
)
def update_graph(variable):
    fig = px.scatter(df, x=variable, y="Adj Close",
                     trendline="ols",
                     title=f"Oro vs {variable}")
    return fig

if __name__ == "__main__":
    app.run(debug=True)