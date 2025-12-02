from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Carrega o modelo na memória quando a API sobe
modelo = joblib.load("modelo_combustivel.joblib")

app = FastAPI(
    title="API de Previsão de Consumo de Combustível",
    description="Recebe características do veículo e retorna o consumo médio em L/100km.",
    version="1.0.0"
)

# Modelo de entrada (JSON do Flutter)
class FuelRequest(BaseModel):
    enginesize: float      # tamanho do motor (ex: 2.0)
    cylinders: int         # número de cilindros
    vehicleclass: str      # classe do veículo, ex: "SUV", "COMPACT"
    transmission: str      # ex: "A6", "M5"
    fueltype: str          # ex: "Z", "X", "D", "E"
    distance_km: float | None = None  # opcional: distância da viagem


# Modelo de saída (JSON de resposta)
class FuelResponse(BaseModel):
    consumo_l_100km: float          # previsão do dataset
    consumo_litros_viagem: float | None = None  # se distance_km foi enviado
    km_por_litro: float | None = None          # conversão opcional


@app.get("/")
def read_root():
    return {"message": "API de previsão de consumo de combustível está no ar 🚗⛽"}


@app.post("/predict", response_model=FuelResponse)
def predict_consumption(request: FuelRequest):
    # Monta um DataFrame com as colunas esperadas pelo modelo
    df = pd.DataFrame([{
        "ENGINESIZE": request.enginesize,
        "CYLINDERS": request.cylinders,
        "VEHICLECLASS": request.vehicleclass,
        "TRANSMISSION": request.transmission,
        "FUELTYPE": request.fueltype
    }])

    # Faz a previsão em L/100km
    pred_l_100km = float(modelo.predict(df)[0])

    # Calcula informações extras, se distance_km for informado
    consumo_viagem = None
    km_por_litro = None

    if request.distance_km is not None:
        consumo_viagem = (request.distance_km * pred_l_100km) / 100.0
        # km por litro é o inverso: 100 km / (L/100km)
        if pred_l_100km > 0:
            km_por_litro = 100.0 / pred_l_100km

    return FuelResponse(
        consumo_l_100km=pred_l_100km,
        consumo_litros_viagem=consumo_viagem,
        km_por_litro=km_por_litro
    )
