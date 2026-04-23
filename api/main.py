from contextlib import asynccontextmanager
import pandas as pd  # FIX #5 — moved import to module level

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from modules.demand_forecast  import DemandForecaster
from modules.route_cost       import RouteCostPredictor
from modules.inventory        import InventoryManager
from modules.predictive_maint import PredictiveMaintenanceSystem
from modules.route_optimizer  import RouteOptimizer


# ── Startup / Shutdown (lifespan) ────────────────────────────────────────────
#
# FIX #1 — All five endpoints previously reinstantiated a blank, unfitted
# model object on every request, then immediately called predict() on it.
# Every endpoint would crash on the first call.
#
# Fix: load every model exactly once at application startup using FastAPI's
# recommended lifespan context manager (replaces the deprecated @app.on_event).
# Models are stored on app.state and injected into each endpoint.
#
# FIX #10 — lifespan replaces the deprecated @app.on_event("startup") pattern.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────────────────────────
    print('Loading models...')

    # Demand Forecaster
    forecaster = DemandForecaster()
    forecaster.load_model()                          # loads models/demand_xgb.pkl
    app.state.demand_df  = pd.read_csv('data/demand_data.csv')
    app.state.forecaster = forecaster

    # Route Cost Predictor
    route_predictor = RouteCostPredictor()
    route_predictor.load_model()                     # loads lgbm + vehicle encoder
    app.state.route_predictor = route_predictor

    # Inventory Manager (stateless — no model file needed)
    app.state.inventory_manager = InventoryManager()
    # FIX #5 — Load inventory CSV once at startup, not on every GET request
    app.state.inventory_df = pd.read_csv('data/inventory_data.csv')

    # Predictive Maintenance System
    maint_system = PredictiveMaintenanceSystem()
    maint_system.load_models()                       # loads iso_forest + LSTM + scalers
    app.state.maint_system = maint_system

    # Route Optimizer (pure solver — no trained artifact to load)
    app.state.route_optimizer = RouteOptimizer()

    print('All models loaded. API ready.')
    yield
    # ── SHUTDOWN (add cleanup here if needed) ────────────────────────────────


# ── App Initialisation ───────────────────────────────────────────────────────

app = FastAPI(
    title='LogiSense ML Platform',
    description='Unified logistics ML API — 5 modules',
    version='1.0.0',
    lifespan=lifespan,
)

# FIX #9 — CORS middleware for Streamlit dashboard (runs on a different port).
# Without this every browser request from the dashboard is blocked by the
# browser's same-origin policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],      # tighten to specific origins in production
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    sku_id: str
    horizon_days: int = 30

class RouteCostRequest(BaseModel):
    distance_km: float
    load_weight_kg: float
    num_stops: int
    fuel_price: float
    # FIX #3 — Changed vehicle_type_encoded: int → vehicle_type: str.
    # Our fixed route_cost.py encodes vehicle_type internally via LabelEncoder.
    # Sending a pre-encoded int now causes LabelEncoder.transform() to fail.
    vehicle_type: str
    departure_hour: int
    driver_overtime_hrs: float = 0.0
    fuel_consumption_per_km: float = 0.12
    vehicle_capacity_kg: float = 1000.0
    base_cost: float = 500.0

class SensorReading(BaseModel):
    vehicle_id: str
    temperature: float
    vibration: float
    pressure: float
    rpm: float
    oil_level: float

class RouteOptRequest(BaseModel):
    locations: List[List[float]]   # [[lat, lon], ...]
    num_vehicles: int = 3
    max_distance_km: int = 500
    demands: Optional[List[int]] = None          # per-location demand (0 for depot)
    vehicle_capacity: Optional[int] = None       # max load per vehicle


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get('/')
def root():
    return {'message': 'LogiSense ML Platform is running', 'modules': 5}

# FIX #8 — Added /health endpoint for Docker health checks and AWS ALB target
# group health probes. Without this the container orchestrator has no way to
# confirm the service is alive after startup.
@app.get('/health')
def health():
    return {'status': 'healthy'}


@app.post('/forecast/demand')
def forecast_demand(req: ForecastRequest):
    # FIX #2 — predict_for_sku() doesn't exist on DemandForecaster.
    # The actual method is predict(df). Filter the pre-loaded demand DataFrame
    # by sku_id and pass the result to predict().
    try:
        forecaster = app.state.forecaster
        sku_df = app.state.demand_df[app.state.demand_df['sku_id'] == req.sku_id]
        if sku_df.empty:
            raise HTTPException(status_code=404, detail=f"SKU '{req.sku_id}' not found.")
        result = forecaster.predict(sku_df, horizon=req.horizon_days)
        return {'sku_id': req.sku_id, 'horizon_days': req.horizon_days, **result}
    except HTTPException:
        raise
    except Exception as e:
        # FIX #4 — All endpoints now catch exceptions and return structured
        # HTTPException instead of raw Python tracebacks.
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict/route-cost')
def predict_route_cost(req: RouteCostRequest):
    try:
        # FIX #6 — .dict() is deprecated in Pydantic v2; use .model_dump().
        cost = app.state.route_predictor.predict(req.model_dump())
        return {'predicted_cost_inr': round(cost, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/monitor/sensor')
def monitor_sensor(reading: SensorReading):
    try:
        # Pass only the sensor fields — vehicle_id is metadata, not a sensor input.
        sensor_data = reading.model_dump(exclude={'vehicle_id'})
        result = app.state.maint_system.detect_anomaly(sensor_data)
        alert  = 'MAINTENANCE ALERT' if result['anomaly'] else 'NORMAL'
        return {'vehicle_id': reading.vehicle_id, 'status': alert, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/optimise/routes')
def optimise_routes(req: RouteOptRequest):
    # FIX #7 — num_vehicles is now an __init__ param on our fixed RouteOptimizer,
    # not a solve() param. Instantiate with the request's values.
    # RouteOptimizer is a pure solver (no fitted artifact) so per-request
    # instantiation is acceptable here.
    try:
        optimizer = RouteOptimizer(
            num_vehicles=req.num_vehicles,
            max_distance_km=req.max_distance_km,
        )
        result = optimizer.solve(
            locations=req.locations,
            demands=req.demands,
            vehicle_capacity=req.vehicle_capacity,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/inventory/signals')
def get_inventory_signals():
    # FIX #5 — CSV is pre-loaded at startup; this endpoint now just runs the
    # vectorised signal generation on the already-loaded DataFrame.
    try:
        signals = app.state.inventory_manager.generate_replenishment_signals(
            app.state.inventory_df
        )
        return signals.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)