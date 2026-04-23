"""
dashboard/app.py — LogiSense ML Platform
Master-class Streamlit dashboard — all 5 modules.
Run: streamlit run dashboard/app.py
"""

import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="LogiSense ML Platform",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#4F6AF5"
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
DANGER  = "#EF4444"
NEUTRAL = "#64748B"

st.markdown("""
<style>
.metric-card{background:#1E293B;border-radius:12px;padding:1.2rem 1.5rem;border-left:4px solid #4F6AF5;}
.metric-card h3{margin:0;font-size:.85rem;color:#94A3B8;}
.metric-card p{margin:.3rem 0 0;font-size:2rem;font-weight:700;color:#F1F5F9;}
.alert-ok{background:#052e16;border-left:4px solid #22C55E;padding:.7rem 1rem;border-radius:8px;color:#bbf7d0;}
.alert-warn{background:#431407;border-left:4px solid #EF4444;padding:.7rem 1rem;border-radius:8px;color:#fecaca;}
div[data-testid="stTabs"] button{font-size:.95rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚚 LogiSense ML")
    st.caption("Unified Logistics Intelligence Platform")
    st.divider()
    st.subheader("🔌 API Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("API Online ✓")
        else:
            st.error(f"API Error ({r.status_code})")
    except Exception:
        st.error("API Offline\n\nRun:\n`uvicorn api.main:app --port 8000`")
    st.divider()
    st.subheader("⚙️ Connection")
    API_URL = st.text_input("API Base URL", value=API_URL, label_visibility="collapsed")
    st.divider()
    st.markdown("""
**Modules**
- 📈 Demand Forecast — XGBoost+ARIMA
- 💰 Route Cost — LightGBM
- 📦 Inventory — Safety Stock
- 🔧 Maintenance — IsoForest+LSTM
- 🗺️ Route Optimiser — OR-Tools VRP
""")
    st.caption("LogiSense v1.0 · Python 3.12 · FastAPI")

# ── Helpers ───────────────────────────────────────────────────────────────────
def api_post(endpoint, payload, timeout=30):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", f"HTTP {r.status_code}")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is it running on port 8000?"
    except Exception as e:
        return None, str(e)

def api_get(endpoint, timeout=30):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=timeout)
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", f"HTTP {r.status_code}")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is it running on port 8000?"
    except Exception as e:
        return None, str(e)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Demand Forecast",
    "💰  Route Cost",
    "📦  Inventory",
    "🔧  Maintenance",
    "🗺️  Route Optimiser",
])

# ════════════════════════════════════════════════════════════
# TAB 1 — DEMAND FORECAST
# ════════════════════════════════════════════════════════════
with tab1:
    st.header("📈 Demand Forecasting")
    st.markdown("Hybrid **ARIMA + XGBoost** · Confidence intervals · Per-SKU 30-day horizon")
    st.divider()

    col_ctrl, col_chart = st.columns([1, 3])

    with col_ctrl:
        st.subheader("Parameters")
        sku = st.text_input("SKU ID", "SKU_0001",
                            help="SKUs range from SKU_0001 to SKU_0200")
        horizon = st.slider("Horizon (days)", 7, 90, 30)
        show_ci = st.toggle("Confidence Band", value=True)
        st.markdown("**Quick select**")
        qc = st.columns(3)
        for i, q in enumerate(["SKU_0001","SKU_0050","SKU_0100"]):
            if qc[i].button(q, key=f"qs{i}", use_container_width=True):
                sku = q
        run_fc = st.button("▶ Run Forecast", type="primary", use_container_width=True)

    with col_chart:
        if run_fc:
            with st.spinner(f"Forecasting {sku} for {horizon} days ..."):
                data, err = api_post("/forecast/demand",
                                     {"sku_id": sku, "horizon_days": horizon})
            if err:
                st.error(f"❌ {err}")
            else:
                forecast = data["forecast"]
                ci       = data["confidence_interval"]
                days     = list(range(1, len(forecast)+1))
                lower    = [c[0] for c in ci]
                upper    = [c[1] for c in ci]

                fig = go.Figure()
                if show_ci:
                    fig.add_trace(go.Scatter(
                        x=days+days[::-1], y=upper+lower[::-1],
                        fill="toself", fillcolor="rgba(79,106,245,0.15)",
                        line=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip", name="95% CI",
                    ))
                fig.add_trace(go.Scatter(
                    x=days, y=forecast, mode="lines+markers",
                    name="Forecast", line=dict(color=PRIMARY, width=2.5),
                    marker=dict(size=5),
                ))
                fig.update_layout(
                    title=f"<b>{sku}</b> — {horizon}-Day Demand Forecast",
                    xaxis_title="Day", yaxis_title="Units",
                    template="plotly_dark", hovermode="x unified",
                    legend=dict(orientation="h", y=1.05),
                    margin=dict(t=60, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Avg Daily",   f"{np.mean(forecast):.0f}")
                mc2.metric("Peak Day",    f"{max(forecast):.0f}")
                mc3.metric("Trough Day",  f"{min(forecast):.0f}")
                mc4.metric("Total",       f"{sum(forecast):,.0f}")

                rolling = pd.Series(forecast).rolling(7, min_periods=1).mean()
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=days, y=forecast, name="Daily",
                                     opacity=0.4, marker_color=PRIMARY))
                fig2.add_trace(go.Scatter(x=days, y=rolling, name="7-day MA",
                                         line=dict(color=WARNING, width=2)))
                fig2.update_layout(template="plotly_dark", height=220,
                                   margin=dict(t=20,b=20), barmode="overlay",
                                   legend=dict(orientation="h",y=1.1))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("👈 Set parameters and click **Run Forecast**")

# ════════════════════════════════════════════════════════════
# TAB 2 — ROUTE COST
# ════════════════════════════════════════════════════════════
with tab2:
    st.header("💰 Route Cost Prediction")
    st.markdown("**LightGBM** regression · 12 engineered features · R² ~0.91")
    st.divider()

    col_form, col_result = st.columns([1, 2])

    with col_form:
        st.subheader("Route Details")
        rc_distance  = st.number_input("Distance (km)",      10.0,1000.0,150.0,step=10.0)
        rc_load      = st.number_input("Load Weight (kg)",   10.0,5000.0,800.0,step=50.0)
        rc_stops     = st.number_input("Number of Stops",    1,30,5,step=1)
        rc_fuel      = st.number_input("Fuel Price (₹/L)",   80.0,120.0,95.5,step=0.5)
        rc_vehicle   = st.selectbox("Vehicle Type",
                                    ["Truck","Van","Mini-Truck","Tempo","Bike"])
        rc_hour      = st.slider("Departure Hour (0–23)", 0, 23, 9)
        rc_overtime  = st.number_input("Driver Overtime (hrs)", 0.0,8.0,0.0,step=0.5)
        with st.expander("Advanced Settings"):
            rc_fcons  = st.number_input("Fuel Consumption (L/km)",0.05,0.30,0.12,step=0.01)
            rc_cap    = st.number_input("Vehicle Capacity (kg)",100.0,10000.0,1000.0,step=100.0)
            rc_base   = st.number_input("Base Cost (₹)",100.0,2000.0,500.0,step=50.0)
        predict_btn = st.button("▶ Predict Cost", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            payload = {
                "distance_km": rc_distance, "load_weight_kg": rc_load,
                "num_stops": rc_stops, "fuel_price": rc_fuel,
                "vehicle_type": rc_vehicle, "departure_hour": rc_hour,
                "driver_overtime_hrs": rc_overtime,
                "fuel_consumption_per_km": rc_fcons,
                "vehicle_capacity_kg": rc_cap, "base_cost": rc_base,
            }
            with st.spinner("Running LightGBM inference ..."):
                data, err = api_post("/predict/route-cost", payload)
            if err:
                st.error(f"❌ {err}")
            else:
                cost = data["predicted_cost_inr"]
                st.markdown(f"""
                <div style="background:#1E293B;border-radius:16px;padding:2rem;text-align:center;
                            border:2px solid #4F6AF5;margin-bottom:1rem">
                    <div style="color:#94A3B8;font-size:.9rem">Predicted Route Cost</div>
                    <div style="color:#F1F5F9;font-size:3rem;font-weight:800">₹{cost:,.2f}</div>
                    <div style="color:#94A3B8;font-size:.85rem">
                        ₹{cost/rc_distance:.2f}/km &nbsp;·&nbsp;
                        ₹{cost/max(rc_stops,1):.2f}/stop
                    </div>
                </div>""", unsafe_allow_html=True)

                fuel_est  = rc_distance * rc_fuel * rc_fcons
                stop_est  = rc_stops * 80
                ot_est    = rc_overtime * 300
                other     = max(0, cost - fuel_est - stop_est - ot_est - rc_base)

                fig = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=["relative","relative","relative","relative","relative","total"],
                    x=["Base","Fuel","Stops","Overtime","Other","Total"],
                    y=[rc_base, fuel_est, stop_est, ot_est, other, 0],
                    connector={"line":{"color":"#334155"}},
                    decreasing={"marker":{"color":SUCCESS}},
                    increasing={"marker":{"color":PRIMARY}},
                    totals={"marker":{"color":WARNING}},
                    text=[f"₹{v:,.0f}" for v in [rc_base,fuel_est,stop_est,ot_est,other,cost]],
                    textposition="outside",
                ))
                fig.update_layout(title="Estimated Cost Breakdown",
                                  template="plotly_dark", height=350,
                                  yaxis_title="₹ INR", margin=dict(t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)

                if 8<=rc_hour<=10 or 17<=rc_hour<=19:
                    st.warning(f"⚠️ Peak traffic departure (hour {rc_hour})")
                if rc_hour>=20 or rc_hour<=5:
                    st.info("🌙 Overnight departure — driver overtime risk")
        else:
            st.info("👈 Fill in route details and click **Predict Cost**")
            st.dataframe(pd.DataFrame({
                "Vehicle":        ["Bike","Tempo","Van","Mini-Truck","Truck"],
                "Capacity (kg)":  ["100","600","1,500","1,000","5,000"],
                "Base Cost (₹)":  ["150","300","500","400","800"],
                "Best For":       ["<50km","50–150km","City routes","Regional","Long haul"],
            }), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — INVENTORY
# ════════════════════════════════════════════════════════════
with tab3:
    st.header("📦 Inventory Management")
    st.markdown("**Safety Stock · Reorder Points · ABC Classification**")
    st.divider()

    col_c3, col_ch3 = st.columns([1, 3])

    with col_c3:
        st.subheader("Filters")
        f_status    = st.multiselect("Status", ["REORDER NOW","OK"], default=["REORDER NOW","OK"])
        f_overstock = st.toggle("Overstocked Only", value=False)
        f_rows      = st.slider("Max rows", 10, 200, 50)
        load_inv    = st.button("▶ Load Signals", type="primary", use_container_width=True)
        st.caption("Reads `inventory_data.csv` via API startup cache")

    with col_ch3:
        if load_inv:
            with st.spinner("Fetching replenishment signals ..."):
                raw, err = api_get("/inventory/signals")
            if err:
                st.error(f"❌ {err}")
            else:
                df_inv = pd.DataFrame(raw)
                df_filt = df_inv.copy()
                if f_status:
                    df_filt = df_filt[df_filt["status"].isin(f_status)]
                if f_overstock:
                    df_filt = df_filt[df_filt["overstock_flag"] == True]

                total   = len(raw)
                reorder = sum(1 for r in raw if r["status"]=="REORDER NOW")
                over    = sum(1 for r in raw if r["overstock_flag"])

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Total SKUs",     total)
                m2.metric("🔴 Reorder Now", reorder,
                           delta=f"{reorder/total*100:.0f}%", delta_color="inverse")
                m3.metric("✅ Healthy",      total-reorder)
                m4.metric("🟡 Overstocked", over)

                ch1, ch2 = st.columns(2)
                with ch1:
                    vc = pd.DataFrame(raw)["status"].value_counts().reset_index()
                    vc.columns = ["Status","Count"]
                    fig_p = px.pie(vc, names="Status", values="Count",
                                   title="Inventory Status",
                                   color="Status",
                                   color_discrete_map={"REORDER NOW":DANGER,"OK":SUCCESS},
                                   hole=0.45, template="plotly_dark")
                    fig_p.update_layout(height=280, margin=dict(t=40,b=0,l=0,r=0))
                    st.plotly_chart(fig_p, use_container_width=True)

                with ch2:
                    dp = pd.DataFrame(raw).head(100)
                    fig_s = px.scatter(dp, x="reorder_point", y="current_stock",
                                       color="status", title="Stock vs Reorder Point",
                                       color_discrete_map={"REORDER NOW":DANGER,"OK":SUCCESS},
                                       hover_data=["sku_id"], template="plotly_dark")
                    lim = max(dp["reorder_point"].max(), dp["current_stock"].max())*1.05
                    fig_s.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                                    line=dict(color="#64748B",dash="dash",width=1))
                    fig_s.update_layout(height=280, margin=dict(t=40,b=0,l=0,r=0))
                    st.plotly_chart(fig_s, use_container_width=True)

                st.subheader(f"Signals — {len(df_filt)} shown")
                disp = ["sku_id","current_stock","reorder_point","safety_stock",
                        "status","overstock_flag"]
                disp = [c for c in disp if c in df_filt.columns]

                def hl(row):
                    return ["background-color:#2d0e0e"]*len(row) if row.get("status")=="REORDER NOW" else [""]*len(row)

                st.dataframe(
                    df_filt[disp].head(f_rows).style.apply(hl, axis=1)
                    .format({"current_stock":"{:,.0f}","reorder_point":"{:,.0f}","safety_stock":"{:,.0f}"}),
                    use_container_width=True, hide_index=True
                )
                csv = df_filt.to_csv(index=False).encode()
                st.download_button("⬇ Download CSV", csv, "inventory_signals.csv", "text/csv")
        else:
            st.info("👈 Click **Load Signals** to fetch live inventory status")

# ════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIVE MAINTENANCE
# ════════════════════════════════════════════════════════════
with tab4:
    st.header("🔧 Predictive Maintenance")
    st.markdown("**Isolation Forest** anomaly detection · Real-time sensor scoring")
    st.divider()

    preset = st.radio("Load Preset",
                       ["Normal Operation","Early Warning","Critical Fault","Manual"],
                       horizontal=True)
    presets = {
        "Normal Operation": (82.0, 0.32, 31.5, 2350.0, 80.0),
        "Early Warning":    (98.0, 0.71, 28.0, 3100.0, 55.0),
        "Critical Fault":   (142.0,1.85, 19.5, 4200.0, 18.0),
        "Manual":           (85.0, 0.35, 32.0, 2400.0, 78.0),
    }
    dt, dv, dp, dr, do = presets[preset]

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        vid  = st.text_input("Vehicle ID", "VH_001")
        temp = st.number_input("🌡 Temperature (°C)", 50.0, 200.0, dt, step=0.5)
        vib  = st.number_input("📳 Vibration",         0.0,   5.0, dv, step=0.05)
    with sc2:
        pres = st.number_input("⏱ Pressure (bar)",    10.0,  60.0, dp, step=0.5)
        rpm  = st.number_input("⚙️ RPM",             500.0,5000.0, dr, step=50.0)
    with sc3:
        oil  = st.number_input("🛢 Oil Level (%)",      0.0, 100.0, do, step=1.0)
        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button("▶ Analyse Sensor", type="primary", use_container_width=True)

    st.divider()

    if analyse_btn:
        payload = {"vehicle_id":vid,"temperature":temp,"vibration":vib,
                   "pressure":pres,"rpm":rpm,"oil_level":oil}
        with st.spinner("Running Isolation Forest ..."):
            data, err = api_post("/monitor/sensor", payload)
        if err:
            st.error(f"❌ {err}")
        else:
            is_anom = data["anomaly"]
            score   = data["anomaly_score"]
            bar_col = DANGER if is_anom else SUCCESS

            if is_anom:
                st.markdown(f'<div class="alert-warn">⚠️ <strong>MAINTENANCE ALERT — {vid}</strong><br>Anomalous sensor pattern. Schedule inspection immediately.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-ok">✅ <strong>NORMAL — {vid}</strong><br>All sensors within expected operating range.</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            cg, cr = st.columns(2)

            with cg:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=round(score,4),
                    title={"text":"Anomaly Score<br><span style='font-size:.8em'>Negative = risk</span>"},
                    delta={"reference":0,"valueformat":".4f"},
                    gauge={
                        "axis":{"range":[-0.5,0.5],"tickformat":".2f"},
                        "bar":{"color":bar_col},
                        "bgcolor":"#1E293B","bordercolor":"#334155",
                        "steps":[
                            {"range":[-0.5,-0.1],"color":"#450a0a"},
                            {"range":[-0.1, 0.1],"color":"#1c1917"},
                            {"range":[ 0.1, 0.5],"color":"#052e16"},
                        ],
                        "threshold":{"line":{"color":"#F1F5F9","width":2},"thickness":0.75,"value":0},
                    },
                    number={"valueformat":".4f"},
                ))
                fig_g.update_layout(template="plotly_dark", height=300,
                                    margin=dict(t=60,b=20,l=20,r=20))
                st.plotly_chart(fig_g, use_container_width=True)

            with cr:
                sv = [temp, vib*100, pres, rpm/50, oil]
                sl = ["Temperature","Vibration×100","Pressure","RPM÷50","Oil Level"]
                lo = [65,15,25,35,60]; hi = [95,80,37,70,100]
                norm = [min(1.0,max(0.0,(v-l)/max(h-l,1))) for v,l,h in zip(sv,lo,hi)]
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(
                    r=norm+[norm[0]], theta=sl+[sl[0]], fill="toself", name="Current",
                    line=dict(color=bar_col,width=2),
                    fillcolor=f"{'rgba(239,68,68,0.15)' if is_anom else 'rgba(34,197,94,0.15)'}",
                ))
                fig_r.add_trace(go.Scatterpolar(
                    r=[0.5]*len(sl)+[0.5], theta=sl+[sl[0]], fill=None, name="Normal",
                    line=dict(color=NEUTRAL,width=1,dash="dot"),
                ))
                fig_r.update_layout(
                    polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                    template="plotly_dark", height=300,
                    margin=dict(t=20,b=20,l=20,r=20), showlegend=True,
                    legend=dict(orientation="h",y=-0.1),
                )
                st.plotly_chart(fig_r, use_container_width=True)

            ranges = {"temperature":(65,95,"°C"),"vibration":(0.1,0.8,""),
                      "pressure":(25,37,"bar"),"rpm":(1800,3200,"RPM"),"oil_level":(60,100,"%")}
            rows = []
            for s,(lo,hi,unit) in ranges.items():
                val = payload[s]
                rows.append({"Sensor":s.replace("_"," ").title(),
                              "Value":f"{val:.2f} {unit}",
                              "Normal Range":f"{lo}–{hi} {unit}",
                              "Status":"✅ Normal" if lo<=val<=hi else "⚠️ Out of Range"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("👈 Select a preset or enter values and click **Analyse Sensor**")

# ════════════════════════════════════════════════════════════
# TAB 5 — ROUTE OPTIMISER
# ════════════════════════════════════════════════════════════
with tab5:
    st.header("🗺️ Route Optimisation")
    st.markdown("**Google OR-Tools VRP Solver** · Haversine matrix · Capacitated VRP support")
    st.divider()

    col_in, col_out = st.columns([1, 2])

    city_coords = {
        "Bengaluru": [[12.9716,77.5946],[12.9800,77.6100],[12.9600,77.5800],[12.9900,77.6200],[12.9500,77.5700],[12.9850,77.6050],[12.9650,77.5850],[12.9750,77.6150],[12.9550,77.5650],[12.9950,77.6000]],
        "Mumbai":    [[19.0760,72.8777],[19.0850,72.8900],[19.0650,72.8650],[19.0950,72.8800],[19.0700,72.8600],[19.0820,72.8720],[19.0680,72.8850],[19.0780,72.8950],[19.0600,72.8700],[19.0900,72.8650]],
        "Delhi":     [[28.6139,77.2090],[28.6250,77.2200],[28.6050,77.1980],[28.6300,77.2100],[28.6000,77.2000],[28.6200,77.2150],[28.6100,77.1900],[28.6350,77.2050],[28.5950,77.2100],[28.6180,77.2250]],
        "Hyderabad": [[17.3850,78.4867],[17.3950,78.4967],[17.3750,78.4767],[17.4050,78.4867],[17.3700,78.4867],[17.3900,78.4767],[17.3800,78.4967],[17.3950,78.4667],[17.3650,78.4967],[17.4000,78.5067]],
        "Chennai":   [[13.0827,80.2707],[13.0927,80.2807],[13.0727,80.2607],[13.1027,80.2707],[13.0700,80.2707],[13.0877,80.2607],[13.0777,80.2807],[13.0927,80.2507],[13.0650,80.2807],[13.0977,80.2907]],
    }

    with col_in:
        st.subheader("Solver Parameters")
        num_v    = st.slider("Number of Vehicles", 1, 10, 3)
        max_d    = st.slider("Max Distance / Vehicle (km)", 50, 500, 200)
        use_cap  = st.toggle("Enable CVRP (Capacity Constraints)", value=False)
        if use_cap:
            v_cap = st.number_input("Vehicle Capacity (units)", 100, 2000, 500, step=50)

        city = st.selectbox("City Preset", ["Custom","Bengaluru","Mumbai","Delhi","Hyderabad","Chennai"])
        if city != "Custom":
            def_locs = "\n".join(f"{a},{b}" for a,b in city_coords[city])
        else:
            def_locs = "12.9716,77.5946\n12.9800,77.6100\n12.9600,77.5800\n12.9900,77.6200\n12.9500,77.5700"

        locs_text = st.text_area("Locations (lat,lon — first = depot)", def_locs, height=180)

        if use_cap:
            n_lines = len([l for l in locs_text.strip().splitlines() if l.strip()])
            dem_text = st.text_area("Demands (depot=0, one per line)",
                                     "0\n" + "\n".join(["50"]*(n_lines-1)), height=120)

        solve_btn = st.button("▶ Optimise Routes", type="primary", use_container_width=True)

    with col_out:
        if solve_btn:
            try:
                locations = [[float(x) for x in l.strip().split(",")]
                             for l in locs_text.strip().splitlines() if l.strip()]
            except ValueError:
                st.error("❌ Invalid format. Use: lat,lon — one per line"); st.stop()
            if len(locations) < 3:
                st.error("Need at least 3 locations."); st.stop()

            payload = {"locations":locations,"num_vehicles":num_v,"max_distance_km":max_d}
            if use_cap:
                try:
                    demands = [int(x.strip()) for x in dem_text.strip().splitlines()]
                    payload["demands"] = demands
                    payload["vehicle_capacity"] = v_cap
                except ValueError:
                    st.error("❌ Demands must be integers."); st.stop()

            with st.spinner(f"Solving VRP: {len(locations)} locations, {num_v} vehicles ..."):
                data, err = api_post("/optimise/routes", payload, timeout=45)

            if err:
                st.error(f"❌ {err}")
            elif data.get("status") != "success":
                st.warning(f"Solver: {data.get('solver_status','No solution')}\n\nTip: increase max distance or add more vehicles")
            else:
                routes = data["routes"]
                m1,m2,m3 = st.columns(3)
                m1.metric("Vehicles Used",  data["vehicles_used"])
                m2.metric("Total Distance", f"{data['total_distance_km']} km")
                m3.metric("Avg/Vehicle",    f"{data['total_distance_km']//max(data['vehicles_used'],1)} km")

                colors_hex = [PRIMARY,SUCCESS,WARNING,DANGER,"#A855F7","#06B6D4","#EC4899","#84CC16","#F97316","#6366F1"]
                fig_map = go.Figure()
                fig_map.add_trace(go.Scattermapbox(
                    lat=[locations[0][0]], lon=[locations[0][1]],
                    mode="markers", marker=dict(size=18,color=WARNING,symbol="star"),
                    name="🏭 Depot",
                ))
                for i, r in enumerate(routes):
                    col = colors_hex[i % len(colors_hex)]
                    ri  = r["route"]
                    lats = [locations[idx][0] for idx in ri]
                    lons = [locations[idx][1] for idx in ri]
                    lbls = ["Depot" if idx==0 else f"Stop {idx}" for idx in ri]
                    fig_map.add_trace(go.Scattermapbox(
                        lat=lats, lon=lons, mode="lines+markers",
                        name=f"Vehicle {r['vehicle']} ({r['distance_km']} km)",
                        line=dict(width=3,color=col), marker=dict(size=10,color=col),
                        text=lbls, hoverinfo="text+name",
                    ))
                fig_map.update_layout(
                    mapbox=dict(style="open-street-map",
                                center=dict(lat=locations[0][0],lon=locations[0][1]),zoom=11),
                    margin=dict(l=0,r=0,t=0,b=0), height=400,
                    legend=dict(orientation="v",x=0.01,y=0.99,
                                bgcolor="rgba(0,0,0,0.5)",font=dict(color="white")),
                )
                st.plotly_chart(fig_map, use_container_width=True)

                summary = [{"Vehicle":f"Vehicle {r['vehicle']}",
                            "Stops":len([s for s in r["route"] if s!=0]),
                            "Route":"Depot → "+" → ".join(str(s) for s in r["route"] if s!=0)+" → Depot",
                            "Distance":f"{r['distance_km']} km"} for r in routes]
                st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

                fig_bar = go.Figure(go.Bar(
                    x=[f"Vehicle {r['vehicle']}" for r in routes],
                    y=[r["distance_km"] for r in routes],
                    marker_color=colors_hex[:len(routes)],
                    text=[f"{r['distance_km']} km" for r in routes],
                    textposition="outside",
                ))
                fig_bar.update_layout(title="Distance per Vehicle", template="plotly_dark",
                                      height=220, margin=dict(t=40,b=20), yaxis_title="km")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("👈 Choose a city preset and click **Optimise Routes**")
            sample = [{"city":"Bengaluru","lat":12.9716,"lon":77.5946},
                      {"city":"Mumbai",   "lat":19.0760,"lon":72.8777},
                      {"city":"Delhi",    "lat":28.6139,"lon":77.2090},
                      {"city":"Hyderabad","lat":17.3850,"lon":78.4867},
                      {"city":"Chennai",  "lat":13.0827,"lon":80.2707}]
            fig_prev = px.scatter_mapbox(pd.DataFrame(sample), lat="lat", lon="lon",
                                          hover_name="city", zoom=4,
                                          mapbox_style="open-street-map", height=380,
                                          color_discrete_sequence=[PRIMARY])
            fig_prev.update_layout(margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_prev, use_container_width=True)
