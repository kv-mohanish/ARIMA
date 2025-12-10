# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import io

st.set_page_config(page_title="ML-grade ARIMA (Streamlit)", layout="wide")
st.title("ML-grade ARIMA — fitted with statsmodels (MLE)")

# -------------------------
# Utilities & data creation
# -------------------------
@st.cache_data
def generate_series(n, seasonal_period, seasonal_strength, trend_strength, noise_level, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(n)
    trend = trend_strength * t
    seasonal = seasonal_strength * np.sin(2 * np.pi * t / seasonal_period)
    noise = np.random.normal(scale=noise_level, size=n)
    series = 100.0 + trend + seasonal + noise
    idx = pd.RangeIndex(start=0, stop=n, step=1)
    return pd.Series(series, index=idx)

def train_test_split(ts, test_size):
    n = len(ts)
    split = max(n - int(test_size), 0)
    train = ts.iloc[:split]
    test = ts.iloc[split:]
    return train, test

def safe_fit_arima(train_series, order, enforce_stationarity=True, enforce_invertibility=True):
    model = ARIMA(
        train_series,
        order=order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility
    )
    fitted = model.fit()   # ← FIXED (no disp)
    return fitted


def aic_grid_search(train_series, p_max, d, q_max, ic='aic'):
    best = None
    best_order = None
    best_aic = np.inf
    history = []
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            try:
                res = safe_fit_arima(train_series, order=(p, d, q))
                val = getattr(res, ic)
                history.append((p, d, q, val))
                if val < best_aic:
                    best_aic = val
                    best = res
                    best_order = (p, d, q)
            except Exception as e:
                # skip problematic configurations (singular matrices etc.)
                history.append((p, d, q, np.nan))
                continue
    return best_order, best, pd.DataFrame(history, columns=['p','d','q',ic])

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Data generation")
    data_points = st.slider("Data points", 60, 2000, 300, step=10)
    seasonal_period = st.slider("Seasonal period", 2, 52, 12)
    seasonal_strength = st.slider("Seasonal strength", 0.0, 100.0, 10.0)
    trend_strength = st.slider("Trend strength (slope)", 0.0, 5.0, 0.3, step=0.01)
    noise_level = st.slider("Noise std dev", 0.0, 50.0, 5.0)
    seed = st.number_input("Random seed (0 for deterministic)", value=0, step=1)
    if seed == 0:
        seed = None

    st.markdown("---")
    st.header("Training / Forecast")
    test_size = st.slider("Forecast horizon (steps)", 5, 200, 20)
    train_fraction = st.slider("Train fraction (if <1, part held out)", 50, 95, 80)
    train_fraction = train_fraction / 100.0

    st.markdown("---")
    st.header("ARIMA options")
    mode = st.radio("Select mode", ["Manual (p,d,q)", "Auto-select (AIC grid search)"])
    if mode == "Manual (p,d,q)":
        p = st.number_input("p (AR order)", min_value=0, max_value=10, value=2, step=1)
        d = st.number_input("d (difference order)", min_value=0, max_value=2, value=1, step=1)
        q = st.number_input("q (MA order)", min_value=0, max_value=10, value=1, step=1)
        selected_order = (int(p), int(d), int(q))
        run_auto = False
    else:
        p_max = st.number_input("Max p (grid)", min_value=0, max_value=8, value=5, step=1)
        d = st.number_input("d (difference order)", min_value=0, max_value=2, value=1, step=1)
        q_max = st.number_input("Max q (grid)", min_value=0, max_value=8, value=5, step=1)
        run_auto = st.button("Run auto AIC grid search")

    enforce_stationarity = st.checkbox("Enforce stationarity", value=True)
    enforce_invertibility = st.checkbox("Enforce invertibility", value=True)

# --------------
# Generate data
# --------------
ts = generate_series(data_points, seasonal_period, seasonal_strength, trend_strength, noise_level, seed=seed)

# Train/test set by fraction
train_len = int(len(ts) * train_fraction)
train, test = ts.iloc[:train_len], ts.iloc[train_len:train_len + test_size]

# Display basic info
st.markdown("### Data snapshot")
col1, col2 = st.columns([2,1])
with col1:
    st.line_chart(ts)
with col2:
    st.write("Length:", len(ts))
    st.write("Train length:", len(train))
    st.write("Test length (horizon):", len(test))

# --------------------------
# Fit model (manual or auto)
# --------------------------
fitted_result = None
selected_order_text = ""
aic_table = None
if mode == "Manual (p,d,q)":
    selected_order_text = f"Manual order selected: (p,d,q) = {selected_order}"
    st.info(selected_order_text)
    try:
        with st.spinner("Fitting ARIMA (MLE)..."):
            fitted_result = safe_fit_arima(train, selected_order, enforce_stationarity=enforce_stationarity, enforce_invertibility=enforce_invertibility)
    except Exception as e:
        st.error(f"ARIMA fit failed: {e}")
else:
    st.info(f"Auto grid search set to p ≤ {p_max}, d = {d}, q ≤ {q_max}. Click the button to run.")
    if run_auto:
        with st.spinner("Running AIC grid search (this can take time)..."):
            best_order, best_res, aic_table = aic_grid_search(train, int(p_max), int(d), int(q_max), ic='aic')
            if best_res is None:
                st.error("Auto-selection failed for all grid candidates.")
            else:
                fitted_result = best_res
                selected_order = best_order
                selected_order_text = f"Auto-selected order: (p,d,q) = {best_order} with AIC={best_res.aic:.3f}"
                st.success(selected_order_text)

# If we have a fitted model, show results and forecasts
if fitted_result is not None:
    st.subheader("Model fit summary (MLE)")
    st.text(f"Order (p,d,q): {fitted_result.model_orders if hasattr(fitted_result,'model_orders') else selected_order}")
    with st.expander("Show full statsmodels summary"):
        st.text(fitted_result.summary().as_text())

    # In-sample fitted values and residuals
    fitted_in_sample = fitted_result.fittedvalues
    resid = fitted_result.resid

    # Forecast: use get_forecast for steps ahead from end of train (covering test length)
    steps = len(test) if len(test) > 0 else test_size
    forecast_obj = fitted_result.get_forecast(steps=steps)
    forecast_mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.05)  # 95%

    # Combine for plotting
    index_train = train.index
    index_test = np.arange(train.index[-1] + 1, train.index[-1] + 1 + steps)
    df_plot = pd.DataFrame({
        "observed": ts.values,
    }, index=np.arange(len(ts)))
    # Construct arrays for fitted / forecast
    fitted_plot = np.full(len(ts), np.nan)
    fitted_plot[:len(fitted_in_sample)] = fitted_in_sample
    forecast_plot = np.full(len(ts), np.nan)
    forecast_plot[index_test] = forecast_mean.values
    lower = np.full(len(ts), np.nan)
    upper = np.full(len(ts), np.nan)
    lower[index_test] = conf_int.iloc[:,0].values
    upper[index_test] = conf_int.iloc[:,1].values

    # Plot observed, fitted, forecast + CI
    st.subheader("Observed, Fitted (in-sample) and Forecast (out-of-sample)")
    fig, ax = plt.subplots(figsize=(10,4.5))
    ax.plot(df_plot.index, df_plot['observed'], label='Observed', linewidth=1.4)
    ax.plot(df_plot.index, fitted_plot, label='Fitted (in-sample)', linewidth=1.2)
    ax.plot(df_plot.index, forecast_plot, label='Forecast (out-of-sample)', linestyle='--', linewidth=1.4)
    ax.fill_between(df_plot.index, lower, upper, color='lightgrey', alpha=0.5, label='95% CI')
    ax.axvline(x=len(train)-1, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    # Performance metrics on test set (if available)
    if len(test) > 0:
        obs = test.values[:len(forecast_mean)]
        pred = forecast_mean.values
        mae = np.mean(np.abs(obs - pred))
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        mape = np.mean(np.abs((obs - pred) / obs)) * 100
        st.subheader("Forecast performance on holdout")
        st.write(f"MAE = {mae:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.2f}%")

    # Residual diagnostics
    st.subheader("Residual diagnostics")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.write("Residual mean (approx):", float(resid.mean()))
        st.write("Residual std:", float(resid.std()))
        # Ljung-Box
        try:
            lb = acorr_ljungbox(resid, lags=[10], return_df=True)
            st.write("Ljung-Box (lag 10) p-value:", float(lb['lb_pvalue'].iloc[0]))
            if float(lb['lb_pvalue'].iloc[0]) < 0.05:
                st.warning("Residuals show significant autocorrelation (p < 0.05). Model may be misspecified.")
            else:
                st.success("No significant residual autocorrelation (Ljung-Box p >= 0.05).")
        except Exception as e:
            st.write("Could not compute Ljung-Box:", e)

    with col_b:
        st.write("ACF of residuals (first 20 lags)")
        acf_vals = sm_acf(resid.dropna(), nlags=20, fft=False)
        fig2, ax2 = plt.subplots(figsize=(6,2.6))
        ax2.bar(np.arange(len(acf_vals)), acf_vals)
        ax2.axhline(0, color='k')
        ax2.set_ylim(-1,1)
        st.pyplot(fig2)

    with col_c:
        st.write("QQ-plot of residuals")
        fig3, ax3 = plt.subplots(figsize=(4.5,3))
        stats.probplot(resid.dropna(), plot=ax3)
        st.pyplot(fig3)

    # Show AIC/BIC
    st.subheader("Information criteria")
    st.write(f"AIC = {fitted_result.aic:.4f}, BIC = {fitted_result.bic:.4f}")

    # Show parameter table
    st.subheader("Estimated parameters")
    params_table = pd.DataFrame({
        "param": fitted_result.params.index,
        "estimate": fitted_result.params.values,
        "stderr": fitted_result.bse.values
    })
    st.dataframe(params_table.style.format({"estimate":"{:.6f}", "stderr":"{:.6f}"}), height=200)

    # Download forecast CSV
    out_df = pd.DataFrame({
        "index": index_test,
        "forecast_mean": forecast_mean.values,
        "lower_95": conf_int.iloc[:,0].values,
        "upper_95": conf_int.iloc[:,1].values
    }).set_index("index")
    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf)
    csv_bytes = csv_buf.getvalue().encode()

    st.download_button("Download forecast CSV", data=csv_bytes, file_name="forecast.csv", mime="text/csv")

    # If auto search ran, show AIC table for grid
    if aic_table is not None:
        st.subheader("Grid search results (AIC)")
        st.dataframe(aic_table.sort_values('aic').reset_index(drop=True).head(30))

else:
    st.info("No fitted model yet. For manual mode, model fits automatically. For auto mode press 'Run auto AIC grid search'.")

st.markdown("---")
st.caption("Notes: This app fits ARIMA via statsmodels using MLE (statespace). Auto-selection uses AIC grid search over (p,q). For production forecasting use additional validation and consider exogenous regressors, seasonal SARIMA variants, or more advanced tools (pmdarima, tbats, Prophet, or full Bayesian inference).")
