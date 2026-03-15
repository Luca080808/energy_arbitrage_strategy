import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Energy Arbitrage Handbook",
    layout="wide"
)


# ============================================================
# Constants
# ============================================================
FEATURE_LIST = [
    "imb_volume_lag6",
    "imb_volume_jump_lag6",
    "vwap_jump_lag6",
    "ID_QH_VWAP_lag6",
    "load_error_lag6",
    "wind_error_lag6",
    "solar_error_lag6",
    "total_reserve_lag6",
    "net_reserve_lag6",
    "reserve_ramping_lag6",
    "15_min_incr_sin",
    "15_min_incr_cos",
    "hour_incr_sin",
    "hour_incr_cos",
    "weekday_incr_sin",
    "weekday_incr_cos",
]


# ============================================================
# Cached data loaders
# ============================================================
@st.cache_data
def load_numpy_array(path: str):
    return np.load(path, allow_pickle=True)


@st.cache_data
def load_importance_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Unnamed: 0": "feature", "0": "importance"})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


@st.cache_data
def load_table(path: str, index_col=None) -> pd.DataFrame:
    return pd.read_csv(path, index_col=index_col)


@st.cache_data
def build_feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Feature_Class": [
                "Forecast Errors",
                "System imbalance",
                "Reserves",
                "Intraday Prices",
                "Temporal Data",
            ],
            "Features": [
                "solar_error_lag6, wind_error_lag6, load_error_lag6",
                "imb_volume_lag6, imb_volume_jump_lag6",
                "total_reserve_lag6, net_reserve_lag6, reserve_ramping_lag6",
                "ID_QH_VWAP_lag6, vwap_jump_lag6",
                "15_min_incr_sin, 15_min_incr_cos, hour_incr_sin, hour_incr_cos, weekday_incr_sin, weekday_incr_cos",
            ],
        }
    )


@st.cache_data
def load_all_data():
    data = {
        "X_test_index": load_numpy_array("../results/numpy_objects/X_test_index.npy"),
        "rf_pnl_vector": load_numpy_array("../results/numpy_objects/rf_pnl_vector.npy"),
        "hgb_pnl_vector": load_numpy_array("../results/numpy_objects/hgb_pnl_vector.npy"),
        "lgbm_pnl_vector": load_numpy_array("../results/numpy_objects/lgbm_pnl_vector.npy"),
        "perfect_pnl_vector": load_numpy_array("../results/numpy_objects/perfect_pnl_vector.npy"),
        "hgb_importance_avg": load_importance_table("../results/tables/hgb_importance_avg.csv"),
        "lgbm_importance_avg": load_importance_table("../results/tables/lgbm_importance_avg.csv"),
        "rf_importance_avg": load_importance_table("../results/tables/rf_importance_avg.csv"),
        "model_prediction_performance": load_table(
            "../results/tables/model_prediction_performance.csv", index_col=0
        ),
        "trading_performance": load_table(
            "../results/tables/trading_performance.csv", index_col=0
        ),
        "feature_table": build_feature_table(),
    }
    return data


DATA = load_all_data()


# ============================================================
# Helper functions
# ============================================================
def nul_f(x):
    return f"{x:,.0f}"


def percent_f(x):
    return f"{x:.2%}"


# ============================================================
# Plot functions
# ============================================================
def make_cumulative_pnl_plot(
    x_index,
    rf_pnl_vector,
    hgb_pnl_vector,
    lgbm_pnl_vector,
    perfect_pnl_vector=None,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x_index, rf_pnl_vector.cumsum(), label="Random Forest")
    ax.plot(x_index, hgb_pnl_vector.cumsum(), label="HistGradientBoosting")
    ax.plot(x_index, lgbm_pnl_vector.cumsum(), label="Light GBM")

    if perfect_pnl_vector is not None:
        ax.plot(
            x_index,
            perfect_pnl_vector.cumsum(),
            label="Theoretical Maximum PnL",
            color="red",
            linestyle="--",
        )

    ax.set_title("Cumulative PnL Results")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative PnL")
    ax.legend()
    ax.grid(True)

    return fig


def make_feature_importance_bar_plot(hgb_df, lgbm_df, rf_df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 8))

    ax[0].bar(hgb_df["feature"], hgb_df["importance"])
    ax[0].set_title("HGB")
    ax[0].set_xlabel("Feature")
    ax[0].set_ylabel("Importance")
    ax[0].tick_params(axis="x", rotation=90)

    ax[1].bar(lgbm_df["feature"], lgbm_df["importance"])
    ax[1].set_title("LGBM")
    ax[1].set_xlabel("Feature")
    ax[1].set_ylabel("Importance")
    ax[1].tick_params(axis="x", rotation=90)

    ax[2].bar(rf_df["feature"], rf_df["importance"])
    ax[2].set_title("RF")
    ax[2].set_xlabel("Feature")
    ax[2].set_ylabel("Importance")
    ax[2].tick_params(axis="x", rotation=90)

    plt.tight_layout()
    return fig


# ============================================================
# Page renderers
# ============================================================
def show_intro():
    st.title("Energy Arbitrage Handbook")
    st.markdown(
        """
In this handbook, we cover the machine learning pipeline
for identifying profitable arbitrage opportunities between
intraday and imbalance electricity prices.
"""
    )
    st.markdown(
        """
The imposed strategy consisted of using three regression models:

- **HistGradientBoosting**
- **LightGBM**
- **Random Forest**

All three models were split into two submodels, each responsible for predicting
long or short spreads.

The main question the models attempted to answer is:

**Given the predicted spreads for each side, when should we go long, short, or do nothing at all?**
"""
    )


def show_data_and_features():
    st.title("2. Selected Features")

    st.markdown(
        """
The selected feature dataset attempts to incorporate information
that a trader would typically use: time, forecast errors,
reserves, price, and volume indicators.
"""
    )

    st.dataframe(DATA["feature_table"], use_container_width=True)

    st.markdown(
        """
As mentioned previously, the targets were the long/short spreads defined as follows:

1. **Long Spread**: imbalance_price_pos - ID_QH_VWAP  
2. **Short Spread**: ID_QH_VWAP - imbalance_price_neg
"""
    )


def show_modeling():
    st.title("3. Models")
    st.markdown(
        """
In order to answer the following questions:

1. On average, how much do our predicted spreads deviate from the real ones?
2. What is the probability that each of our predicted spreads has the same sign as the real spread?
"""
    )

    st.markdown(
        r"""
We chose the following two evaluation metrics:

1. Mean Absolute Error:
$$
MAE(\hat{y}) = \frac{1}{n} \sum_{i=1}^n |\hat{y}_i - y_i|
$$

2. Mean Directional Accuracy:
$$
MDA(\hat{y}) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}_{\operatorname{sgn}(\hat{y}_i)=\operatorname{sgn}(y_i)}
$$
"""
    )

    st.markdown(
        "Below is a table regarding the model performance for each model and its constituent submodels. It is worth noting that we performed Time Series cross-validation on each model in order select the best performing variant."
    )
    st.subheader("Model Performance")
    st.dataframe(DATA["model_prediction_performance"], use_container_width=True)
    
    st.markdown(
        "Some additional plots we might be interested in are the feature importances selected by each model."
    )
    st.header("Feature Importance Selection")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("HGB")
        st.dataframe(DATA["hgb_importance_avg"], use_container_width=True)

    with col2:
        st.subheader("LGBM")
        st.dataframe(DATA["lgbm_importance_avg"], use_container_width=True)

    with col3:
        st.subheader("RF")
        st.dataframe(DATA["rf_importance_avg"], use_container_width=True)

    st.pyplot(
        make_feature_importance_bar_plot(
            DATA["hgb_importance_avg"],
            DATA["lgbm_importance_avg"],
            DATA["rf_importance_avg"],
        )
    )
    
    st.info("It is worth noting that we performed Time Series cross-validation on each model in order select the best performing variant.")



def show_strategy_and_performance():
    st.title("4. Buy/Sell Logic and Results")

    st.markdown(
        """
Having trained the models, we can execute our strategy:

1. Develop data-driven buy and sell thresholds that would theoretically optimize PnL.
2. Develop a rule based on which a decision is made to go long, short, or do nothing.
"""
    )

    st.markdown(
        """
Due to the lengthy code for both steps, consult the Jupyter notebook.
See the functions `optimize_buy_sell_thresholds_and_pnl` and `compute_strategy_pnl`.
"""
    )

    st.markdown(
        """
In order to have an upper-bound perspective of the maximum PnL a trader could achieve,
we introduce the notion of **Perfect PnL**.

This Perfect PnL represents the maximum amount of profit we could have gained for each trade,
given omniscience.
"""
    )

    st.markdown(
        r"""
$$
(\text{Perfect\_PnL}_i)_{i=1}^n
=
(\max\{\text{spread\_buy}_i,\ \text{spread\_sell}_i,\ 0\})_{i=1}^n
$$
"""
    )

    st.markdown(
        "We can now have a look at how each model compares to one another via cumulative PnL plots."
    )

    st.pyplot(
        make_cumulative_pnl_plot(
            DATA["X_test_index"],
            DATA["rf_pnl_vector"],
            DATA["hgb_pnl_vector"],
            DATA["lgbm_pnl_vector"],
            DATA["perfect_pnl_vector"],
        )
    )

    st.pyplot(
        make_cumulative_pnl_plot(
            DATA["X_test_index"],
            DATA["rf_pnl_vector"],
            DATA["hgb_pnl_vector"],
            DATA["lgbm_pnl_vector"],
        )
    )

    st.markdown(
        "Finally, here are the essential summary statistics regarding the trading performance for each model."
    )
    st.dataframe(DATA["trading_performance"], use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(r"""
    $$
    \text{Max\_Drawdown} = \max_{1 \le t \le n} D_t
    $$

    $$
    D_t = \max_{1 \le i \le t}\left(\sum_{j=1}^{i} X_j\right) - \sum_{j=1}^{t} X_j
    $$
                
    $$
    \text{where } X_i \text{ is the PnL of the }  i^\text{th} \, trade.
    $$
    """)
    
    with col2:
        st.info(r"""
    $$
    \text{PnL\_T\_Stat} = \frac{\overline{X}}{SE}
    $$
                
    $$
    SE = \frac{\sqrt{\frac{1}{n-1}\sum\limits_{i=1}^n (X_i - \overline{X})^2}}{\sqrt{n}}
    $$
    
    $$
    \overline{X} = \frac{1}{n}\sum\limits_{i=1}^n X_i 
    $$
    $$
    \text{where } X_i \text{ is the PnL of the }  i^\text{th} \, trade.
    $$
                """)


def show_conclusions():
    st.title("5. Conclusion")
    st.markdown("In conclusion, while all three models generate modest, statistically significant and steady returns, the gradient boosting models (HGB, LGBM) have a performance and predictive edge over the bagging ones (RF).")

    st.markdown("The most succesful model is LGBM with a cumulative PNL of 20,463,070 RON.")

    st.header("5.1 Suggestions of improvements")
    st.markdown("1. A note-worthy investigation could be the replacement of dominant feature **imb_volume** with some derivation off of it, such as a rolling mean of the first n **imb_volume** data. The ambitious reader is invited to try this out.")
    st.markdown("2. Whilst the Perfect Pnl offers us the best achievable upper-bound, a good lower-bound would be the cumulative PnL of the responsible traders. This benchmark would allows us to determine if our models outperform/underperform relative to them")



# ============================================================
# Sidebar navigation
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Data & Features",
        "Modeling", #Migrate both of them
        "Buy/Sell Logic and Results", #Migrate both of them
        "Conclusions",
    ],
)


# ============================================================
# Router
# ============================================================
if page == "Overview":
    show_intro()
elif page == "Data & Features":
    show_data_and_features()
elif page == "Modeling":
    show_modeling()
elif page == "Buy/Sell Logic and Results":
    show_strategy_and_performance()
elif page == "Conclusions":
    show_conclusions()
