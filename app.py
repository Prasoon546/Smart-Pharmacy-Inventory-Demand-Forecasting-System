import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(
    page_title="SmartPharm | Inventory & Demand Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .main-title {font-size: 2.1rem; font-weight: 800; margin-bottom: 0.1rem;}
    .subtitle {color: #64748b; font-size: 1rem; margin-bottom: 1.3rem;}
    .section-title {font-size: 1.35rem; font-weight: 750; margin-top: 0.6rem;}
    .info-box {
        padding: 14px 16px; border-radius: 12px; background: #f8fafc;
        border: 1px solid #e2e8f0; margin-bottom: 10px;
    }
    .flow-box {
        padding: 12px 14px; border-radius: 10px; background: white;
        border: 1px solid #e2e8f0; text-align: center; min-height: 85px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------

def format_number(num: float) -> str:
    """Formats large numbers into readable strings with K or M suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return f"{num:.0f}"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in out.columns
    ]

    aliases = {
        "date": "sale_date",
        "sales_date": "sale_date",
        "transaction_date": "sale_date",
        "medicine_name": "medicine",
        "drug_name": "medicine",
        "qty": "quantity_sold",
        "quantity": "quantity_sold",
        "sold_quantity": "quantity_sold",
        "batch": "batch_number",
        "mfg_date": "manufacturing_date",
        "manufactured_date": "manufacturing_date",
        "exp_date": "expiry_date",
        "expiry": "expiry_date",
        "stock": "current_stock",
        "purchase_cost": "purchase_price",
        "cost_price": "purchase_price",
        "selling_rate": "selling_price",
        "mrp": "selling_price",
    }

    return out.rename(columns={c: aliases.get(c, c) for c in out.columns})


def coerce_dates(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def prepare_data(
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    medicines_master: pd.DataFrame
):
    sales = normalize_columns(sales)
    inventory = normalize_columns(inventory)
    medicines_master = normalize_columns(medicines_master)

    sales = coerce_dates(sales, ["sale_date"])
    sales = coerce_numeric(sales, ["quantity_sold", "selling_price"])

    inventory = coerce_dates(
        inventory,
        ["manufacturing_date", "expiry_date"]
    )

    inventory = coerce_numeric(
        inventory,
        ["current_stock", "minimum_stock_level", "reorder_level"]
    )

    medicines_master = coerce_numeric(
        medicines_master,
        ["purchase_price", "selling_price"]
    )

    required_sales = {
        "sale_date",
        "medicine",
        "quantity_sold"
    }

    required_inv = {
        "inventory_id",
        "medicine_id",
        "batch_number",
        "expiry_date",
        "current_stock",
        "minimum_stock_level",
        "reorder_level",
    }

    required_master = {
        "medicine_id",
        "medicine",
        "category",
        "company",
        "purchase_price",
        "selling_price",
        "supplier",
    }

    missing_sales = required_sales - set(sales.columns)
    missing_inv = required_inv - set(inventory.columns)
    missing_master = required_master - set(medicines_master.columns)

    if missing_sales:
        raise ValueError(
            f"Sales file is missing: {', '.join(sorted(missing_sales))}"
        )

    if missing_inv:
        raise ValueError(
            f"Inventory file is missing: {', '.join(sorted(missing_inv))}"
        )

    if missing_master:
        raise ValueError(
            f"Medicine file is missing: {', '.join(sorted(missing_master))}"
        )

    if "stock_status" in inventory.columns:
        inventory = inventory.drop(columns=["stock_status"])

    sales = sales.dropna(
        subset=["sale_date", "medicine", "quantity_sold"]
    ).copy()

    inventory = inventory.dropna(
        subset=[
            "inventory_id",
            "medicine_id",
            "batch_number",
            "expiry_date",
            "current_stock",
            "minimum_stock_level",
            "reorder_level",
        ]
    ).copy()

    medicines_master = medicines_master.dropna(
        subset=[
            "medicine_id",
            "medicine",
            "category",
            "company",
            "purchase_price",
            "selling_price",
            "supplier",
        ]
    ).copy()

    sales["medicine"] = sales["medicine"].astype(str)
    inventory["medicine_id"] = inventory["medicine_id"].astype(str)
    medicines_master["medicine_id"] = medicines_master["medicine_id"].astype(str)

    master_cols = [
        "medicine_id",
        "medicine",
        "category",
        "company",
        "purchase_price",
        "selling_price",
        "supplier",
    ]

    inventory = inventory.merge(
        medicines_master[master_cols],
        on="medicine_id",
        how="left",
        validate="many_to_one",
    )

    inventory = inventory.dropna(
        subset=[
            "medicine",
            "category",
            "company",
            "purchase_price",
            "selling_price",
            "supplier",
        ]
    ).copy()

    sales["quantity_sold"] = sales["quantity_sold"].clip(lower=0)

    inventory["current_stock"] = (
        inventory["current_stock"]
        .fillna(0)
        .clip(lower=0)
    )

    inventory["purchase_price"] = (
        inventory["purchase_price"]
        .fillna(0)
        .clip(lower=0)
    )

    return sales.sort_values("sale_date"), inventory, medicines_master


@st.cache_data(show_spinner=False)
def inventory_summary(
    inventory: pd.DataFrame,
    critical_days: int = 30,
    warning_days: int = 90,
):
    """
    Builds the expiry-risk view of inventory.

    critical_days / warning_days are configurable business rules (not
    hardcoded constants) — they come from sidebar controls so the
    thresholds can match a store's real reorder lead time / shelf-life
    policy instead of an arbitrary fixed number baked into the code.
    """

    today = pd.Timestamp.today().normalize()

    inv = inventory.copy()

    inv["days_until_expiry"] = (
        inv["expiry_date"] - today
    ).dt.days

    inv["inventory_value"] = (
        inv["current_stock"] * inv["purchase_price"]
    ).round(0).astype(int)

    inv["expiry_status"] = np.select(
        [
            inv["days_until_expiry"] < 0,
            inv["days_until_expiry"].between(0, critical_days),
            inv["days_until_expiry"].between(critical_days + 1, warning_days),
        ],
        [
            "Expired",
            "Critical",
            "Warning",
        ],
        default="Safe",
    )

    return inv


@st.cache_data(show_spinner=False)
def build_daily_series(
    sales: pd.DataFrame,
    medicine: str
) -> pd.DataFrame:

    x = (
        sales[sales["medicine"] == medicine]
        .groupby("sale_date", as_index=False)["quantity_sold"]
        .sum()
    )

    if x.empty:
        return pd.DataFrame(
            columns=["sale_date", "quantity_sold"]
        )

    idx = pd.date_range(
        x["sale_date"].min(),
        x["sale_date"].max(),
        freq="D"
    )

    x = (
        x.set_index("sale_date")
        .reindex(idx, fill_value=0)
        .rename_axis("sale_date")
        .reset_index()
    )

    return x


@st.cache_data(show_spinner=False)
def forecast_medicine(
    sales: pd.DataFrame,
    medicine: str,
    horizon: int = 30
):

    ts = build_daily_series(sales, medicine)

    if len(ts) < 21:

        avg = (
            float(
                ts["quantity_sold"]
                .tail(min(14, len(ts)))
                .mean()
            )
            if len(ts)
            else 0.0
        )

        future_dates = pd.date_range(
            pd.Timestamp.today().normalize()
            + pd.Timedelta(days=1),
            periods=horizon,
            freq="D"
        )

        pred = np.maximum(
            0,
            np.round(
                np.repeat(avg, horizon),
                1
            )
        )

        return pd.DataFrame(
            {
                "date": future_dates,
                "predicted_demand": pred
            }
        )

    data = ts.copy()

    for lag in [1, 2, 3, 7, 14]:
        data[f"lag_{lag}"] = (
            data["quantity_sold"].shift(lag)
        )

    for window in [7, 14]:
        data[f"roll_{window}"] = (
            data["quantity_sold"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    data["dow"] = data["sale_date"].dt.dayofweek
    data["day_of_month"] = data["sale_date"].dt.day

    data = data.dropna().reset_index(drop=True)

    feature_cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "roll_7",
        "roll_14",
        "dow",
        "day_of_month",
    ]

    X = data[feature_cols]
    y = data["quantity_sold"]

    final_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X, y)

    history_values = (
        ts["quantity_sold"]
        .astype(float)
        .tolist()
    )

    future = []

    next_date = (
        ts["sale_date"].max()
        + pd.Timedelta(days=1)
    )

    for i in range(horizon):

        d = (
            next_date
            + pd.Timedelta(days=i)
        )

        l1 = history_values[-1]
        l2 = history_values[-2]
        l3 = history_values[-3]
        l7 = history_values[-7]
        l14 = history_values[-14]

        r7 = float(
            np.mean(history_values[-7:])
        )

        r14 = float(
            np.mean(history_values[-14:])
        )

        row = pd.DataFrame(
            [
                {
                    "lag_1": l1,
                    "lag_2": l2,
                    "lag_3": l3,
                    "lag_7": l7,
                    "lag_14": l14,
                    "roll_7": r7,
                    "roll_14": r14,
                    "dow": d.dayofweek,
                    "day_of_month": d.day,
                }
            ]
        )[feature_cols]

        p = float(
            max(
                0,
                final_model.predict(row)[0]
            )
        )

        future.append(p)
        history_values.append(p)

    future_dates = pd.date_range(
        next_date,
        periods=horizon,
        freq="D"
    )

    return pd.DataFrame(
        {
            "date": future_dates,
            "predicted_demand": future
        }
    )


@st.cache_data(show_spinner=False)
def build_sales_analytics(
    sales: pd.DataFrame,
    medicines_master: pd.DataFrame
) -> pd.DataFrame:

    price_lookup = (
        medicines_master
        .groupby("medicine", as_index=False)
        .agg(
            category=("category", "first"),
            company=("company", "first"),
            purchase_price=("purchase_price", "mean"),
            selling_price=("selling_price", "mean"),
        )
    )

    overlap_cols = [
        c for c in ["category", "company", "purchase_price", "selling_price"]
        if c in sales.columns
    ]

    sales_for_merge = sales.drop(columns=overlap_cols)

    merged = sales_for_merge.merge(
        price_lookup,
        on="medicine",
        how="left"
    )

    merged["selling_price"] = merged["selling_price"].fillna(0)
    merged["purchase_price"] = merged["purchase_price"].fillna(0)

    merged["revenue"] = merged["quantity_sold"] * merged["selling_price"]
    merged["cost"] = merged["quantity_sold"] * merged["purchase_price"]
    merged["profit"] = merged["revenue"] - merged["cost"]

    merged["margin_pct"] = np.where(
        merged["selling_price"] > 0,
        (merged["selling_price"] - merged["purchase_price"])
        / merged["selling_price"] * 100,
        0.0,
    )

    return merged


# -----------------------------
# Sidebar & data loading
# -----------------------------

st.sidebar.markdown("## 🏥 SmartPharm")
st.sidebar.caption("Inventory & Demand Intelligence")

sales_file = st.sidebar.file_uploader(
    "Upload sales CSV",
    type=["csv"]
)

inventory_file = st.sidebar.file_uploader(
    "Upload inventory CSV",
    type=["csv"]
)

medicines_file = st.sidebar.file_uploader(
    "Upload medicine CSV",
    type=["csv"]
)

if (
    not sales_file
    or not inventory_file
    or not medicines_file
):

    st.info(
        "Upload sales, inventory and medicine CSV files to start."
    )

    st.stop()


sales_df = pd.read_csv(sales_file)
inventory_df = pd.read_csv(inventory_file)
medicines_df = pd.read_csv(medicines_file)


try:

    sales_df, inventory_df, medicines_df = prepare_data(
        sales_df,
        inventory_df,
        medicines_df
    )

except Exception as e:

    st.error(str(e))
    st.stop()


# -----------------------------
# Configurable expiry-risk thresholds
# -----------------------------
# These used to be hardcoded (<=30 / <=90 days) inside the code. They are
# now sidebar controls, because "how many days is risky" is a business
# decision (tied to reorder lead time / shelf-life policy) — not a
# constant that belongs baked into the source.

st.sidebar.markdown("### ⚙️ Expiry Risk Thresholds")
st.sidebar.caption(
    "Set these to match your real reorder lead time / shelf-life policy."
)

critical_days = st.sidebar.slider(
    "🟠 Critical window (days)",
    min_value=7,
    max_value=60,
    value=30,
    step=1,
    help="Batches expiring within this many days are flagged Critical.",
)

warning_days = st.sidebar.slider(
    "🟡 Warning window (days)",
    min_value=15,
    max_value=180,
    value=90,
    step=1,
    help="Batches expiring after the Critical window, but within this many "
         "days, are flagged Warning.",
)

if warning_days <= critical_days:
    st.sidebar.warning(
        "Warning window should be greater than the Critical window. "
        "Using Critical + 1 day as the Warning boundary."
    )
    warning_days = critical_days + 1


inventory_df = inventory_summary(
    inventory_df,
    critical_days=critical_days,
    warning_days=warning_days,
)

medicines = sorted(
    sales_df["medicine"]
    .dropna()
    .unique()
    .tolist()
)


# -----------------------------
# Navigation
# -----------------------------

page = st.sidebar.radio(
    "Navigate",
    [
        "📦 Inventory",
        "📈 Sales Analytics",
        "🤖 Demand Forecast",
        "⏰ Expiry Tracking"
    ],
)


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="main-title">SmartPharm</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Smart Pharmacy Inventory & Demand Forecasting System</div>',
    unsafe_allow_html=True
)


if page == "📦 Inventory":

    st.markdown(
        '<div class="section-title">📦 Inventory Management</div>',
        unsafe_allow_html=True
    )

    query = st.selectbox(
        "🔎 Select medicine",
        options=["All Medicines"] + medicines
    )

    inv_view = inventory_df.copy()

    if query != "All Medicines":

        inv_view = inv_view[
            inv_view["medicine"] == query
        ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Medicines in View",
        inv_view["medicine"].nunique()
    )

    c2.metric(
        "Stock in View",
        f"{int(inv_view['current_stock'].sum()):,}"
    )

    c3.metric(
        "Batches",
        len(inv_view)
    )

    c4.metric(
        "Inventory Value",
        f"₹{format_number(inv_view['inventory_value'].sum())}"
    )

    display_cols = [
        c for c in [
            "medicine",
            "category",
            "company",
            "supplier",
            "batch_number",
            "current_stock",
            "purchase_price",
            "selling_price",
            "expiry_date",
            "days_until_expiry",
            "expiry_status",
        ]
        if c in inv_view.columns
    ]

    st.dataframe(
        inv_view[
            display_cols
        ]
        .sort_values(
            ["medicine", "expiry_date"]
        )
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

    if query != "All Medicines" and not inv_view.empty:

        selected = query

        st.markdown(
            f"### 💊 Batch details — {selected}"
        )

        batch_view = (
            inv_view[
                inv_view["medicine"] == selected
            ]
            .sort_values("expiry_date")
        )

        bcols = [
            c for c in [
                "batch_number",
                "current_stock",
                "manufacturing_date",
                "expiry_date",
                "purchase_price",
                "selling_price",
                "days_until_expiry",
                "expiry_status",
            ]
            if c in batch_view.columns
        ]

        st.dataframe(
            batch_view[bcols],
            use_container_width=True,
            hide_index=True
        )


elif page == "📈 Sales Analytics":

    st.markdown(
        '<div class="section-title">📈 Sales Analytics</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Revenue, pricing and margin patterns across all sales — "
        "select a medicine to drill every chart down to it."
    )

    sales_analytics = build_sales_analytics(
        sales_df,
        medicines_df
    )

    search_med = st.selectbox(
        "🔎 Select a medicine to filter the charts below",
        options=["All Medicines"] + medicines
    )

    is_filtered = search_med != "All Medicines"

    if is_filtered:
        filtered = sales_analytics[
            sales_analytics["medicine"] == search_med
        ]
        suffix = f" — {search_med}"
    else:
        filtered = sales_analytics
        suffix = " — All Medicines"

    # KPI row formatted cleanly
    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Revenue",
        f"₹{format_number(filtered['revenue'].sum())}"
    )

    k2.metric(
        "Profit",
        f"₹{format_number(filtered['profit'].sum())}"
    )

    k3.metric(
        "Avg Margin",
        f"{filtered['margin_pct'].mean():.1f}%"
        if not filtered.empty else "0.0%"
    )

    k4.metric(
        "Units Sold",
        f"{format_number(filtered['quantity_sold'].sum())}"
    )

    # 1. Monthly Revenue trend over time
    trend_source = filtered.copy()
    trend_source["period"] = (
        trend_source["sale_date"].dt.to_period("M").dt.to_timestamp()
    )

    trend = (
        trend_source
        .groupby("period", as_index=False)["revenue"]
        .sum()
        .sort_values("period")
    )

    fig1 = px.line(
        trend,
        x="period",
        y="revenue",
        markers=True,
        title=f"Monthly Revenue Trend{suffix}"
    )

    fig1.update_layout(
        height=380,
        xaxis_title="Month",
        yaxis_title="Revenue (₹)"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # 2. Top medicines by revenue
    top_rev = (
        sales_analytics
        .groupby("medicine", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .head(15)
    )

    top_rev["match"] = top_rev["medicine"].apply(
        lambda m: "Searched" if is_filtered and m == search_med else "Other"
    )

    fig2 = px.bar(
        top_rev.sort_values("revenue"),
        x="revenue",
        y="medicine",
        orientation="h",
        color="match",
        color_discrete_map={
            "Searched": "#ef4444",
            "Other": "#3b82f6"
        },
        title="Top 15 Medicines by Revenue"
    )

    fig2.update_layout(
        height=460,
        xaxis_title="Revenue (₹)",
        yaxis_title="",
        showlegend=is_filtered
    )

    st.plotly_chart(fig2, use_container_width=True)

    # 3. Selling price vs units sold
    price_vol = (
        sales_analytics
        .groupby("medicine", as_index=False)
        .agg(
            selling_price=("selling_price", "mean"),
            quantity_sold=("quantity_sold", "sum"),
            revenue=("revenue", "sum"),
        )
    )

    price_vol["match"] = price_vol["medicine"].apply(
        lambda m: "Searched" if is_filtered and m == search_med else "Other"
    )

    fig3 = px.scatter(
        price_vol,
        x="selling_price",
        y="quantity_sold",
        size="revenue",
        color="match",
        color_discrete_map={
            "Searched": "#ef4444",
            "Other": "#94a3b8"
        },
        hover_name="medicine",
        title="Selling Price vs Units Sold (bubble size = revenue)"
    )

    fig3.update_layout(
        height=420,
        xaxis_title="Selling Price (₹)",
        yaxis_title="Units Sold",
        showlegend=is_filtered
    )

    st.plotly_chart(fig3, use_container_width=True)

    # 4. Profit margin %
    margin_view = (
        sales_analytics
        .groupby("medicine", as_index=False)["margin_pct"]
        .mean()
        .sort_values("margin_pct", ascending=False)
    )

    if is_filtered:
        margin_slice = margin_view[
            margin_view["medicine"] == search_med
        ]
        margin_title = f"Profit Margin{suffix}"
    else:
        margin_slice = pd.concat(
            [margin_view.head(10), margin_view.tail(10)]
        )
        margin_title = "Profit Margin — Highest & Lowest 10 Medicines"

    fig4 = px.bar(
        margin_slice.sort_values("margin_pct"),
        x="margin_pct",
        y="medicine",
        orientation="h",
        title=margin_title,
        color="margin_pct",
        color_continuous_scale="RdYlGn"
    )

    fig4.update_layout(
        height=440,
        xaxis_title="Margin (%)",
        yaxis_title=""
    )

    st.plotly_chart(fig4, use_container_width=True)


elif page == "🤖 Demand Forecast":

    st.markdown(
        '<div class="section-title">🤖 Demand Forecasting</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Main AI component: Random Forest learns recent demand patterns using lag and rolling-demand features."
    )

    med = st.selectbox(
        "Select medicine",
        medicines
    )

    horizon = st.select_slider(
        "Forecast horizon",
        options=[7, 14, 30],
        value=7
    )

    future = forecast_medicine(
        sales_df,
        med,
        horizon=horizon
    )

    hist = (
        build_daily_series(
            sales_df,
            med
        )
        .tail(60)
    )

    plot_df = pd.concat(
        [
            hist.rename(
                columns={
                    "sale_date": "date",
                    "quantity_sold": "demand"
                }
            )[
                ["date", "demand"]
            ].assign(type="Actual"),

            future.rename(
                columns={
                    "predicted_demand": "demand"
                }
            )[
                ["date", "demand"]
            ].assign(type="Forecast"),
        ],
        ignore_index=True
    )

    fig = px.line(
        plot_df,
        x="date",
        y="demand",
        color="type",
        markers=False,
        title=f"Actual vs Forecast — {med}"
    )

    fig.update_layout(
        height=430,
        xaxis_title="Date",
        yaxis_title="Units"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    d1, d2 = st.columns(2)

    avg_daily = float(
        build_daily_series(
            sales_df,
            med
        )["quantity_sold"]
        .tail(30)
        .mean()
    )

    d1.metric(
        "Avg Daily Demand",
        f"{avg_daily:.1f}"
    )

    d2.metric(
        f"{horizon}-Day Forecast",
        f"{future['predicted_demand'].sum():.0f} units"
    )

    st.markdown(
        "### 📋 Forecast table"
    )

    st.dataframe(
        future.assign(
            predicted_demand=
            future["predicted_demand"].round(1)
        ),
        use_container_width=True,
        hide_index=True
    )


elif page == "⏰ Expiry Tracking":

    st.markdown(
        '<div class="section-title">⏰ Expiry Tracking & Risk</div>',
        unsafe_allow_html=True
    )

    st.caption(
        f"Expiry status is based on days remaining and purchase value at risk. "
        f"Currently: Critical ≤ {critical_days} days, Warning {critical_days + 1}–{warning_days} days "
        f"— adjust these in the sidebar under 'Expiry Risk Thresholds'."
    )

    # UI Row for side-by-side dropdown filters
    filter_col1, filter_col2 = st.columns(2)

    selected_med = filter_col1.selectbox(
        "💊 Filter by Medicine",
        options=["All Medicines"] + medicines
    )

    selected_status = filter_col2.selectbox(
        "⚠️ Filter by Status",
        options=["All Statuses", "Safe", "Warning", "Critical", "Expired"]
    )

    # Filter data based on dropdowns
    filtered_inv = inventory_df.copy()

    if selected_med != "All Medicines":
        filtered_inv = filtered_inv[filtered_inv["medicine"] == selected_med]

    if selected_status != "All Statuses":
        filtered_inv = filtered_inv[filtered_inv["expiry_status"] == selected_status]

    # A batch with 0 units left isn't a real risk — there's nothing to
    # write off, sell, or return. Exclude dead-stock rows so counts and
    # ₹ values reflect only batches that actually matter.
    filtered_inv = filtered_inv[filtered_inv["current_stock"] > 0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Expired Batches",
        int((filtered_inv["expiry_status"] == "Expired").sum())
    )

    c2.metric(
        f"Critical ≤{critical_days}d",
        int((filtered_inv["expiry_status"] == "Critical").sum())
    )

    c3.metric(
        f"Warning {critical_days + 1}–{warning_days}d",
        int((filtered_inv["expiry_status"] == "Warning").sum())
    )

    # The extra "within warning window" restriction is only a sensible
    # DEFAULT (so "All Statuses" doesn't dump every Safe batch into a
    # table meant to highlight risk). If the user has explicitly picked
    # a status from the dropdown — including "Safe" — that choice should
    # be respected as-is, not silently re-filtered on top of.
    if selected_status == "All Statuses":
        risk_base = filtered_inv[
            filtered_inv["days_until_expiry"] <= warning_days
        ]
    else:
        risk_base = filtered_inv

    if selected_status == "All Statuses":
        value_at_risk_label = f"Value at Risk (≤{warning_days}d)"
        risk_caption = (
            "'At risk' below = everything within the Warning window you set "
            f"above (≤ {warning_days} days). Pick a specific status above to "
            "see that status only, regardless of this window."
        )
    else:
        value_at_risk_label = f"Inventory Value ({selected_status})"
        risk_caption = (
            f"Showing only batches with status '{selected_status}', as "
            "selected above — not limited to the Warning window."
        )

    c4.metric(
        value_at_risk_label,
        f"₹{format_number(risk_base['inventory_value'].sum())}"
    )

    st.caption(risk_caption)

    # Batch-level risk table, sorted by urgency first, then by value at
    # risk within the same urgency — so the highest-value risk surfaces
    # first, not just the soonest-to-expire.
    risk = (
        risk_base
        .copy()
        .sort_values(
            ["days_until_expiry", "inventory_value"],
            ascending=[True, False],
        )
    )

    risk_cols = [
        c for c in [
            "medicine",
            "batch_number",
            "current_stock",
            "purchase_price",
            "inventory_value",
            "expiry_date",
            "days_until_expiry",
            "expiry_status",
            "supplier",
        ]
        if c in risk.columns
    ]

    st.dataframe(
        risk[risk_cols],
        use_container_width=True,
        hide_index=True
    )

    chart = (
        filtered_inv
        .groupby(
            "expiry_status",
            as_index=False
        )["inventory_value"]
        .sum()
    )

    if not chart.empty:
        fig = px.bar(
            chart,
            x="expiry_status",
            y="inventory_value",
            title="Inventory Value by Expiry Status",
            text_auto=".2s" # Formats chart text similarly to our M/K format
        )

        fig.update_layout(
            height=360,
            yaxis_title="Inventory Value (₹)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown(
        "### 📌 Expiry risk by medicine"
    )

    risk_med = (
        risk_base
        .groupby(
            "medicine",
            as_index=False
        )
        .agg(
            stock_at_risk=("current_stock", "sum"),
            value_at_risk=("inventory_value", "sum")
        )
        .sort_values(
            "value_at_risk",
            ascending=False
        )
    )

    if risk_med.empty:

        if selected_status == "All Statuses":
            st.success(
                f"No medicines currently fall within the {warning_days}-day "
                "expiry-risk window based on your current filters."
            )
        else:
            st.info(
                f"No medicines currently have batches in the '{selected_status}' "
                "status based on your current filters."
            )

    else:

        fig2 = px.bar(
            risk_med.head(10)
            .sort_values("value_at_risk"),
            x="value_at_risk",
            y="medicine",
            orientation="h",
            title="Top Medicines by Expiry Risk Value"
        )

        fig2.update_layout(
            height=420,
            xaxis_title="Purchase Value at Risk (₹)",
            yaxis_title=""
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


# ----------------------------
# Footer
# ----------------------------

st.markdown("---")