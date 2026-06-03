"""
Store Intelligence Dashboard — Streamlit Application

A real-time dashboard that polls the FastAPI backend and displays:
- Store-level metric cards (visitors, conversion, dwell, queue, abandonment)
- Zone heatmap with colour-coded intensity grid
- Active anomaly alerts with severity badges
- Conversion funnel as a horizontal bar chart

Auto-refreshes every 5 seconds. Handles API unreachable gracefully.

Configure via environment variable:
    API_URL  (default: http://localhost:8000 for local, http://api:8000 for Docker)
"""

import os
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.environ.get("API_URL", "http://localhost:8000")
REFRESH_INTERVAL = 5  # seconds

# Available stores — fetched dynamically, with fallback
DEFAULT_STORES = ["ST1008"]

# ---------------------------------------------------------------------------
# Page Config & Theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Store Intelligence Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme styling and metric cards
st.markdown(
    """
    <style>
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 5px;
        border: 1px solid #3d3d5c;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-card h3 {
        color: #a0a0c0;
        font-size: 14px;
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    .metric-card .subtitle {
        color: #7a7a9a;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Severity badges */
    .badge-critical {
        background-color: #ff4444;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
    }
    .badge-warn {
        background-color: #ffaa00;
        color: #1a1a2e;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
    }
    .badge-info {
        background-color: #4488ff;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
    }

    /* Anomaly card */
    .anomaly-card {
        background: #2a1a1a;
        border-left: 4px solid #ff4444;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .anomaly-card.warn {
        background: #2a2a1a;
        border-left-color: #ffaa00;
    }
    .anomaly-card.info {
        background: #1a1a2a;
        border-left-color: #4488ff;
    }

    /* Connection status */
    .connecting {
        text-align: center;
        padding: 40px;
        color: #7a7a9a;
        font-size: 18px;
    }

    /* Heatmap cell styling */
    .heatmap-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 8px;
        padding: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------


class APIClient:
    """Lightweight wrapper around the FastAPI backend endpoints."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, path: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """Make a GET request, returning None on failure."""
        try:
            resp = self.session.get(f"{self.base_url}{path}", timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout):
            return None
        except requests.HTTPError:
            return None

    def health(self) -> Optional[Dict[str, Any]]:
        return self._get("/health")

    def stores(self) -> List[str]:
        """Fetch available store IDs. Falls back to defaults on failure."""
        data = self._get("/stores")
        if data and isinstance(data, list):
            return data
        if data and "stores" in data:
            return data["stores"]
        return DEFAULT_STORES

    def metrics(self, store_id: str, date: str = None) -> Optional[Dict[str, Any]]:
        qs = f"?date={date}" if date else ""
        return self._get(f"/stores/{store_id}/metrics{qs}")

    def heatmap(self, store_id: str, date: str = None) -> Optional[Dict[str, Any]]:
        qs = f"?date={date}" if date else ""
        return self._get(f"/stores/{store_id}/heatmap{qs}")

    def funnel(self, store_id: str, date: str = None) -> Optional[Dict[str, Any]]:
        qs = f"?date={date}" if date else ""
        return self._get(f"/stores/{store_id}/funnel{qs}")

    def anomalies(self, store_id: str, date: str = None) -> Optional[Dict[str, Any]]:
        qs = f"?date={date}" if date else ""
        return self._get(f"/stores/{store_id}/anomalies{qs}")


api = APIClient(API_URL)


# ---------------------------------------------------------------------------
# Helper: Render a metric card
# ---------------------------------------------------------------------------


def metric_card(title: str, value: str, subtitle: str = ""):
    """Render a styled metric card using HTML."""
    subtitle_html = f'<p class="subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>{title}</h3>
            <p class="value">{value}</p>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helper: Severity badge
# ---------------------------------------------------------------------------


def severity_badge(severity: str) -> str:
    """Return an HTML badge span for the given severity level."""
    css_class = {
        "CRITICAL": "badge-critical",
        "WARN": "badge-warn",
        "INFO": "badge-info",
    }.get(severity.upper(), "badge-info")
    return f'<span class="{css_class}">{severity}</span>'


# ---------------------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------------------


def main():
    # --- Sidebar: Store selector + status ---
    with st.sidebar:
        st.title("🏪 Store Intelligence")
        st.caption("Real-time retail analytics dashboard")
        st.divider()

        # Check API health
        health = api.health()
        if health and health.get("status") == "ok":
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Unreachable")
            st.caption(f"Trying: {API_URL}")

        st.divider()

        # Store selector
        stores = api.stores()
        selected_store = st.selectbox(
            "Select Store",
            options=stores,
            index=0,
            help="Choose a store to view analytics",
        )

        selected_date = st.date_input(
            "Select Date",
            help="View analytics for a specific date",
        )
        date_str = selected_date.isoformat()

        st.divider()

        # Auto-refresh toggle
        auto_refresh = st.toggle("Auto-Refresh (5s)", value=True)

        st.divider()
        st.caption(f"API: {API_URL}")
        st.caption(f"Refresh: {REFRESH_INTERVAL}s")

    # --- Main content ---
    st.title(f"📊 {selected_store}")

    # Check connectivity
    if health is None:
        st.markdown(
            '<div class="connecting">⏳ Connecting to API server...</div>',
            unsafe_allow_html=True,
        )
        st.info(
            f"Attempting to connect to **{API_URL}**. "
            "Make sure the API server is running."
        )
        if auto_refresh:
            time.sleep(REFRESH_INTERVAL)
            st.rerun()
        return

    # -----------------------------------------------------------------------
    # 1. METRIC CARDS
    # -----------------------------------------------------------------------
    metrics_data = api.metrics(selected_store, date_str)

    st.subheader("📈 Key Metrics")

    if metrics_data:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            metric_card(
                "Unique Visitors",
                str(metrics_data.get("unique_visitors", 0)),
                "today",
            )

        with col2:
            rate = metrics_data.get("conversion_rate", 0.0)
            metric_card(
                "Conversion Rate",
                f"{rate * 100:.1f}%",
                f"{rate:.3f} raw",
            )

        with col3:
            # avg_dwell_per_zone is a dict {zone_id: avg_dwell_ms}
            dwell_dict = metrics_data.get("avg_dwell_per_zone", {})
            if dwell_dict:
                avg_dwell_ms = sum(dwell_dict.values()) / len(dwell_dict)
            else:
                avg_dwell_ms = 0
            dwell_sec = avg_dwell_ms / 1000
            minutes = int(dwell_sec // 60)
            seconds = int(dwell_sec % 60)
            metric_card(
                "Avg Dwell Time",
                f"{minutes}m {seconds}s",
                f"{avg_dwell_ms:.0f} ms avg across zones",
            )

        with col4:
            queue = metrics_data.get("current_queue_depth", 0)
            metric_card(
                "Queue Depth",
                str(queue),
                "people in billing queue",
            )

        with col5:
            abandon = metrics_data.get("abandonment_rate", 0.0)
            metric_card(
                "Abandonment Rate",
                f"{abandon * 100:.1f}%",
                "billing drop-off",
            )
    else:
        st.warning("No metrics data available for this store.")

    st.divider()

    # -----------------------------------------------------------------------
    # 2. ZONE HEATMAP + ANOMALIES (side by side)
    # -----------------------------------------------------------------------
    heatmap_col, anomaly_col = st.columns([3, 2])

    # --- Heatmap ---
    with heatmap_col:
        st.subheader("🗺️ Zone Heatmap")
        heatmap_data = api.heatmap(selected_store, date_str)

        if heatmap_data:
            # API returns a list of HeatmapCell dicts directly (not wrapped)
            zones = heatmap_data if isinstance(heatmap_data, list) else heatmap_data.get("zones", [])

            # Determine confidence from the first cell
            confidence = "unknown"
            if zones and isinstance(zones[0], dict):
                confidence = zones[0].get("data_confidence", "unknown")

            if confidence == "low":
                st.caption("⚠️ Data confidence: **LOW** (< 20 sessions)")
            else:
                st.caption(f"✅ Data confidence: **{confidence.upper()}**")

            if zones and isinstance(zones[0], dict):
                zone_names = [z.get("zone_id", z.get("zone", f"Zone {i}")) for i, z in enumerate(zones)]
                zone_scores = [z.get("normalized_score", z.get("score", 0)) for z in zones]
            elif zones:
                zone_names = [f"Zone {i+1}" for i in range(len(zones))]
                zone_scores = zones
            else:
                zone_names = []
                zone_scores = []

            if zone_names:
                # Create a heatmap using Plotly
                df_heatmap = pd.DataFrame(
                    {"Zone": zone_names, "Activity Score": zone_scores}
                )
                df_heatmap = df_heatmap.sort_values("Activity Score", ascending=True)

                fig_heatmap = go.Figure(
                    go.Bar(
                        x=df_heatmap["Activity Score"],
                        y=df_heatmap["Zone"],
                        orientation="h",
                        marker=dict(
                            color=df_heatmap["Activity Score"],
                            colorscale=[
                                [0, "#1a1a2e"],
                                [0.3, "#16213e"],
                                [0.5, "#e94560"],
                                [0.7, "#ff6b6b"],
                                [1, "#ffd93d"],
                            ],
                            line=dict(width=0),
                        ),
                        text=df_heatmap["Activity Score"].apply(lambda x: f"{x:.0f}"),
                        textposition="auto",
                        textfont=dict(color="white", size=14),
                    )
                )

                fig_heatmap.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#a0a0c0"),
                    xaxis=dict(
                        title="Activity Score (0–100)",
                        range=[0, 105],
                        gridcolor="#2d2d44",
                    ),
                    yaxis=dict(title="", gridcolor="#2d2d44"),
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=40),
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No zone data available.")
        else:
            st.warning("Heatmap data unavailable.")

    # --- Active Anomalies ---
    with anomaly_col:
        st.subheader("🚨 Active Anomalies")
        anomaly_data = api.anomalies(selected_store, date_str)

        if anomaly_data:
            anomalies = anomaly_data.get("anomalies", [])

            if not anomalies:
                st.markdown(
                    """
                    <div style="text-align: center; padding: 40px 20px; color: #4a9;">
                        <div style="font-size: 48px;">✅</div>
                        <p style="font-size: 16px; margin-top: 12px;">No anomalies detected</p>
                        <p style="font-size: 12px; color: #7a7a9a;">All zones operating normally</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                for anomaly in anomalies:
                    atype = anomaly.get("type", "UNKNOWN")
                    severity = anomaly.get("severity", "INFO")
                    zone = anomaly.get("zone_id", "—")
                    detail = anomaly.get("detail", "No reason provided.")
                    action = anomaly.get("suggested_action", "No action specified.")

                    # CSS class for card border color
                    card_class = {
                        "CRITICAL": "",
                        "WARN": "warn",
                        "INFO": "info",
                    }.get(severity, "info")

                    st.markdown(
                        f"""
                        <div class="anomaly-card {card_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong style="color: #e0e0e0;">{atype}</strong>
                                {severity_badge(severity)}
                            </div>
                            <p style="color: #a0a0a0; margin: 6px 0 2px 0; font-size: 13px;">
                                📍 Zone: {zone}
                            </p>
                            <p style="color: #ffffff; margin: 4px 0 6px 0; font-size: 14px; font-weight: 500;">
                                {detail}
                            </p>
                            <p style="color: #c0c0e0; margin: 4px 0; font-size: 13px;">
                                💡 {action}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("Anomaly data unavailable.")

    st.divider()

    # -----------------------------------------------------------------------
    # 3. CONVERSION FUNNEL
    # -----------------------------------------------------------------------
    st.subheader("🔄 Conversion Funnel")
    funnel_data = api.funnel(selected_store, date_str)

    if funnel_data:
        stages = funnel_data.get("stages", funnel_data.get("funnel", []))

        if stages:
            if isinstance(stages[0], dict):
                stage_names = [s["stage"] for s in stages]
                stage_counts = [s["count"] for s in stages]
            else:
                stage_names = ["Entry", "Zone Visit", "Billing Queue", "Purchase"]
                stage_counts = stages[:4]

            # Compute drop-off percentages
            drop_offs = []
            for i in range(len(stage_counts)):
                if i == 0:
                    drop_offs.append(0)
                else:
                    if stage_counts[i - 1] > 0:
                        drop_pct = (
                            (stage_counts[i - 1] - stage_counts[i])
                            / stage_counts[i - 1]
                            * 100
                        )
                    else:
                        drop_pct = 0
                    drop_offs.append(drop_pct)

            # Funnel chart using Plotly
            colors = ["#667eea", "#764ba2", "#f093fb", "#f5576c"]
            while len(colors) < len(stage_names):
                colors.append("#a0a0c0")

            fig_funnel = go.Figure()

            fig_funnel.add_trace(
                go.Funnel(
                    y=stage_names,
                    x=stage_counts,
                    textinfo="value+percent initial+percent previous",
                    marker=dict(
                        color=colors[: len(stage_names)],
                        line=dict(width=1, color="#1a1a2e"),
                    ),
                    connector=dict(line=dict(color="#3d3d5c", width=1)),
                    textfont=dict(color="white", size=13),
                )
            )

            fig_funnel.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a0a0c0"),
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                funnelmode="stack",
            )

            st.plotly_chart(fig_funnel, use_container_width=True)

            # Drop-off summary table
            with st.expander("📋 Funnel Details"):
                df_funnel = pd.DataFrame(
                    {
                        "Stage": stage_names,
                        "Count": stage_counts,
                        "Drop-off %": [f"{d:.1f}%" for d in drop_offs],
                    }
                )
                st.dataframe(
                    df_funnel,
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No funnel data available.")
    else:
        st.warning("Funnel data unavailable.")

    # -----------------------------------------------------------------------
    # 4. AUTO-REFRESH
    # -----------------------------------------------------------------------
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
else:
    main()
