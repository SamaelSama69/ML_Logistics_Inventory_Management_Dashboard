import pandas as pd
import numpy as np
from scipy import stats


class InventoryManager:

    def __init__(self, service_level: float = 0.95, overstock_multiplier: float = 3.0):
        self.service_level        = service_level
        self.overstock_multiplier = overstock_multiplier

    def calculate_safety_stock(self, avg_demand, std_demand,
                                lead_time_days, service_level=None):
        z = stats.norm.ppf(service_level if service_level is not None else self.service_level)
        return round(z * std_demand * np.sqrt(lead_time_days))

    def calculate_reorder_point(self, avg_demand, lead_time_days, safety_stock):
        return round(avg_demand * lead_time_days + safety_stock)

    def abc_classification(self, df):
        """
        Classify SKUs by total revenue contribution.

        Works with demand_data.csv (has unit_price + demand per row)
        OR inventory_data.csv (has unit_price per SKU row).
        """
        df = df.copy()

        # ── DATASET FIX ──────────────────────────────────────────────────────
        # demand_data.csv has one row per (sku_id, date).
        # inventory_data.csv has one row per (sku_id, snapshot_date).
        # Both have unit_price. We aggregate to one row per sku_id.

        if 'demand' in df.columns:
            # Called with demand_data.csv — sum demand, take median price
            sku_revenue = (
                df.groupby('sku_id', as_index=False)
                  .agg(total_demand=('demand',     'sum'),
                       unit_price  =('unit_price', 'median'))
            )
        else:
            # Called with inventory_data.csv — use avg_demand as proxy
            sku_revenue = (
                df.groupby('sku_id', as_index=False)
                  .agg(total_demand=('avg_demand', 'mean'),
                       unit_price  =('unit_price', 'median'))
            )

        sku_revenue['revenue'] = sku_revenue['total_demand'] * sku_revenue['unit_price']

        total_revenue = sku_revenue['revenue'].sum()
        if total_revenue == 0:
            raise ValueError(
                "abc_classification: total revenue is zero. "
                "Check that 'demand'/'avg_demand' and 'unit_price' contain positive values."
            )

        sku_revenue = sku_revenue.sort_values('revenue', ascending=False)
        sku_revenue['cum_pct'] = sku_revenue['revenue'].cumsum() / total_revenue

        sku_revenue['abc_class'] = pd.cut(
            sku_revenue['cum_pct'],
            bins=[0, 0.70, 0.90, 1.0],
            labels=['A', 'B', 'C'],
            include_lowest=True,
        )

        return sku_revenue.reset_index(drop=True)

    def generate_replenishment_signals(self, df, service_level=None):
        """
        Flag SKUs that need reordering.

        Expects inventory_data.csv columns:
            sku_id, avg_demand, std_demand, lead_time_days, current_stock
        Extra columns (category, warehouse, snapshot_date, unit_price) are
        passed through untouched so the caller can filter/group on them.
        """
        # ── DATASET FIX ──────────────────────────────────────────────────────
        # inventory_data.csv has snapshot_date — use the latest snapshot per
        # SKU so we don't produce duplicate signals for the same item.
        if 'snapshot_date' in df.columns:
            df = (df.sort_values('snapshot_date')
                    .groupby('sku_id', as_index=False)
                    .last()
                    .reset_index(drop=True))

        sl = service_level if service_level is not None else self.service_level
        z  = stats.norm.ppf(sl)

        df = df.copy()

        df['safety_stock']  = (z * df['std_demand'] * np.sqrt(df['lead_time_days'])).round()
        df['reorder_point'] = (df['avg_demand'] * df['lead_time_days'] + df['safety_stock']).round()

        df['status'] = np.where(
            df['current_stock'] <= df['reorder_point'], 'REORDER NOW', 'OK'
        )
        df['overstock_flag'] = df['current_stock'] > (
            df['reorder_point'] * self.overstock_multiplier
        )

        # Return core signal columns + any extra context columns that exist
        core_cols = ['sku_id', 'current_stock', 'reorder_point',
                     'safety_stock', 'status', 'overstock_flag']
        extra_cols = [c for c in ['category', 'warehouse', 'unit_price']
                      if c in df.columns]

        return df[core_cols + extra_cols].reset_index(drop=True)
