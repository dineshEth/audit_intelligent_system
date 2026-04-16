from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils.dates import utcnow_iso


class ChartBuilder:
    def __init__(self, settings) -> None:
        self.settings = settings

    def build_transaction_charts(self, dataframe: pd.DataFrame, stem: str) -> List[str]:
        if dataframe.empty:
            return []

        paths: List[str] = []
        df = dataframe.copy()
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        timestamp = utcnow_iso()

        if "CATEGORY" in df.columns:
            cat_df = (
                df.assign(AMOUNT=df["DEBIT"].where(df["DEBIT"] > 0, df["CREDIT"]))
                .groupby("CATEGORY")["AMOUNT"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            if not cat_df.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=cat_df, x="CATEGORY", y="AMOUNT", hue="CATEGORY", dodge=False, legend=False, ax=ax, palette="deep")
                ax.set_title("Transaction Volume by Category")
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                chart_path = self.settings.charts_dir / f"{stem}_category_{timestamp}.png"
                fig.savefig(chart_path, dpi=150)
                plt.close(fig)
                paths.append(str(chart_path))

        if "BALANCE" in df.columns:
            bal_df = df[["DATE", "BALANCE"]].dropna()
            if not bal_df.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=bal_df, x="DATE", y="BALANCE", marker="o", ax=ax, color="tab:blue")
                ax.set_title("Balance Trend")
                fig.autofmt_xdate()
                fig.tight_layout()
                chart_path = self.settings.charts_dir / f"{stem}_balance_{timestamp}.png"
                fig.savefig(chart_path, dpi=150)
                plt.close(fig)
                paths.append(str(chart_path))

        return paths
