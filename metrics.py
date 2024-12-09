import math
from enum import Enum

import polars as pl

class AggregationType(Enum):
    DAY = "day"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class ColumnNames(str, Enum):
    DATE = "date"
    CLOSE = "close"
    RETURN = "return"
    CUMULATIVE_RETURN = "cumulative_return"

class Metrics:
    def __init__(self, data: pl.DataFrame):
        self.data = data

    @property
    def numeric_columns(self):
        """Return only the float columns, leaving out date or other non-numeric columns."""
        return [col for col, dtype in zip(self.data.columns, self.data.dtypes) if dtype == pl.Float64]

    @property
    def date_column(self):
        """Return the date column, assuming it is of type pl.Date or pl.Datetime."""
        return [col for col, dtype in zip(self.data.columns, self.data.dtypes) if dtype in (pl.Date, pl.Datetime)]

    def annualization_faactor(self, convert_to: AggregationType) -> float:
        if convert_to == AggregationType.DAY:
            return 252
        elif convert_to == AggregationType.WEEKLY:
            return 52
        elif convert_to == AggregationType.MONTHLY:
            return 12
        elif convert_to == AggregationType.QUARTERLY:
            return 4
        elif convert_to == AggregationType.YEARLY:
            return 1
        else:
            raise ValueError(
                f"Aggregation type {convert_to} not supported. Choose from {AggregationType.__members__}"
            )

    def simple_returns(self) -> pl.DataFrame:
        numeric_df = self.data.select(self.numeric_columns)
        returns = numeric_df.select([
            pl.col("*"),
            (pl.col(ColumnNames.CLOSE) / pl.col(ColumnNames.CLOSE).shift(1) - 1).alias(ColumnNames.RETURN)
        ])
        return self.data.select(ColumnNames.DATE).with_columns(returns)

    def cumulative_returns(self, simple_returns: pl.DataFrame) -> pl.DataFrame:
        returns = simple_returns.select([
            pl.col(ColumnNames.DATE),
            pl.col(ColumnNames.RETURN),
            (1 + pl.col(ColumnNames.RETURN)).cum_prod().alias(ColumnNames.CUMULATIVE_RETURN)
        ])
        return returns

    def cumulative_return_final(self, cumulative_returns: pl.DataFrame) -> float:
        result = cumulative_returns.select([
            (1 + pl.col(ColumnNames.CUMULATIVE_RETURN)).product().alias("cumulative_return_final")
        ])
        return result.get_column(ColumnNames.CUMULATIVE_RETURN)[0]


    def aggregate_returns(self, convert_to: AggregationType) -> pl.DataFrame:
        _group_by_key = "group_by_key"
        returns = self.simple_returns()
        # if convert_to == AggregationType.WEEKLY:
        #     grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
        if convert_to == AggregationType.MONTHLY:
            return returns.with_columns(pl.col(ColumnNames.DATE).dt.strftime("%Y-%m").alias(_group_by_key)).group_by(_group_by_key)
        # elif convert_to == AggregationType.QUARTERLY:
        #     grouping = [lambda x: x.year, lambda x: int(math.ceil(x.month / 3.))]
        elif convert_to == AggregationType.YEARLY:
            return returns.with_columns(pl.col(ColumnNames.DATE).dt.strftime("%Y").alias(_group_by_key)).group_by(_group_by_key)
        else:
            raise ValueError(
                f"Aggregation type {convert_to} not supported. Choose from {AggregationType.__members__}"
            )

    def max_drawdown(self, simple_returns: pl.DataFrame) -> float:
        cumulative_returns = self.cumulative_returns(simple_returns)

        out = (
            cumulative_returns
            .with_columns(pl.col("cumulative_return").cum_max().alias("max_return"))
            .with_columns((pl.col("cumulative_return") / pl.col("max_return") - 1).alias("drawdown"))
            .select(pl.col("drawdown").max().alias("max_drawdown"))
            .item()
        )
        return out

    def compound_annual_growth_rate(self, simple_returns: pl.DataFrame,
                                    period=AggregationType.DAY):
        ann_factor = self.annualization_faactor(period)
        num_years = simple_returns.count().item() / ann_factor
        ending_value = self.cumulative_return_final(self.cumulative_returns(simple_returns))
        return ending_value ** (1 / num_years) - 1

    def annual_volatility(self, simple_returns: pl.DataFrame, period=AggregationType.DAY):
        ann_factor = self.annualization_faactor(period)
        return simple_returns.select([
            pl.col(ColumnNames.RETURN).std().alias("annual_volatility")
        ]).get_column("annual_volatility")[0] * math.sqrt(ann_factor)

    def calmar_ratio(self, simple_returns: pl.DataFrame, period=AggregationType.DAY):
        cagr = self.compound_annual_growth_rate(simple_returns, period)
        max_drawdown = abs(self.max_drawdown(simple_returns))
        return cagr / max_drawdown

    # fixme
    def omega_ratio(self, simple_returns: pl.DataFrame, risk_free_rate: float, period=AggregationType.DAY):
        ann_factor = self.annualization_faactor(period)
        return (simple_returns.select([
            (pl.col(ColumnNames.RETURN) - risk_free_rate).filter(pl.col(ColumnNames.RETURN) < risk_free_rate).sum().alias("omega_ratio")
        ]).get_column("omega_ratio")[0] / ann_factor) / self.annual_volatility(simple_returns, period)

    def sharpe_ratio(self, simple_returns: pl.DataFrame, risk_free_rate: float, period=AggregationType.DAY):
        ann_factor = self.annualization_faactor(period)
        return (simple_returns.select([
            (pl.col("simple_return") - risk_free_rate).sum().alias("excess_return")
        ]).get_column("excess_return")[0] / ann_factor) / self.annual_volatility(simple_returns, period)

    def sortino_ratio(self, simple_returns: pl.DataFrame, risk_free_rate: float, period=AggregationType.DAY):
        ann_factor = self.annualization_faactor(period)
        return (simple_returns.select([
            (pl.col("simple_return") - risk_free_rate).filter(pl.col("simple_return") < risk_free_rate).sum().alias("downside_return")
        ]).get_column("downside_return")[0] / ann_factor) / self.annual_volatility(simple_returns, period)