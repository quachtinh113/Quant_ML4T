"""Contract specifications for futures and other derivatives.

This module provides contract specifications needed for proper P&L calculation
and margin requirements in backtesting and live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .asset_class import AssetClass

if TYPE_CHECKING:
    pass


@dataclass
class ContractSpec:
    """Contract specification for futures and other derivatives.

    Defines characteristics that affect P&L calculation and margin:
    - Futures: multiplier varies (ES=$50, CL=$1000, etc.)
    - Equities: multiplier=1, tick_size=0.01
    - Forex: pip value varies by pair and account currency

    Example:
        # E-mini S&P 500 futures
        es_spec = ContractSpec(
            symbol="ES",
            asset_class=AssetClass.FUTURE,
            multiplier=50.0,      # $50 per point
            tick_size=0.25,       # Minimum price move
            margin=15000.0,       # Initial margin per contract
            exchange="CME",
        )

        # Apple stock (default equity spec)
        aapl_spec = ContractSpec(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            # multiplier=1.0 (default)
            # tick_size=0.01 (default)
        )
    """

    symbol: str
    asset_class: AssetClass = AssetClass.EQUITY
    multiplier: float = 1.0  # Point value ($ per point move)
    tick_size: float = 0.01  # Minimum price increment
    margin: float | None = None  # Initial margin per contract
    exchange: str | None = None  # Exchange (CME, CBOT, NYMEX, etc.)
    currency: str = "USD"
    name: str | None = None  # Human-readable name

    @property
    def tick_value(self) -> float:
        """Dollar value of one tick move."""
        return self.multiplier * self.tick_size


# Common futures contract specifications
# Source: CME Group, ICE, verified against IBKR contract details
FUTURES_REGISTRY: dict[str, ContractSpec] = {
    # === Equity Index Futures (CME) ===
    "ES": ContractSpec(
        symbol="ES",
        asset_class=AssetClass.FUTURE,
        multiplier=50.0,
        tick_size=0.25,
        margin=15000.0,
        exchange="CME",
        name="E-mini S&P 500",
    ),
    "MES": ContractSpec(
        symbol="MES",
        asset_class=AssetClass.FUTURE,
        multiplier=5.0,
        tick_size=0.25,
        margin=1500.0,
        exchange="CME",
        name="Micro E-mini S&P 500",
    ),
    "NQ": ContractSpec(
        symbol="NQ",
        asset_class=AssetClass.FUTURE,
        multiplier=20.0,
        tick_size=0.25,
        margin=18000.0,
        exchange="CME",
        name="E-mini Nasdaq-100",
    ),
    "MNQ": ContractSpec(
        symbol="MNQ",
        asset_class=AssetClass.FUTURE,
        multiplier=2.0,
        tick_size=0.25,
        margin=1800.0,
        exchange="CME",
        name="Micro E-mini Nasdaq-100",
    ),
    "RTY": ContractSpec(
        symbol="RTY",
        asset_class=AssetClass.FUTURE,
        multiplier=50.0,
        tick_size=0.10,
        margin=8000.0,
        exchange="CME",
        name="E-mini Russell 2000",
    ),
    "YM": ContractSpec(
        symbol="YM",
        asset_class=AssetClass.FUTURE,
        multiplier=5.0,
        tick_size=1.0,
        margin=10000.0,
        exchange="CBOT",
        name="E-mini Dow ($5)",
    ),
    # === Energy Futures (NYMEX) ===
    "CL": ContractSpec(
        symbol="CL",
        asset_class=AssetClass.FUTURE,
        multiplier=1000.0,
        tick_size=0.01,
        margin=7000.0,
        exchange="NYMEX",
        name="Crude Oil (WTI)",
    ),
    "QM": ContractSpec(
        symbol="QM",
        asset_class=AssetClass.FUTURE,
        multiplier=500.0,
        tick_size=0.025,
        margin=3500.0,
        exchange="NYMEX",
        name="E-mini Crude Oil",
    ),
    "NG": ContractSpec(
        symbol="NG",
        asset_class=AssetClass.FUTURE,
        multiplier=10000.0,
        tick_size=0.001,
        margin=5000.0,
        exchange="NYMEX",
        name="Natural Gas",
    ),
    "RB": ContractSpec(
        symbol="RB",
        asset_class=AssetClass.FUTURE,
        multiplier=42000.0,
        tick_size=0.0001,
        margin=7000.0,
        exchange="NYMEX",
        name="RBOB Gasoline",
    ),
    "HO": ContractSpec(
        symbol="HO",
        asset_class=AssetClass.FUTURE,
        multiplier=42000.0,
        tick_size=0.0001,
        margin=7000.0,
        exchange="NYMEX",
        name="Heating Oil",
    ),
    # === Metals Futures (COMEX) ===
    "GC": ContractSpec(
        symbol="GC",
        asset_class=AssetClass.FUTURE,
        multiplier=100.0,
        tick_size=0.10,
        margin=10000.0,
        exchange="COMEX",
        name="Gold",
    ),
    "MGC": ContractSpec(
        symbol="MGC",
        asset_class=AssetClass.FUTURE,
        multiplier=10.0,
        tick_size=0.10,
        margin=1000.0,
        exchange="COMEX",
        name="Micro Gold",
    ),
    "SI": ContractSpec(
        symbol="SI",
        asset_class=AssetClass.FUTURE,
        multiplier=5000.0,
        tick_size=0.005,
        margin=12000.0,
        exchange="COMEX",
        name="Silver",
    ),
    "HG": ContractSpec(
        symbol="HG",
        asset_class=AssetClass.FUTURE,
        multiplier=25000.0,
        tick_size=0.0005,
        margin=6000.0,
        exchange="COMEX",
        name="Copper",
    ),
    "PL": ContractSpec(
        symbol="PL",
        asset_class=AssetClass.FUTURE,
        multiplier=50.0,
        tick_size=0.10,
        margin=5000.0,
        exchange="NYMEX",
        name="Platinum",
    ),
    # === Treasury Futures (CBOT) ===
    "ZB": ContractSpec(
        symbol="ZB",
        asset_class=AssetClass.FUTURE,
        multiplier=1000.0,
        tick_size=0.03125,  # 1/32
        margin=4000.0,
        exchange="CBOT",
        name="30-Year T-Bond",
    ),
    "ZN": ContractSpec(
        symbol="ZN",
        asset_class=AssetClass.FUTURE,
        multiplier=1000.0,
        tick_size=0.015625,  # 1/64
        margin=2500.0,
        exchange="CBOT",
        name="10-Year T-Note",
    ),
    "ZF": ContractSpec(
        symbol="ZF",
        asset_class=AssetClass.FUTURE,
        multiplier=1000.0,
        tick_size=0.0078125,  # 1/128
        margin=1500.0,
        exchange="CBOT",
        name="5-Year T-Note",
    ),
    "ZT": ContractSpec(
        symbol="ZT",
        asset_class=AssetClass.FUTURE,
        multiplier=2000.0,
        tick_size=0.00390625,  # 1/256
        margin=1000.0,
        exchange="CBOT",
        name="2-Year T-Note",
    ),
    # === Agricultural Futures (CBOT) ===
    "ZC": ContractSpec(
        symbol="ZC",
        asset_class=AssetClass.FUTURE,
        multiplier=50.0,
        tick_size=0.25,
        margin=2000.0,
        exchange="CBOT",
        name="Corn",
    ),
    "ZS": ContractSpec(
        symbol="ZS",
        asset_class=AssetClass.FUTURE,
        multiplier=50.0,
        tick_size=0.25,
        margin=3000.0,
        exchange="CBOT",
        name="Soybeans",
    ),
    "ZW": ContractSpec(
        symbol="ZW",
        asset_class=AssetClass.FUTURE,
        multiplier=50.0,
        tick_size=0.25,
        margin=2500.0,
        exchange="CBOT",
        name="Wheat",
    ),
    "ZL": ContractSpec(
        symbol="ZL",
        asset_class=AssetClass.FUTURE,
        multiplier=600.0,
        tick_size=0.01,
        margin=2000.0,
        exchange="CBOT",
        name="Soybean Oil",
    ),
    "ZM": ContractSpec(
        symbol="ZM",
        asset_class=AssetClass.FUTURE,
        multiplier=100.0,
        tick_size=0.10,
        margin=2500.0,
        exchange="CBOT",
        name="Soybean Meal",
    ),
    # === Currency Futures (CME) ===
    "6E": ContractSpec(
        symbol="6E",
        asset_class=AssetClass.FUTURE,
        multiplier=125000.0,
        tick_size=0.00005,
        margin=3000.0,
        exchange="CME",
        name="Euro FX",
    ),
    "6J": ContractSpec(
        symbol="6J",
        asset_class=AssetClass.FUTURE,
        multiplier=12500000.0,
        tick_size=0.0000005,
        margin=4000.0,
        exchange="CME",
        name="Japanese Yen",
    ),
    "6B": ContractSpec(
        symbol="6B",
        asset_class=AssetClass.FUTURE,
        multiplier=62500.0,
        tick_size=0.0001,
        margin=3500.0,
        exchange="CME",
        name="British Pound",
    ),
    "6A": ContractSpec(
        symbol="6A",
        asset_class=AssetClass.FUTURE,
        multiplier=100000.0,
        tick_size=0.0001,
        margin=2500.0,
        exchange="CME",
        name="Australian Dollar",
    ),
    "6C": ContractSpec(
        symbol="6C",
        asset_class=AssetClass.FUTURE,
        multiplier=100000.0,
        tick_size=0.00005,
        margin=2000.0,
        exchange="CME",
        name="Canadian Dollar",
    ),
    # === Volatility Futures (CFE) ===
    "VX": ContractSpec(
        symbol="VX",
        asset_class=AssetClass.FUTURE,
        multiplier=1000.0,
        tick_size=0.05,
        margin=10000.0,
        exchange="CFE",
        name="VIX Futures",
    ),
    # === Crypto Futures (CME) ===
    "BTC": ContractSpec(
        symbol="BTC",
        asset_class=AssetClass.FUTURE,
        multiplier=5.0,
        tick_size=5.0,
        margin=50000.0,
        exchange="CME",
        name="Bitcoin Futures",
    ),
    "MBT": ContractSpec(
        symbol="MBT",
        asset_class=AssetClass.FUTURE,
        multiplier=0.1,
        tick_size=5.0,
        margin=5000.0,
        exchange="CME",
        name="Micro Bitcoin",
    ),
    "ETH": ContractSpec(
        symbol="ETH",
        asset_class=AssetClass.FUTURE,
        multiplier=50.0,
        tick_size=0.25,
        margin=20000.0,
        exchange="CME",
        name="Ether Futures",
    ),
}


def get_contract_spec(symbol: str) -> ContractSpec | None:
    """Get contract specification for a symbol.

    Args:
        symbol: Contract symbol (e.g., "ES", "CL", "GC")

    Returns:
        ContractSpec if found, None otherwise
    """
    return FUTURES_REGISTRY.get(symbol)


def load_contract_specs(
    symbols: list[str] | None = None,
    include_all: bool = False,
) -> dict[str, ContractSpec]:
    """Load contract specifications.

    Args:
        symbols: Specific symbols to load. If None and include_all=False, returns empty dict.
        include_all: If True, returns all registered contracts.

    Returns:
        Dictionary of symbol -> ContractSpec
    """
    if include_all:
        return dict(FUTURES_REGISTRY)

    if symbols is None:
        return {}

    return {s: FUTURES_REGISTRY[s] for s in symbols if s in FUTURES_REGISTRY}


def register_contract_spec(spec: ContractSpec) -> None:
    """Register a custom contract specification.

    Args:
        spec: ContractSpec to register
    """
    FUTURES_REGISTRY[spec.symbol] = spec
