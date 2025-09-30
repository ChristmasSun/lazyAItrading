from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import os
from datetime import datetime, timezone


@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0


class Portfolio:
    def __init__(self, cash: float = 10_000.0, fee_rate: float = 0.0, fee_fixed: float = 0.0, slippage_bps: float = 0.0, trade_log_path: Optional[str] = None):
        self.cash = cash
        self.positions: Dict[str, Position] = {}
        # simple cost model
        self.fee_rate = max(0.0, float(fee_rate))  # fraction of notional, e.g. 0.0005 = 5 bps
        self.fee_fixed = max(0.0, float(fee_fixed))  # per-order fixed fee
        self.slippage_bps = max(0.0, float(slippage_bps))  # bps applied to price
        self.trade_log_path = trade_log_path

    def _log_trade(self, side: str, symbol: str, qty: float, exec_px: float, gross: float, net_cash_delta: float) -> None:
        if not self.trade_log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.trade_log_path), exist_ok=True)
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "side": side,
                "symbol": symbol,
                "qty": qty,
                "exec_px": exec_px,
                "gross_notional": gross,
                "net_cash_delta": net_cash_delta,
                "fee_rate": self.fee_rate,
                "fee_fixed": self.fee_fixed,
                "slippage_bps": self.slippage_bps,
                "cash_after": self.cash,
                "position_qty_after": self.positions.get(symbol, Position()).qty,
            }
            with open(self.trade_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def value(self, prices: Dict[str, float]) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            px = prices.get(sym, pos.avg_price)
            eq += pos.qty * px
        return eq

    def buy(self, symbol: str, price: float, amount_cash: float) -> None:
        if amount_cash <= 0 or price <= 0:
            return
        spend = min(self.cash, amount_cash)
        # effective execution price with slippage
        exec_px = price * (1.0 + self.slippage_bps / 10_000.0)
        # total cost includes fees on notional plus fixed
        denom = exec_px * (1.0 + self.fee_rate)
        if denom <= 0:
            return
        # ensure we have enough to cover fixed fee first
        if spend <= self.fee_fixed:
            return
        qty = (spend - self.fee_fixed) / denom
        self.cash -= spend
        pos = self.positions.get(symbol, Position())
        new_qty = pos.qty + qty
        pos.avg_price = (pos.avg_price * pos.qty + price * qty) / new_qty if new_qty > 0 else 0.0
        pos.qty = new_qty
        self.positions[symbol] = pos
        self._log_trade("BUY", symbol, qty, exec_px, gross=qty * exec_px, net_cash_delta=-spend)

    def sell(self, symbol: str, price: float, qty: float) -> None:
        if qty <= 0 or price <= 0:
            return
        pos = self.positions.get(symbol)
        if not pos or pos.qty <= 0:
            return
        sell_qty = min(pos.qty, qty)
        # effective execution price with slippage against us
        exec_px = price * (1.0 - self.slippage_bps / 10_000.0)
        gross = sell_qty * exec_px
        proceeds = max(0.0, gross * (1.0 - self.fee_rate) - self.fee_fixed)
        self.cash += proceeds
        pos.qty -= sell_qty
        if pos.qty <= 0:
            pos.avg_price = 0.0
        self.positions[symbol] = pos
        self._log_trade("SELL", symbol, sell_qty, exec_px, gross=gross, net_cash_delta=proceeds)
