import os
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from tinkoff.invest import Client, InstrumentIdType, MoneyValue, OrderDirection, OrderType
from tinkoff.invest.sandbox.client import SandboxClient


TINKOFF_SANDBOX_TOKEN = os.getenv("TINKOFF_SANDBOX_TOKEN", "").strip()
TINKOFF_SANDBOX_ACCOUNT_ID = os.getenv("TINKOFF_SANDBOX_ACCOUNT_ID", "").strip()
TINKOFF_SANDBOX_ENABLED = os.getenv("TINKOFF_SANDBOX_ENABLED", "true").lower() == "true"


@dataclass
class SandboxPortfolioSnapshot:
    account_id: str
    total_amount_portfolio: str
    expected_yield: str
    total_amount_shares: str
    total_amount_bonds: str
    total_amount_etf: str
    total_amount_currencies: str
    total_amount_futures: str
    positions: list[dict[str, Any]]
    money: list[dict[str, str]]
    source: str = "tinkoff_sandbox"

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "source": self.source,
            "total_amount_portfolio": self.total_amount_portfolio,
            "expected_yield": self.expected_yield,
            "total_amount_shares": self.total_amount_shares,
            "total_amount_bonds": self.total_amount_bonds,
            "total_amount_etf": self.total_amount_etf,
            "total_amount_currencies": self.total_amount_currencies,
            "total_amount_futures": self.total_amount_futures,
            "positions": self.positions,
            "money": self.money,
        }

    def to_prompt_text(self) -> str:
        if not self.positions and not self.money:
            return (
                f"Sandbox account_id={self.account_id}. "
                "The account has no positions or cash balances."
            )

        chunks = [
            f"Sandbox account_id={self.account_id}",
            f"Portfolio value: {self.total_amount_portfolio}",
            f"Expected yield: {self.expected_yield}",
        ]
        if self.money:
            money_text = ", ".join(f"{item['currency']}: {item['amount']}" for item in self.money)
            chunks.append(f"Cash: {money_text}")
        if self.positions:
            lines = []
            for position in self.positions:
                line = (
                    f"{position['ticker']} ({position['instrument_type']}): "
                    f"{position['quantity']} units, current price {position['current_price']}, "
                    f"average price {position['average_position_price']}, "
                    f"share {position['share_of_portfolio']}"
                )
                lines.append(line)
            chunks.append("Positions: " + " | ".join(lines))
        return ". ".join(chunks)


def sandbox_is_configured() -> bool:
    return TINKOFF_SANDBOX_ENABLED and bool(TINKOFF_SANDBOX_TOKEN)


def decimal_to_money_value(amount: Decimal, currency: str = "rub") -> MoneyValue:
    normalized = amount.quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP)
    units = int(normalized)
    nano = int((normalized - Decimal(units)) * Decimal("1000000000"))
    return MoneyValue(currency=currency, units=units, nano=nano)


def money_to_decimal(value: Any) -> Decimal:
    units = getattr(value, "units", 0)
    nano = getattr(value, "nano", 0)
    return Decimal(units) + (Decimal(nano) / Decimal("1000000000"))


def format_money(value: Any) -> str:
    amount = money_to_decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    currency = getattr(value, "currency", "") or ""
    return f"{amount} {currency}".strip()


def _get_account_id(client: SandboxClient, account_id: str | None = None) -> str:
    if account_id:
        return account_id
    if TINKOFF_SANDBOX_ACCOUNT_ID:
        return TINKOFF_SANDBOX_ACCOUNT_ID
    accounts = client.users.get_accounts().accounts
    if not accounts:
        opened = client.sandbox.open_sandbox_account()
        return opened.account_id
    return accounts[0].id


def fetch_sandbox_portfolio(account_id: str | None = None) -> SandboxPortfolioSnapshot:
    if not sandbox_is_configured():
        raise RuntimeError("Tinkoff Sandbox is not configured. Set TINKOFF_SANDBOX_TOKEN.")

    with SandboxClient(TINKOFF_SANDBOX_TOKEN) as client:
        resolved_account_id = _get_account_id(client, account_id)
        portfolio = client.operations.get_portfolio(account_id=resolved_account_id)
        positions_response = client.operations.get_positions(account_id=resolved_account_id)
        positions: list[dict[str, Any]] = []

        for position in portfolio.positions:
            instrument = client.instruments.get_instrument_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_POSITION_UID,
                id=position.position_uid,
            ).instrument
            current_price = format_money(position.current_price)
            average_price = format_money(position.average_position_price)
            expected_yield = format_money(position.expected_yield)
            quantity = str(money_to_decimal(position.quantity))
            quantity_lots = str(money_to_decimal(position.quantity_lots))
            share_ratio = money_to_decimal(position.current_price) * money_to_decimal(position.quantity)
            total_ratio = money_to_decimal(portfolio.total_amount_portfolio)
            share_pct = Decimal("0")
            if total_ratio:
                share_pct = (share_ratio / total_ratio * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            positions.append(
                {
                    "name": instrument.name,
                    "ticker": instrument.ticker or instrument.figi,
                    "figi": instrument.figi,
                    "instrument_type": position.instrument_type,
                    "quantity": quantity,
                    "quantity_lots": quantity_lots,
                    "current_price": current_price,
                    "average_position_price": average_price,
                    "expected_yield": expected_yield,
                    "share_of_portfolio": f"{share_pct}%",
                }
            )

        money_items = [
            {
                "currency": money.currency,
                "amount": format_money(money),
            }
            for money in positions_response.money
        ]

        return SandboxPortfolioSnapshot(
            account_id=resolved_account_id,
            total_amount_portfolio=format_money(portfolio.total_amount_portfolio),
            expected_yield=format_money(portfolio.expected_yield),
            total_amount_shares=format_money(portfolio.total_amount_shares),
            total_amount_bonds=format_money(portfolio.total_amount_bonds),
            total_amount_etf=format_money(portfolio.total_amount_etf),
            total_amount_currencies=format_money(portfolio.total_amount_currencies),
            total_amount_futures=format_money(portfolio.total_amount_futures),
            positions=positions,
            money=money_items,
        )


def open_sandbox_account(client: SandboxClient) -> str:
    return client.sandbox.open_sandbox_account().account_id


def sandbox_pay_in(client: SandboxClient, account_id: str, amount_rub: Decimal) -> None:
    client.sandbox.sandbox_pay_in(
        account_id=account_id,
        amount=decimal_to_money_value(amount_rub, currency="rub"),
    )


def find_instrument_by_ticker(client: Client, ticker: str) -> Any:
    matches = client.instruments.find_instrument(query=ticker).instruments
    for instrument in matches:
        if instrument.ticker == ticker and instrument.api_trade_available_flag:
            return instrument
    for instrument in matches:
        if instrument.ticker == ticker:
            return instrument
    raise RuntimeError(f"Instrument was not found for ticker {ticker}.")


def buy_market_lots(client: SandboxClient, account_id: str, ticker: str, quantity_lots: int) -> None:
    instrument = find_instrument_by_ticker(client, ticker)
    client.orders.post_order(
        instrument_id=instrument.figi,
        figi=instrument.figi,
        quantity=quantity_lots,
        direction=OrderDirection.ORDER_DIRECTION_BUY,
        account_id=account_id,
        order_type=OrderType.ORDER_TYPE_MARKET,
        order_id=str(uuid.uuid4()),
    )
