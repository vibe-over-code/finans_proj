import argparse
from decimal import Decimal

from tinkoff.invest.sandbox.client import SandboxClient

from tinkoff_sandbox import (
    TINKOFF_SANDBOX_TOKEN,
    buy_market_lots,
    fetch_sandbox_portfolio,
    open_sandbox_account,
    sandbox_pay_in,
)


DEFAULT_PORTFOLIOS = {
    "conservative": {
        "cash_rub": Decimal("250000"),
        "positions": [
            {"ticker": "LQDT", "lots": 40},
            {"ticker": "SBER", "lots": 5},
            {"ticker": "GAZP", "lots": 5},
        ],
    },
    "balanced": {
        "cash_rub": Decimal("350000"),
        "positions": [
            {"ticker": "SBER", "lots": 15},
            {"ticker": "GAZP", "lots": 12},
            {"ticker": "LKOH", "lots": 3},
            {"ticker": "TMOS", "lots": 25},
        ],
    },
    "aggressive": {
        "cash_rub": Decimal("500000"),
        "positions": [
            {"ticker": "SBER", "lots": 20},
            {"ticker": "GAZP", "lots": 20},
            {"ticker": "LKOH", "lots": 5},
            {"ticker": "YDEX", "lots": 8},
            {"ticker": "TMOS", "lots": 40},
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Tinkoff Invest sandbox accounts and test portfolios.")
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=list(DEFAULT_PORTFOLIOS.keys()),
        help="Profiles to create. Default: conservative balanced aggressive",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not TINKOFF_SANDBOX_TOKEN:
        raise RuntimeError("TINKOFF_SANDBOX_TOKEN is not set.")

    with SandboxClient(TINKOFF_SANDBOX_TOKEN) as client:
        for profile_name in args.profiles:
            if profile_name not in DEFAULT_PORTFOLIOS:
                raise RuntimeError(f"Unknown profile: {profile_name}")

            profile = DEFAULT_PORTFOLIOS[profile_name]
            account_id = open_sandbox_account(client)
            sandbox_pay_in(client, account_id, profile["cash_rub"])
            print(f"[{profile_name}] account_id={account_id}")

            for item in profile["positions"]:
                ticker = item["ticker"]
                lots = int(item["lots"])
                try:
                    buy_market_lots(client, account_id, ticker, lots)
                    print(f"  BUY {ticker} lots={lots}")
                except Exception as error:
                    print(f"  SKIP {ticker} lots={lots}: {error}")

            snapshot = fetch_sandbox_portfolio(account_id=account_id)
            print(f"  portfolio={snapshot.to_dict()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
