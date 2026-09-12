"""The count-based ("deep") history path of ``bots/_shared/mt5_loader.py``.

Why this file exists: on 2026-09-07 the same XAUUSD H1 series on the same Exness login answered
2022-10-25 to ``copy_rates_range`` in calendar chunks and 2017-03-06 to ``copy_rates_from`` with a
bar count, minutes apart. ``exness_fx_d1`` had rejected an H1 design on the first number. These
tests pin the two behaviours apart against ``bots/_shared/testing/fake_mt5.FakeMT5``, whose
``deep_rates`` models the deeper history the *server* holds.

No terminal, no network::

    uv run --with pytest==9.0.3 python -m pytest bots/exness_gold_sess/tests -q
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from bots._shared import mt5_loader as M
from bots._shared.sessions import ServerClock
from bots._shared.testing.fake_mt5 import TIMEFRAME_H1, FakeMT5, synthetic_rates

SHALLOW_START = datetime(2024, 6, 3)  # a Monday: what the terminal has synchronised
DEEP_START = datetime(2024, 1, 1)  # a Monday: what the server will serve
NOW = datetime(2024, 12, 2, 12, tzinfo=UTC)
MW = "XAUUSDm"


@pytest.fixture(scope="module")
def clock() -> ServerClock:
    return ServerClock(0, measured_at=NOW, follows_dst_of=None)


@pytest.fixture(scope="module")
def fake() -> FakeMT5:
    """A terminal whose chart cache starts 5 months after the server history does."""
    deep = synthetic_rates(DEEP_START, 3_000, TIMEFRAME_H1, seed=1, price=2000.0)
    shallow_from = int(SHALLOW_START.replace(tzinfo=UTC).timestamp())
    shallow = deep[deep["time"] >= shallow_from]
    return FakeMT5(
        rates={(MW, TIMEFRAME_H1): shallow},
        deep_rates={(MW, TIMEFRAME_H1): deep},
        server_offset_minutes=0,
        now_utc=NOW,
    )


def _end_server(fake: FakeMT5) -> datetime:
    last = int(fake.deep_rates[(MW, TIMEFRAME_H1)]["time"][-1])
    return datetime.fromtimestamp(last, UTC).replace(tzinfo=None)


def test_range_path_sees_only_the_synchronised_cache(fake: FakeMT5) -> None:
    """The regression this whole file guards: a range request under-reports the depth."""
    rates = M._copy_rates_chunked(fake, MW, "1h", datetime(2000, 1, 1), _end_server(fake), 100_000)
    assert rates is not None
    first = datetime.fromtimestamp(int(rates[0]["time"]), UTC).replace(tzinfo=None)
    assert first == SHALLOW_START
    assert len(rates) == len(fake.rates[(MW, TIMEFRAME_H1)])


def test_deep_path_reaches_the_server_history(fake: FakeMT5) -> None:
    rates = M._copy_rates_deep(fake, MW, "1h", datetime(2000, 1, 1), _end_server(fake), 100_000)
    assert rates is not None
    first = datetime.fromtimestamp(int(rates[0]["time"]), UTC).replace(tzinfo=None)
    assert first == DEEP_START
    assert len(rates) == 3_000
    # strictly deeper than the range path, which is the claim the bot's design rests on
    shallow = M._copy_rates_chunked(fake, MW, "1h", datetime(2000, 1, 1), _end_server(fake), 100_000)
    assert len(rates) > len(shallow)


def test_deep_path_walks_backwards_in_chunks_and_dedupes(fake: FakeMT5) -> None:
    """``max_bars`` smaller than the history forces several requests; edges must not duplicate."""
    fake.calls.clear()
    rates = M._copy_rates_deep(fake, MW, "1h", datetime(2000, 1, 1), _end_server(fake), 501)
    assert rates is not None
    n_requests = sum(1 for name, *_ in fake.calls if name == "copy_rates_from")
    assert n_requests >= 6  # 3,000 bars in chunks of 500
    times = rates["time"]
    assert len(set(times.tolist())) == len(times)
    assert (times[1:] > times[:-1]).all()  # sorted ascending, MT5's own order
    assert len(rates) == 3_000


def test_deep_path_is_clipped_to_start_and_end(fake: FakeMT5) -> None:
    start = datetime(2024, 3, 1)
    end = datetime(2024, 9, 1)
    rates = M._copy_rates_deep(fake, MW, "1h", start, end, 100_000)
    assert rates is not None
    lo = datetime.fromtimestamp(int(rates["time"].min()), UTC).replace(tzinfo=None)
    hi = datetime.fromtimestamp(int(rates["time"].max()), UTC).replace(tzinfo=None)
    assert lo >= start and hi <= end


def test_deep_path_passes_epoch_seconds_not_naive_datetimes(fake: FakeMT5) -> None:
    """A naive datetime would be shifted by the machine's local zone (UTC+7 here)."""
    fake.calls.clear()
    M._copy_rates_deep(fake, MW, "1h", datetime(2000, 1, 1), _end_server(fake), 100_000)
    cursors = [args[2] for name, args, _ in fake.calls if name == "copy_rates_from"]
    assert cursors and all(isinstance(c, int) for c in cursors)
    assert cursors[0] == int(_end_server(fake).replace(tzinfo=UTC).timestamp())


def test_deep_path_returns_none_when_the_symbol_is_unknown(fake: FakeMT5) -> None:
    assert M._copy_rates_deep(fake, "NOSUCHm", "1h", datetime(2000, 1, 1), NOW.replace(tzinfo=None), 100) is None


def test_dispatcher_defaults_to_range_and_rejects_a_typo(fake: FakeMT5) -> None:
    end = _end_server(fake)
    default = M._copy_rates(fake, MW, "1h", datetime(2000, 1, 1), end, 100_000)
    ranged = M._copy_rates(fake, MW, "1h", datetime(2000, 1, 1), end, 100_000, "range")
    assert default is not None and (default == ranged).all()
    with pytest.raises(ValueError, match="history_mode"):
        M._copy_rates(fake, MW, "1h", datetime(2000, 1, 1), end, 100_000, "deepest")  # type: ignore[arg-type]


def test_download_mt5_bars_keeps_the_range_path_by_default(tmp_path, fake: FakeMT5, clock: ServerClock) -> None:
    """The FX bot's recorded depths must not move: the default download is unchanged."""
    written = M.download_mt5_bars(
        ["XAUUSD"],
        ("1h",),
        start=datetime(2000, 1, 1),
        end=_end_server(fake) + timedelta(hours=1),
        clock=clock,
        mt5=fake,
        data_dir=tmp_path,
    )
    depth = M.read_history_depth(tmp_path)["timeframes"]["1h"]["XAUUSD"]
    assert depth["history_mode"] == "range"
    assert depth["first_server_day"] == str(SHALLOW_START.date())
    frame = pl.read_parquet(written["1h"])
    assert frame.height == depth["n_bars"] == len(fake.rates[(MW, TIMEFRAME_H1)])


def test_download_mt5_bars_with_history_mode_deep_writes_the_deeper_history(
    tmp_path, fake: FakeMT5, clock: ServerClock
) -> None:
    M.download_mt5_bars(
        ["XAUUSD"],
        ("1h",),
        start=datetime(2000, 1, 1),
        end=_end_server(fake) + timedelta(hours=1),
        clock=clock,
        mt5=fake,
        data_dir=tmp_path,
        history_mode="deep",
    )
    depth = M.read_history_depth(tmp_path)["timeframes"]["1h"]["XAUUSD"]
    assert depth["history_mode"] == "deep"
    assert depth["first_server_day"] == str(DEEP_START.date())
    assert depth["n_bars"] == 3_000


def test_fetch_mt5_bars_deep_returns_frames_and_writes_nothing(tmp_path, fake: FakeMT5, clock: ServerClock) -> None:
    frames, depth = M.fetch_mt5_bars_deep(
        ["XAUUSD"],
        ("1h",),
        end=_end_server(fake) + timedelta(hours=1),
        clock=clock,
        mt5=fake,
    )
    assert list(frames) == ["1h"]
    assert frames["1h"].columns == M.PARQUET_COLUMNS
    assert frames["1h"].height == 3_000
    assert depth["1h"]["XAUUSD"]["history_mode"] == "deep"
    assert depth["1h"]["XAUUSD"]["first_server_day"] == str(DEEP_START.date())
    assert list(tmp_path.iterdir()) == []  # it is an in-memory reader; the caller merges
    assert fake.calls[-1][0] == "shutdown"


def test_fetch_mt5_bars_deep_rejects_a_derived_timeframe(fake: FakeMT5, clock: ServerClock) -> None:
    with pytest.raises(ValueError, match="unknown timeframes"):
        M.fetch_mt5_bars_deep(["XAUUSD"], ("8h",), clock=clock, mt5=fake)
