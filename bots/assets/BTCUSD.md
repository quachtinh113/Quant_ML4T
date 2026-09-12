---
symbol: BTCUSD
asset_class: crypto
bot: exness_btc_8h
template_case_study: "exness_gold_sess (Route B fork, quyết định 2026-09-08; KHÔNG phải crypto_perps_funding — xem bots/exness_btc_8h/BOT.md Decisions log 2026-09-08 'Template')"
timezone: UTC
sessions:
  new_york:
    winter: "13:00-22:00"
    summer: "12:00-21:00"
  us_cash:
    winter: "14:30-21:00"
    summer: "13:30-20:00"
  london:
    winter: "08:00-17:00"
    summer: "07:00-16:00"
decision_times:
  - name: "Quyết định"
    when: "00:00, 08:00, 16:00 UTC (cadence `8_hour` của case study — KHÔNG phải `8_hour_funding_aligned`: CFD không có mốc funding, xem case_studies/exness_btc_8h/config/setup.yaml:62-65)"
  - name: "Khớp lệnh"
    when: "Nến kế tiếp"
  - name: "Backtest theo phiên (mức 2)"
    when: "`SESSION_FILTER`: chỉ mốc 16:00 (giờ Mỹ), chỉ ngày trong tuần, tất cả"
avoid_windows:
  - name: "Cuối tuần"
    when: "thứ Bảy và Chủ nhật"
  - name: "Đêm Mỹ"
    when: "04:00-07:00 UTC"
  - name: "Rollover swap CFD"
    when: "Giờ rollover server, hằng ngày kể cả cuối tuần (xác minh)"
news:
  - name: "Mốc funding của sàn perp"
    when: "00:00, 08:00, 16:00 UTC"
  - name: "Đáo hạn quyền chọn Deribit"
    when: "thứ Sáu 08:00 UTC, lớn nhất cuối tháng và cuối quý"
  - name: "Đáo hạn hợp đồng tương lai CME BTC"
    when: "thứ Sáu cuối tháng 15:00 UTC"
  - name: "Dòng tiền ETF BTC giao ngay Mỹ"
    when: "Công bố sau phiên Mỹ"
  - name: "Sự cố sàn, thanh lý dây chuyền"
    when: "Không báo trước, thường ngoài giờ Mỹ"
  - name: "NFP"
    when: "thứ Sáu đầu tháng 13:30 UTC (mùa hè 12:30)"
  - name: "CPI Mỹ"
    when: "giữa tháng 13:30 UTC (mùa hè 12:30)"
  - name: "FOMC"
    when: "8 lần/năm, quyết định 19:00 UTC, họp báo 19:30 (mùa hè 18:00/18:30)"
verify_on_mt5: [symbol_info.trade_contract_size, volume_min, volume_step, swap_long, swap_short, swap_rollover3days, symbol_info_session_trade]
---

# BTCUSD — hồ sơ phiên và giờ giao dịch

**Lớp tài sản**: crypto · **Bot**: `exness_btc_8h` · **Khuôn case study**: `exness_gold_sess` (Route B fork, quyết định 2026-09-08 — KHÔNG phải `crypto_perps_funding` như ghi trước đây; xem `bots/exness_btc_8h/BOT.md` Decisions log 2026-09-08 "Template")

**Vai trò trong danh mục**: Động lực gần như độc lập với FX và chỉ số; chạy 24/7

**Hợp đồng**: 1 BTC mỗi lot (xác minh `trade_contract_size`, `volume_min` có thể là 0.01)

## 1. Giờ giao dịch

Giao dịch 24/7. Exness có thể có khoảng bảo trì ngắn hằng tuần; đọc từ `symbol_info_session_trade` cho cả 7 ngày.

## 2. Các phiên (UTC)

| Phiên | Mùa đông | Mùa hè | Ghi chú |
|---|---|---|---|
| `new_york` | 13:00-22:00 | 12:00-21:00 | Tin Mỹ 13:30, FOMC 19:00 (mùa đông) |
| `us_cash` | 14:30-21:00 | 13:30-20:00 | Phiên tiền mặt NYSE/Nasdaq; đấu giá mở 14:30 và đóng 21:00 (mùa đông) |
| `london` | 08:00-17:00 | 07:00-16:00 | Phiên thanh khoản lớn nhất của FX và kim loại |

Mùa đông là khoảng đầu tháng 11 đến giữa tháng 3 (giờ chuẩn Mỹ và Anh). Mùa hè sớm hơn một giờ. Tokyo không đổi giờ. `bots/_shared/sessions.py` tính từ múi giờ thật, không hard-code bảng này.

## 3. Lịch tin và sự kiện định kỳ

| Sự kiện | Thời điểm | Ảnh hưởng |
|---|---|---|
| Mốc funding của sàn perp | 00:00, 08:00, 16:00 UTC | Dòng tiền xoay quanh mốc; đây là mốc quyết định của bot |
| Đáo hạn quyền chọn Deribit | thứ Sáu 08:00 UTC, lớn nhất cuối tháng và cuối quý | Biến động và pin quanh strike lớn |
| Đáo hạn hợp đồng tương lai CME BTC | thứ Sáu cuối tháng 15:00 UTC | Biến động |
| Dòng tiền ETF BTC giao ngay Mỹ | Công bố sau phiên Mỹ | Xu hướng nhiều ngày |
| Sự cố sàn, thanh lý dây chuyền | Không báo trước, thường ngoài giờ Mỹ | Breaker phải bắt được |
| NFP | thứ Sáu đầu tháng 13:30 UTC (mùa hè 12:30) | Biến động lớn nhất tháng cho USD, vàng, chỉ số |
| CPI Mỹ | giữa tháng 13:30 UTC (mùa hè 12:30) | Lãi suất thực; vàng và USTEC phản ứng mạnh |
| FOMC | 8 lần/năm, quyết định 19:00 UTC, họp báo 19:30 (mùa hè 18:00/18:30) | Cửa sổ 2 giờ spread giãn trên mọi tài sản USD |

## 4. Rollover và swap

Swap tính tại giờ rollover của server (thường quanh 22:00 UTC mùa đông, 21:00 mùa hè; xác minh trên tài khoản). Ngày swap ba lần đọc từ `symbol_info(...).swap_rollover3days` (thường thứ Tư cho FX và kim loại). Spread giãn 15 đến 30 phút quanh rollover; không đặt lệnh thị trường trong cửa sổ này.

## 5. Giờ nên tránh vào lệnh thị trường

| Cửa sổ | Thời gian | Lý do |
|---|---|---|
| Cuối tuần | thứ Bảy và Chủ nhật | Biến động đột ngột (rủi ro gap, chưa đo được); đánh cờ `weekend`, backtest riêng có và không có cuối tuần. **SPREAD RỘNG ĐÃ BỊ BÁC BỎ trên tài khoản này** — đo 2026-09-08 (mục 13a): p90 weekday 1.572 / Sat 1.587 / Sun 1.587 bps, chênh 0.9 % tại p90 và 0 % tại median; xem `case_studies/exness_btc_8h/config/setup.yaml:277-295` |
| Đêm Mỹ | 04:00-07:00 UTC | Khối lượng thấp nhất trong ngày |
| Rollover swap CFD | Giờ rollover server (00:00 UTC, hằng ngày kể cả cuối tuần) | Swap crypto lớn (long −1,638.6 điểm/lot/đêm, ×3 thứ Sáu) — nhưng KHÔNG đưa vào nhãn: là hai cờ trạng thái (`pays_swap_night`, `pays_triple_swap`) + phí ở `16_costs`, quyết định 2026-09-08 (`bots/exness_btc_8h/BOT.md` Decisions log). **GIÃN SPREAD QUANH ROLLOVER ĐÃ BỊ BÁC BỎ** trên tài khoản này — bucket `rollover` (±15 phút quanh nửa đêm server) đo 1.29/1.58 bps, giống hệt mọi bucket khác |

Tránh ở đây nghĩa là không đặt lệnh thị trường mới, và spread trong cửa sổ đó lấy phân vị 90 khi tính chi phí. Feature vẫn được tính trên mọi nến. Một cửa sổ chỉ được đưa trở lại lịch quyết định khi backtest theo phiên chứng minh nó sống sót qua chi phí.

## 6. Mốc quyết định và khớp lệnh của bot

| Mốc | Thời điểm | Ghi chú |
|---|---|---|
| Quyết định | 00:00, 08:00, 16:00 UTC (cadence `8_hour` của case study — không `8_hour_funding_aligned`) | Bot `exness_btc_8h` |
| Khớp lệnh | Nến kế tiếp |  |
| Backtest theo phiên (mức 2) | `SESSION_FILTER`: chỉ mốc 16:00 (giờ Mỹ), chỉ ngày trong tuần, tất cả | Ba spec, ba hash |

Quyết định luôn tại giá đóng nến, khớp ở nến kế tiếp. Mọi cửa sổ feature của bot theo phiên bị chặn tại biên phiên.

## 7. Feature từ repo

- Momentum và mean reversion nến 8 giờ và D1 (feature của `crypto_perps_funding` 03, trừ funding)
- Biến động thực hiện và khối lượng tương đối
- Cờ cuối tuần, cờ giờ Mỹ
- Beta với USTEC trong giờ Mỹ (Ch08 03)

## 8. Bằng chứng và mẫu trong repo

- BTCUSDT trong `case_studies/crypto_perps_funding/config/setup.yaml`, cadence 8 giờ
- `25_live_trading/09_crypto_funding_deployment_loop.py` — Binance nghiên cứu, Alpaca thực thi; mẫu venue split
- `21_rl_execution_hedging/04_crypto_execution_rl.py` — thực thi crypto

## 9. Tương quan và vị trí trong danh mục

Tương quan với USTEC dương trong giờ Mỹ, gần 0 ngoài giờ; cuối tuần độc lập hoàn toàn.

## 10. Ghi chú riêng

Điểm khác lớn nhất so với case study gốc: không có funding rate, không có basis; chiến lược funding arbitrage của chương 12 không áp dụng được. Chỉ phần momentum, vol, khối lượng còn dùng được. Swap CFD hằng ngày là chi phí giữ lệnh chính.

## 11. Xác minh trên tài khoản Exness trước khi dùng

```python
import MetaTrader5 as mt5
mt5.initialize()
s = "BTCUSD"  # đổi sang tên đúng trong Market Watch
info = mt5.symbol_info(s)
print(info.trade_contract_size, info.volume_min, info.volume_step, info.volume_max,
      info.swap_long, info.swap_short, info.swap_rollover3days, info.spread, info.digits)
for day in range(7):
    i = 0
    while (sess := mt5.symbol_info_session_trade(s, day, i)) is not None:
        print(day, sess); i += 1
rates = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_D1, 0, 100000)
print('D1 bars:', len(rates), 'from', rates[0]['time'] if len(rates) else None)
```

Kết quả đo được ghi đè lên mọi con số ước lượng trong file này. Cập nhật file khi Exness đổi giờ hoặc điều kiện hợp đồng.

## 12. Measured on this account (2026-09-05)

Đo trực tiếp từ terminal MT5 đang đăng nhập (server `Exness-MT5Trial7`, tài khoản **demo**, loại tài khoản **Pro**, tiền tài khoản USD), ngày 2026-09-05, bằng `bots/_shared/mt5_loader.py` và `bots/_shared/costs_mt5.py`. Nguồn: `ML4T_DATA_PATH/mt5/history_depth.json`, `sessions_mt5.json`, `spreads_by_session.json`. Các con số dưới đây ghi đè mọi ước lượng ở các mục trên; đọc lại trên tài khoản thật trước khi giao dịch thật.

| Trường | Giá trị đo được |
|---|---|
| Tên trong Market Watch | `BTCUSDm` (suffix `m`; `path` = `Standard\Crypto\BTCUSDm`) |
| `trade_contract_size` | 1 |
| `volume_min` / `volume_step` / `volume_max` | 0.01 / 0.01 / 200.0 |
| `digits` / `point` | 2 / 0.01 |
| `swap_long` / `swap_short` (`swap_mode` 1 = points/lot/đêm) | -1638.6 / 0.0 |
| `swap_rollover3days` | 5 = Friday (swap x3) |
| Tiền tệ base / profit / margin | BTC / USD / BTC |
| Giờ server | UTC+0 quanh năm (đo `ServerClock.measure` trên `BTCUSDm`: offset 0; tuần FX mở Sun 22:00 server tháng 1 và Sun 21:00 server tháng 7, tức server **không** theo DST New York). Nến D1 đóng 00:00 UTC |
| Lịch sử D1 | 3,130 nến, 2018-02-09 → 2026-09-05 |
| Lịch sử H4 | 18,756 nến, 2018-02-09 → 2026-09-05 |
| Lịch sử H1 | 33,875 nến, 2022-10-25 → 2026-09-05 |
| `terminal_info.maxbars` | 100000 (loader tải theo chunk) |

**Giờ giao dịch (giờ server = UTC), suy ra từ nến H1 của 8 tuần gần nhất** (package Python `MetaTrader5` không có `symbol_info_session_trade`, hàm đó chỉ có trong MQL5; đoạn mã ở mục 11 vì vậy không chạy được nguyên văn):

- mon: 00:00–24:00; tue: 00:00–24:00; wed: 00:00–24:00; thu: 00:00–24:00; fri: 00:00–24:00; sat: 00:00–24:00; sun: 00:00–24:00

## 13. Chi phí đo được (2026-09-08) — spread theo tick, spread theo nến, và swap

Đo trên cùng tài khoản demo `206539306 @ Exness-MT5Trial7` bằng
`bots/exness_btc_8h/tools/measure_btc_costs.py` và `measure_btc_weekend_spread.py`. Bản ghi:
`ML4T_DATA_PATH/mt5/spreads_by_session.{json,parquet}`, `spread_weekend_BTCUSD_2026-09-08.json`,
`symbol_info_btc_2026-09-08.json`. Các con số này ghi đè mọi ước lượng ở mục 5.

### 13a. Spread theo tick (chi phí **hiện tại**, dùng cho lệnh thật)

3.710.378 tick `COPY_TICKS_INFO` trong 30 ngày (2026-08-08 22:52 → 2026-09-07 22:52 UTC). Spread
báo giá là **hằng số 1.000 điểm = 10,00 USD** ở mọi bucket và mọi phân vị:

| Bucket | p50 (bps) | p90 (bps) |
|---|---|---|
| all / rollover / overlap / london / new_york / asia / other | 1,28-1,29 | 1,57-1,59 |

Vòng khứ hồi tại p90 = **3,18 bps**.

**Cuối tuần: không rộng hơn.** `measure_spreads` chia bucket theo phiên và mọi cửa sổ phiên trong
`bots/_shared/sessions.py` chỉ tính ngày trong tuần, nên với mã 24/7 cả cuối tuần rơi vào bucket
`other`. Đo riêng theo ngày lịch UTC: weekday p50 1,291 / p90 1,572; thứ Bảy 1,291 / 1,587; Chủ
nhật 1,294 / 1,587. Nghĩa là **mục 5 nói cuối tuần "spread CFD rộng" là SAI trên tài khoản này,
hôm nay** — với spread. Với thanh khoản và rủi ro gap thì phép đo phân vị spread không trả lời
được, nên cờ `is_weekend` vẫn giữ làm feature trạng thái chứ không loại cuối tuần.

**Rollover: không giãn.** Bucket `rollover` (±15 phút quanh nửa đêm server, 75.463 tick) cho p50
1,29 / p90 1,58 — giống hệt mọi bucket khác. Cảnh báo "không đặt lệnh thị trường quanh rollover"
ở mục 4 và mục 5 **không đúng cho mã này trên tài khoản này**.

### 13b. Spread theo nến (chi phí **trong mẫu**, dùng cho backtest)

Trường `spread` của từng nến H4 trong `ML4T_DATA_PATH/mt5/4h.parquet` (bản ghi point-in-time của
broker). Trên lưới quyết định 8 giờ, cửa sổ phát triển 2018-02-16 → 2025-08-31, 8.261 slot:
**p50 6,32 bps / p90 17,40 bps** mỗi lượt — gấp 11 lần con số tick hôm nay.

Lý do là số học chứ không phải lỗi dữ liệu: spread gần như hằng số theo **điểm** trong khi giá
BTC tăng 15 lần. Trung vị theo năm (bps): 2018 14,57 · 2019 12,55 · 2020 7,48 · 2021 6,36 ·
2022 5,27 · 2023 4,03 · 2024 4,41 · 2025 2,79. Chỉ **0** nến trên lưới quyết định (và đúng 1 trên
18.756 nến H4 thô) báo spread bằng 0, nên không cần quy tắc điền như vàng (1.081 nến).

**Hệ quả**: backtest 2018-2025 định giá bằng con số tick hôm nay sẽ hạ thấp chi phí 11 lần, đúng ở
những năm chứa dữ liệu huấn luyện. `case_studies/exness_btc_8h/config/setup.yaml::costs.spread_bps`
vì vậy mang dải **trong mẫu**; bảng tick nằm ở `costs.spread_bps_by_session` và chỉ dùng cho lệnh
thật và cho cổng go-live.

### 13c. Swap — chi phí lớn nhất của mã này

`symbol_info` đọc 2026-09-08: `swap_long = -1638,6` / `swap_short = 0,0` điểm mỗi lot mỗi đêm
(`swap_mode 1`), `swap_rollover3days = 5` = **thứ Sáu** (FX và kim loại trên cùng tài khoản báo 3 =
thứ Tư), `path = Standard\Crypto\BTCUSDm` nên `costs_mt5.read_swaps` suy ra
`charges_weekends = True`: **tính mọi đêm, kể cả cuối tuần**.

Quy đổi: -1638,6 × 0,01 × 1 = **-16,386 USD mỗi BTC mỗi đêm** = **-2,08 bps** giá trị danh nghĩa
tại giá 78.942. Một tuần là 9 lần tính (7 đêm + 2 lần cộng thêm của thứ Sáu), một năm là 469 lần =
**-7.685 USD mỗi BTC**, tức **-9,7 %/năm** tại giá đo được. Chiều bán trả 0,0 — bất thường, và là
số **demo**, phải đọc lại trên tài khoản thật.

Server là UTC+0 nên swap tính lúc 00:00 UTC, trùng đúng một trong ba mốc quyết định. Đo bằng
`costs_mt5.holding_cost_points`: giữ từ mốc 16:00 đến mốc 00:00 vượt qua đúng một nửa đêm và trả
đủ một đêm; hai mốc 00:00 và 08:00 không trả gì. Nghĩa là **một mốc trong ba** gánh chi phí này,
biết trước được, nên nó là feature trạng thái (`pays_swap_night`, `pays_triple_swap`) chứ không
phải chú thích.

### 13d. Lịch sử nến, đo lại 2026-09-08

Lưới 8 giờ gấp từ nến H4. Tuần đầu tài khoản phục vụ (2018-02-09 → 02-15) thiếu 10 slot và có 6
slot chỉ gấp từ **một** nến H4 — nửa nến mang tên nến đủ. Từ **2018-02-16** trở đi: 9.371 slot
liên tiếp đến 2026-09-05 16:00, **không thiếu slot nào**, mọi slot gấp từ đúng hai nến H4. Vì vậy
`universe.history_start = 2018-02-16`, không phải 2018-02-09.

