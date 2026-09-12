---
symbol: XAGUSD
asset_class: metals
bot: exness_gold_sess
template_case_study: "fx_pairs pipeline + mô hình chuỗi thời gian từng tài sản (Ch16 04, 05)"
timezone: UTC
sessions:
  london:
    winter: "08:00-17:00"
    summer: "07:00-16:00"
  overlap:
    winter: "13:00-17:00"
    summer: "12:00-16:00"
  new_york:
    winter: "13:00-22:00"
    summer: "12:00-21:00"
  tokyo:
    winter: "00:00-09:00"
    summer: "00:00-09:00"
decision_times:
  - name: "Quyết định 1"
    when: "Đầu phiên London: 08:00 UTC mùa đông (07:00 mùa hè), sau block mép phiên 30 phút"
  - name: "Quyết định 2"
    when: "Đầu phiên New York: 13:00 UTC mùa đông (12:00 mùa hè), hoặc 14:00 sau cửa sổ tin 13:30"
  - name: "Thoát"
    when: "Ba nhãn: đóng cuối phiên, giữ qua đêm cộng swap, triple-barrier trong phiên"
  - name: "Backtest theo phiên (mức 2)"
    when: "`SESSION_FILTER` trong stage backtest: london, ny, overlap"
avoid_windows:
  - name: "Giờ đầu tuần"
    when: "22:00-23:00 UTC Chủ nhật (mùa hè 21:00-22:00)"
  - name: "Giờ cuối tuần"
    when: "2 giờ cuối thứ Sáu trước giờ đóng"
  - name: "Rollover"
    when: "±15 phút quanh giờ rollover"
  - name: "Nghỉ hằng ngày Globex"
    when: "22:00-23:00 UTC mùa đông (21:00-22:00 mùa hè)"
  - name: "Trước tin Mỹ"
    when: "±30 phút"
  - name: "Phiên Á"
    when: "00:00-07:00 UTC"
news:
  - name: "LBMA silver price"
    when: "12:00 UTC"
  - name: "Số liệu công nghiệp Trung Quốc"
    when: "01:30-02:00 UTC"
  - name: "NFP"
    when: "thứ Sáu đầu tháng 13:30 UTC (mùa hè 12:30)"
  - name: "CPI Mỹ"
    when: "giữa tháng 13:30 UTC (mùa hè 12:30)"
  - name: "FOMC"
    when: "8 lần/năm, quyết định 19:00 UTC, họp báo 19:30 (mùa hè 18:00/18:30)"
  - name: "Đơn xin trợ cấp thất nghiệp"
    when: "thứ Năm 13:30 UTC"
  - name: "ISM PMI"
    when: "ngày làm việc đầu tháng 15:00 UTC (mùa hè 14:00)"
  - name: "Bán lẻ, GDP"
    when: "13:30 UTC"
verify_on_mt5: [symbol_info.trade_contract_size, volume_min, volume_step, swap_long, swap_short, swap_rollover3days, symbol_info_session_trade]
---

# XAGUSD — hồ sơ phiên và giờ giao dịch

**Lớp tài sản**: metals · **Bot**: `exness_gold_sess` · **Khuôn case study**: fx_pairs pipeline + mô hình chuỗi thời gian từng tài sản (Ch16 04, 05)

**Vai trò trong danh mục**: Cặp đôi của vàng với beta cao hơn và spread rộng hơn; nửa kim loại quý, nửa công nghiệp

**Hợp đồng**: 5000 oz mỗi lot (xác minh `trade_contract_size`)

## 1. Giờ giao dịch

Tài sản cơ sở giao dịch theo CME Globex: 23:00 UTC Chủ nhật đến 22:00 UTC thứ Sáu, nghỉ hằng ngày 22:00-23:00 UTC (mùa hè sớm hơn 1 giờ). CFD trên Exness thường theo sát lịch này, có thể lệch vài phút; đọc từ `symbol_info_session_trade` và ghi vào `sessions.py`.

## 2. Các phiên (UTC)

| Phiên | Mùa đông | Mùa hè | Ghi chú |
|---|---|---|---|
| `london` | 08:00-17:00 | 07:00-16:00 | Phiên thanh khoản lớn nhất của FX và kim loại |
| `overlap` | 13:00-17:00 | 12:00-16:00 | London giao New York: spread hẹp nhất, biên độ lớn nhất |
| `new_york` | 13:00-22:00 | 12:00-21:00 | Tin Mỹ 13:30, FOMC 19:00 (mùa đông) |
| `tokyo` | 00:00-09:00 | 00:00-09:00 | Không có DST |

Mùa đông là khoảng đầu tháng 11 đến giữa tháng 3 (giờ chuẩn Mỹ và Anh). Mùa hè sớm hơn một giờ. Tokyo không đổi giờ. `bots/_shared/sessions.py` tính từ múi giờ thật, không hard-code bảng này.

## 3. Lịch tin và sự kiện định kỳ

| Sự kiện | Thời điểm | Ảnh hưởng |
|---|---|---|
| LBMA silver price | 12:00 UTC | Mốc thanh khoản |
| Số liệu công nghiệp Trung Quốc | 01:30-02:00 UTC | Kênh nhu cầu công nghiệp |
| NFP | thứ Sáu đầu tháng 13:30 UTC (mùa hè 12:30) | Biến động lớn nhất tháng cho USD, vàng, chỉ số |
| CPI Mỹ | giữa tháng 13:30 UTC (mùa hè 12:30) | Lãi suất thực; vàng và USTEC phản ứng mạnh |
| FOMC | 8 lần/năm, quyết định 19:00 UTC, họp báo 19:30 (mùa hè 18:00/18:30) | Cửa sổ 2 giờ spread giãn trên mọi tài sản USD |
| Đơn xin trợ cấp thất nghiệp | thứ Năm 13:30 UTC | Nhỏ, nhưng đều đặn |
| ISM PMI | ngày làm việc đầu tháng 15:00 UTC (mùa hè 14:00) | Chỉ số và USD |
| Bán lẻ, GDP | 13:30 UTC | Chỉ số |

## 4. Rollover và swap

Swap tính tại giờ rollover của server (thường quanh 22:00 UTC mùa đông, 21:00 mùa hè; xác minh trên tài khoản). Ngày swap ba lần đọc từ `symbol_info(...).swap_rollover3days` (thường thứ Tư cho FX và kim loại). Spread giãn 15 đến 30 phút quanh rollover; không đặt lệnh thị trường trong cửa sổ này.

## 5. Giờ nên tránh vào lệnh thị trường

| Cửa sổ | Thời gian | Lý do |
|---|---|---|
| Giờ đầu tuần | 22:00-23:00 UTC Chủ nhật (mùa hè 21:00-22:00) | Gap cuối tuần, spread giãn, thanh khoản mỏng |
| Giờ cuối tuần | 2 giờ cuối thứ Sáu trước giờ đóng | Đóng vị thế của các quỹ, spread giãn, rủi ro gap sang thứ Hai |
| Rollover | ±15 phút quanh giờ rollover | Spread giãn, swap tính |
| Nghỉ hằng ngày Globex | 22:00-23:00 UTC mùa đông (21:00-22:00 mùa hè) | Không có giá; lệnh chờ không khớp; gap nhỏ khi mở lại |
| Trước tin Mỹ | ±30 phút | Spread giãn mạnh hơn vàng |
| Phiên Á | 00:00-07:00 UTC | Thanh khoản mỏng, spread rộng; chỉ tính feature |

Tránh ở đây nghĩa là không đặt lệnh thị trường mới, và spread trong cửa sổ đó lấy phân vị 90 khi tính chi phí. Feature vẫn được tính trên mọi nến. Một cửa sổ chỉ được đưa trở lại lịch quyết định khi backtest theo phiên chứng minh nó sống sót qua chi phí.

## 6. Mốc quyết định và khớp lệnh của bot

| Mốc | Thời điểm | Ghi chú |
|---|---|---|
| Quyết định 1 | Đầu phiên London: 08:00 UTC mùa đông (07:00 mùa hè), sau block mép phiên 30 phút | Bot `exness_gold_sess`, nến H1 |
| Quyết định 2 | Đầu phiên New York: 13:00 UTC mùa đông (12:00 mùa hè), hoặc 14:00 sau cửa sổ tin 13:30 | Hai biến thể là hai spec backtest |
| Thoát | Ba nhãn: đóng cuối phiên, giữ qua đêm cộng swap, triple-barrier trong phiên | Ch07 03, Ch19 02 |
| Backtest theo phiên (mức 2) | `SESSION_FILTER` trong stage backtest: london, ny, overlap | Mỗi bộ lọc một hash; N phiên = N trial trong DSR |

Quyết định luôn tại giá đóng nến, khớp ở nến kế tiếp. Mọi cửa sổ feature của bot theo phiên bị chặn tại biên phiên.

## 7. Feature từ repo

- Tỷ số vàng/bạc và độ lệch của nó (mean reversion cấu trúc, Ch08 03)
- Beta với vàng
- Momentum H1 chặn tại biên phiên
- Biến động thực hiện cao, chuẩn hóa theo vol

## 8. Bằng chứng và mẫu trong repo

- SI trong `case_studies/cme_futures`
- SLV trong `case_studies/etfs`

## 9. Tương quan và vị trí trong danh mục

Tương quan với vàng khoảng 0.8; đóng góp độ rộng thấp nhưng cho feature tỷ số.

## 10. Ghi chú riêng

Chi phí là rào cản chính: spread tính theo bps cao hơn vàng đáng kể; cổng chi phí có thể loại bạc khỏi bot dù tín hiệu tốt. Chấp nhận kết quả đó.

## 11. Xác minh trên tài khoản Exness trước khi dùng

```python
import MetaTrader5 as mt5
mt5.initialize()
s = "XAGUSD"  # đổi sang tên đúng trong Market Watch
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
| Tên trong Market Watch | `XAGUSDm` (suffix `m`; `path` = `Standard\Forex\XAGUSDm`) |
| `trade_contract_size` | 5000 |
| `volume_min` / `volume_step` / `volume_max` | 0.01 / 0.01 / 200.0 |
| `digits` / `point` | 3 / 0.001 |
| `swap_long` / `swap_short` (`swap_mode` 1 = points/lot/đêm) | 0.0 / 0.0 |
| `swap_rollover3days` | 3 = Wednesday (swap x3) |
| Tiền tệ base / profit / margin | XAG / USD / XAG |
| Giờ server | UTC+0 quanh năm (đo `ServerClock.measure` trên `BTCUSDm`: offset 0; tuần FX mở Sun 22:00 server tháng 1 và Sun 21:00 server tháng 7, tức server **không** theo DST New York). Nến D1 đóng 00:00 UTC |
| Lịch sử D1 | 3,892 nến, 2014-01-12 → 2026-09-04 |
| Lịch sử H4 | 16,150 nến, 2014-01-12 → 2026-09-04 |
| Lịch sử H1 | 22,831 nến, 2022-10-25 → 2026-09-04 |
| `terminal_info.maxbars` | 100000 (loader tải theo chunk) |

**Giờ giao dịch (giờ server = UTC), suy ra từ nến H1 của 8 tuần gần nhất** (package Python `MetaTrader5` không có `symbol_info_session_trade`, hàm đó chỉ có trong MQL5; đoạn mã ở mục 11 vì vậy không chạy được nguyên văn):

- mon: 00:00–21:00, 22:00–24:00; tue: 00:00–21:00, 22:00–24:00; wed: 00:00–21:00, 22:00–24:00; thu: 00:00–21:00, 22:00–24:00; fri: 00:00–21:00; sat: đóng; sun: 22:00–24:00

## 12b. Spread theo phiên — đo 2026-09-07

Đo bằng `bots._shared.costs_mt5.measure_spreads(["XAGUSD"], days=30, mt5)` (script:
`bots/exness_gold_sess/tools/measure_metals_costs.py`), **30 ngày** tick `COPY_TICKS_INFO`,
cửa sổ **2026-08-08 15:50 → 2026-09-07 15:50 UTC**, `measured_at_utc = 2026-09-07 15:50:46`,
server UTC+0. Tổng **1,559,623 tick**. Spread = `(ask − bid) / mid × 1e4` (bps), điểm =
`(ask − bid) / point` với `point = 0.001`. Nguồn ghi lại:
`ML4T_DATA_PATH/mt5/spreads_by_session.{json,parquet}` (5 cặp FX của `exness_fx_d1` giữ nguyên;
bản sao lưu trước khi ghi: `*.bak_2026-09-07`) và `mt5/symbol_info_metals_2026-09-07.json`.

| Bucket | n_ticks | p50 (bps) | p90 (bps) | mean (bps) | p50 (points) | p90 (points) |
|---|---|---|---|---|---|---|
| all | 1,559,623 | 4.53 | 4.68 | 4.514 | 30 | 30 |
| rollover (±15′ quanh 00:00 server) | 30,394 | 4.53 | 4.70 | 4.515 | 30 | 30 |
| overlap (London × New York) | 424,425 | 4.53 | 4.64 | 4.502 | 30 | 30 |
| london | 286,223 | 4.54 | 4.68 | 4.517 | 30 | 30 |
| new_york | 262,556 | 4.52 | 4.64 | 4.510 | 30 | 30 |
| asia | 556,025 | 4.53 | 4.70 | 4.523 | 30 | 30 |
| other | **0 tick** | — | — | — | — | — |

Đọc bảng này:

- Spread **tính theo điểm là hằng số 30 điểm (0.030 USD) ở mọi bucket và mọi phân vị** trong suốt
  30 ngày; dao động ở cột bps (4.52 → 4.70) chỉ do giá bạc thay đổi (mid ≈ 64–67). Bạc **không**
  có phiên nào rẻ hơn trên tài khoản Pro này.
- **Bạc đắt hơn vàng 7.7 lần theo bps** (p90 4.70 so với 0.60): đúng như ghi chú mục 10 —
  chi phí là rào cản chính của bạc, và cổng chi phí có thể loại XAGUSD khỏi bot dù tín hiệu tốt.
  Chi phí vòng: **2 × 4.70 = 9.4 bps** (commission 0; swap đo được 0.0/0.0).
- Bucket `other` **rỗng**, cùng lý do như vàng: XAGUSDm dừng báo giá 21:00–22:00 UTC (kiểm chứng
  trên `1h.parquet`) và Sydney mở 22:00 UTC trong mùa AEST.
- Thanh khoản mỏng hơn vàng rõ rệt: 1.56 M tick so với 6.41 M tick của vàng trong cùng cửa sổ
  (tick volume H1 trung vị 1,169 so với 6,449 — xem census mục 4).

Bổ sung các trường `symbol_info` đọc lại ngày 2026-09-07 (những trường mục 12 chưa có):

| Trường | Giá trị |
|---|---|
| `description` | Silver vs US Dollar |
| `trade_tick_size` / `trade_tick_value` | 0.001 / 5.0 USD |
| `spread` (tại lúc đọc) / `spread_float` | 30 điểm / True |
| `trade_stops_level` / `trade_freeze_level` | 0 / 0 |
| `trade_mode` / `trade_calc_mode` / `filling_mode` | 4 (full) / 0 (forex) / 3 (FOK\|IOC) |
| `swap_mode` | 1 (points per lot per night) |

Mọi con số ở mục 12 (contract 5000, volume 0.01/0.01/200, digits 3, point 0.001, swap 0.0/0.0,
`swap_rollover3days` 3, base/profit/margin XAG/USD/XAG) được đọc lại ngày 2026-09-07 và **không đổi**.
Vẫn là tài khoản **demo** `206539306 @ Exness-MT5Trial7`; phải đo lại trên tài khoản thật trước `16_costs`.

## 12c. Độ sâu H1 đo lại bằng đường COUNT-BASED — 2026-09-07 (task B0)

**Lần đầu tiên bạc được đo bằng đường này** (spec phase 1 §0 gọi đây là "ẩn số duy nhất còn lại
của phase 1"). Cùng terminal, cùng login `206539306 @ Exness-MT5Trial7`, cùng ngày:

| Cách hỏi | Hàm | Kết quả XAGUSD H1 |
|---|---|---|
| Theo **khoảng lịch**, chia chunk | `mt5_loader._copy_rates_chunked` → `copy_rates_range` | **22,831 nến, 2022-10-25 → 2026-09-04** (mục 12) |
| Theo **số nến**, epoch seconds | `mt5_loader._copy_rates_deep` → `copy_rates_from` | **57,079 nến, 2014-01-12 → 2026-09-04** |

Nguyên nhân và script giống hệt vàng: xem `bots/assets/XAUUSD.md` §12c và
`bots/exness_gold_sess/tools/deepen_metals_h1.py`.

**Phần trước 2017 không phải lưới H1** (giống vàng). Đếm nến theo tuần:

| Tuần | 2017-01-30 | 2017-02-06 | 2017-02-13 | 2017-02-20 | **2017-02-27** | 2017-03-06 | 2017-03-13 |
|---|---|---|---|---|---|---|---|
| Số nến H1 | 6 | 6 | 6 | 11 | **116** | 118 | 115 |

| Trường | Giá trị đo được 2026-09-07 |
|---|---|
| H1 tổng | 57,079 nến, 2014-01-12 → 2026-09-04 |
| H1 tiền tố thưa (6 nến/tuần) | 976 nến, 2014-01-12 → 2017-02-26 — **không dùng được** |
| **H1 dày** | **56,103 nến, 2017-02-27 → 2026-09-04**, 2,921 ngày lịch |
| `dense_history_start` | **2017-02-27** — **cùng tuần với vàng** |
| Đối chiếu phần chồng lấn với file cũ | 22,831 nến chung, **0 sai lệch OHLC**, 0 nến chỉ có ở file cũ |

**Kết luận cho phase 1**: bạc **không** là mã thiếu lịch sử. Nhánh B của spec (§2.c', đưa XAGUSD ra
khỏi `universe` và giữ nó làm feature tỷ số vàng/bạc trên lưới D1) **không cần dùng**. Cả hai kim
loại vào `universe` với `history_start: 2017-02-27` chung. Câu hỏi còn lại về bạc không phải độ sâu
mà là **chi phí**: round trip 9.4 bps so với 1.2 bps của vàng (mục 12b), tức 7.7×; cổng chi phí
`16_costs` (§2.e mục 4 của spec) vẫn có thể loại XAGUSD khỏi `universe`.
