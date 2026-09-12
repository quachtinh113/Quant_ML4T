---
symbol: USDCAD
asset_class: forex
bot: exness_fx_d1
template_case_study: "fx_pairs"
timezone: UTC
sessions:
  new_york:
    winter: "13:00-22:00"
    summer: "12:00-21:00"
  overlap:
    winter: "13:00-17:00"
    summer: "12:00-16:00"
decision_times:
  - name: "Quyết định"
    when: "Giờ đóng nến D1 của server (một mốc/ngày)"
  - name: "Khớp lệnh"
    when: "Nến kế tiếp sau quyết định (`execution_delay: next_bar_open`)"
  - name: "Backtest theo phiên (mức 1)"
    when: "Ứng viên `decision.snapshot`: đóng nến server, 08:00 UTC London open, 13:00 UTC NY open"
avoid_windows:
  - name: "Giờ đầu tuần"
    when: "22:00-23:00 UTC Chủ nhật (mùa hè 21:00-22:00)"
  - name: "Giờ cuối tuần"
    when: "2 giờ cuối thứ Sáu trước giờ đóng"
  - name: "Rollover"
    when: "±15 phút quanh giờ rollover"
  - name: "Phiên Á và đầu London"
    when: "00:00-12:00 UTC"
  - name: "Cùng lúc NFP và việc làm Canada"
    when: "13:30 UTC"
news:
  - name: "BoC"
    when: "8 lần/năm, 14:45 UTC (mùa hè 13:45)"
  - name: "Việc làm Canada"
    when: "13:30 UTC, thường cùng ngày NFP"
  - name: "Tồn kho dầu EIA"
    when: "thứ Tư 15:30 UTC (mùa hè 14:30)"
  - name: "OPEC+"
    when: "Theo lịch họp"
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

# USDCAD — hồ sơ phiên và giờ giao dịch

**Lớp tài sản**: forex · **Bot**: `exness_fx_d1` · **Khuôn case study**: fx_pairs

**Vai trò trong danh mục**: Gắn với dầu và số liệu Mỹ; động lực khác bốn cặp còn lại

**Hợp đồng**: 100000 đơn vị tiền cơ sở mỗi lot (xác minh `trade_contract_size`)

## 1. Giờ giao dịch

Thị trường FX mở từ 22:00 UTC Chủ nhật (Sydney) đến 22:00 UTC thứ Sáu (mùa đông; mùa hè sớm hơn 1 giờ). Trên Exness, giờ giao dịch thực tế của từng mã và giờ nghỉ hằng ngày đọc từ `symbol_info_session_trade`; không suy từ lịch thị trường.

## 2. Các phiên (UTC)

| Phiên | Mùa đông | Mùa hè | Ghi chú |
|---|---|---|---|
| `new_york` | 13:00-22:00 | 12:00-21:00 | Tin Mỹ 13:30, FOMC 19:00 (mùa đông) |
| `overlap` | 13:00-17:00 | 12:00-16:00 | London giao New York: spread hẹp nhất, biên độ lớn nhất |

Mùa đông là khoảng đầu tháng 11 đến giữa tháng 3 (giờ chuẩn Mỹ và Anh). Mùa hè sớm hơn một giờ. Tokyo không đổi giờ. `bots/_shared/sessions.py` tính từ múi giờ thật, không hard-code bảng này.

## 3. Lịch tin và sự kiện định kỳ

| Sự kiện | Thời điểm | Ảnh hưởng |
|---|---|---|
| BoC | 8 lần/năm, 14:45 UTC (mùa hè 13:45) | Cửa sổ 1 giờ |
| Việc làm Canada | 13:30 UTC, thường cùng ngày NFP | Biến động kép |
| Tồn kho dầu EIA | thứ Tư 15:30 UTC (mùa hè 14:30) | Qua kênh dầu |
| OPEC+ | Theo lịch họp | Qua kênh dầu |
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
| Phiên Á và đầu London | 00:00-12:00 UTC | Biên độ thấp, spread rộng hơn EURUSD |
| Cùng lúc NFP và việc làm Canada | 13:30 UTC | Biến động cực đoan |

Tránh ở đây nghĩa là không đặt lệnh thị trường mới, và spread trong cửa sổ đó lấy phân vị 90 khi tính chi phí. Feature vẫn được tính trên mọi nến. Một cửa sổ chỉ được đưa trở lại lịch quyết định khi backtest theo phiên chứng minh nó sống sót qua chi phí.

## 6. Mốc quyết định và khớp lệnh của bot

| Mốc | Thời điểm | Ghi chú |
|---|---|---|
| Quyết định | Giờ đóng nến D1 của server (một mốc/ngày) | Bot `exness_fx_d1`, cadence daily như `fx_pairs` |
| Khớp lệnh | Nến kế tiếp sau quyết định (`execution_delay: next_bar_open`) | Không khớp tại giá vừa nhìn thấy |
| Backtest theo phiên (mức 1) | Ứng viên `decision.snapshot`: đóng nến server, 08:00 UTC London open, 13:00 UTC NY open | Mỗi mốc một experiment, nhãn tính lại theo mốc, hash riêng |

Quyết định luôn tại giá đóng nến, khớp ở nến kế tiếp. Mọi cửa sổ feature của bot theo phiên bị chặn tại biên phiên.

## 7. Feature từ repo

- Feature liên thị trường với giá dầu WTI (CL trong `cme_futures`)
- Momentum D1
- Carry BoC và Fed
- Cờ phiên NY

## 8. Bằng chứng và mẫu trong repo

- `case_studies/fx_pairs/config/setup.yaml` có USD_CAD trong universe 20 cặp G10
- `02_financial_data_universe/12_fx_pairs_eda.py` — cấu trúc dữ liệu FX daily
- `25_live_trading/11_fx_deployment_loop.py` — vòng lặp triển khai FX (IB), mẫu cho adapter MT5

## 9. Tương quan và vị trí trong danh mục

Tương quan thấp với EUR/GBP/JPY.

## 10. Ghi chú riêng

Nhiều nhất trong 10 cặp phụ thuộc vào một tài sản khác (dầu); feature cấu trúc Ch08 03 là lý do tồn tại của cặp này trong danh mục.

## 11. Xác minh trên tài khoản Exness trước khi dùng

```python
import MetaTrader5 as mt5
mt5.initialize()
s = "USDCAD"  # đổi sang tên đúng trong Market Watch
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
| Tên trong Market Watch | `USDCADm` (suffix `m`; `path` = `Standard\Forex\USDCADm`) |
| `trade_contract_size` | 100000 |
| `volume_min` / `volume_step` / `volume_max` | 0.01 / 0.01 / 200.0 |
| `digits` / `point` | 5 / 1e-05 |
| `swap_long` / `swap_short` (`swap_mode` 1 = points/lot/đêm) | 0.0 / 0.0 |
| `swap_rollover3days` | 3 = Wednesday (swap x3) |
| Tiền tệ base / profit / margin | USD / CAD / USD |
| Giờ server | UTC+0 quanh năm (đo `ServerClock.measure` trên `BTCUSDm`: offset 0; tuần FX mở Sun 22:00 server tháng 1 và Sun 21:00 server tháng 7, tức server **không** theo DST New York). Nến D1 đóng 00:00 UTC |
| Lịch sử D1 | 3,904 nến, 2014-01-14 → 2026-09-04 |
| Lịch sử H4 | 16,230 nến, 2014-01-14 → 2026-09-04 |
| Lịch sử H1 | 24,068 nến, 2022-10-25 → 2026-09-04 |
| `terminal_info.maxbars` | 100000 (loader tải theo chunk) |

**Giờ giao dịch (giờ server = UTC), suy ra từ nến H1 của 8 tuần gần nhất** (package Python `MetaTrader5` không có `symbol_info_session_trade`, hàm đó chỉ có trong MQL5; đoạn mã ở mục 11 vì vậy không chạy được nguyên văn):

- mon: 00:00–24:00; tue: 00:00–24:00; wed: 00:00–24:00; thu: 00:00–24:00; fri: 00:00–21:00; sat: đóng; sun: 21:00–24:00

**Spread theo phiên (bps của mid, 30 ngày tick `COPY_TICKS_INFO`, 2026-08-06 → 2026-09-05, 537,297 tick):**

| Phiên | Số tick | p50 (bps) | p90 (bps) | p50 (points) | p90 (points) |
|---|---|---|---|---|---|
| all | 537,297 | 1.01 | 1.02 | 14 | 14 |
| asia | 121,041 | 1.01 | 1.02 | 14 | 14 |
| london | 110,806 | 1.01 | 1.02 | 14 | 14 |
| overlap | 179,063 | 1.01 | 1.02 | 14 | 14 |
| new_york | 100,675 | 1.01 | 1.02 | 14 | 14 |
| rollover | 8,594 | 1.01 | 1.02 | 14 | 14 |
| other | 17,118 | 2.1 | 4.32 | 29 | 60 |

`rollover` = ±15 phút quanh 00:00 UTC (mốc swap của server); `other` = khoảng 21:00–22:00 UTC sau khi New York đóng và mép tuần, là lúc spread giãn — bot không khớp lệnh trong khoảng này.

**Hệ quả cho mốc quyết định (mục 6):** nến D1 của server đóng lúc 00:00 UTC, tức 2–3 giờ *sau* giờ đóng phiên `CME_FX` (17:00 New York = 21:00/22:00 UTC) và có nến Chủ nhật ngắn (21/22:00–24:00 UTC), nên hai kiểm tra D1-vs-`CME_FX` của `01_feasibility_analysis` (một nến/phiên, đóng nến cách giờ đóng phiên ≤ 60 phút) không thể đạt trên server này; xem `bots/exness_fx_d1/BOT.md` (Decisions log 2026-09-05) cho phương án dự phòng theo nến H4.
