---
symbol: USTEC
asset_class: indices
bot: exness_usidx_sess
template_case_study: "cme_futures cho feature, fx_pairs cho pipeline; mô hình chuỗi thời gian từng chỉ số"
timezone: UTC
sessions:
  us_cash:
    winter: "14:30-21:00"
    summer: "13:30-20:00"
  new_york:
    winter: "13:00-22:00"
    summer: "12:00-21:00"
  london:
    winter: "08:00-17:00"
    summer: "07:00-16:00"
decision_times:
  - name: "Quyết định 1"
    when: "Mở phiên tiền mặt 14:30 UTC hoặc 15:00 sau opening range (mùa hè sớm 1 giờ)"
  - name: "Quyết định 2"
    when: "Đóng phiên 21:00 UTC cho nhãn giữ qua đêm"
  - name: "Backtest theo phiên (mức 2)"
    when: "`SESSION_FILTER`: us_cash, overnight, london"
avoid_windows:
  - name: "Giờ đầu tuần"
    when: "22:00-23:00 UTC Chủ nhật (mùa hè 21:00-22:00)"
  - name: "Giờ cuối tuần"
    when: "2 giờ cuối thứ Sáu trước giờ đóng"
  - name: "Rollover"
    when: "±15 phút quanh giờ rollover"
  - name: "Nghỉ hằng ngày Globex"
    when: "22:00-23:00 UTC mùa đông"
  - name: "Ngoài giờ tiền mặt"
    when: "23:00-14:30 UTC"
  - name: "FOMC"
    when: "19:00-20:00 UTC"
  - name: "30 phút đầu phiên tiền mặt"
    when: "14:30-15:00 UTC"
news:
  - name: "Lợi nhuận Big Tech"
    when: "Sau 21:00 UTC, mùa báo cáo"
  - name: "Lợi suất Mỹ 10 năm"
    when: "Liên tục"
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
  - name: "Mùa báo cáo lợi nhuận"
    when: "Sau 21:00 UTC và trước 14:30 UTC, tháng 1, 4, 7, 10"
  - name: "Đáo hạn quyền chọn tháng"
    when: "thứ Sáu thứ ba, 21:00 UTC; triple witching tháng 3, 6, 9, 12"
  - name: "Mất cân bằng MOC"
    when: "20:50 UTC mùa đông (19:50 mùa hè)"
verify_on_mt5: [symbol_info.trade_contract_size, volume_min, volume_step, swap_long, swap_short, swap_rollover3days, symbol_info_session_trade]
---

# USTEC — hồ sơ phiên và giờ giao dịch

**Lớp tài sản**: indices · **Bot**: `exness_usidx_sess` · **Khuôn case study**: cme_futures cho feature, fx_pairs cho pipeline; mô hình chuỗi thời gian từng chỉ số

**Vai trò trong danh mục**: Beta công nghệ; biến động cao hơn US500, nhạy lãi suất và lợi nhuận Big Tech

**Hợp đồng**: Thường 1 đơn vị chỉ số mỗi lot trên Exness; tên mã có thể là USTEC / NAS100 / NDX; xác minh `trade_contract_size` và tên trong Market Watch

## 1. Giờ giao dịch

Tài sản cơ sở giao dịch theo CME Globex: 23:00 UTC Chủ nhật đến 22:00 UTC thứ Sáu, nghỉ hằng ngày 22:00-23:00 UTC (mùa hè sớm hơn 1 giờ). CFD trên Exness thường theo sát lịch này, có thể lệch vài phút; đọc từ `symbol_info_session_trade` và ghi vào `sessions.py`.

## 2. Các phiên (UTC)

| Phiên | Mùa đông | Mùa hè | Ghi chú |
|---|---|---|---|
| `us_cash` | 14:30-21:00 | 13:30-20:00 | Phiên tiền mặt NYSE/Nasdaq; đấu giá mở 14:30 và đóng 21:00 (mùa đông) |
| `new_york` | 13:00-22:00 | 12:00-21:00 | Tin Mỹ 13:30, FOMC 19:00 (mùa đông) |
| `london` | 08:00-17:00 | 07:00-16:00 | Phiên thanh khoản lớn nhất của FX và kim loại |

Mùa đông là khoảng đầu tháng 11 đến giữa tháng 3 (giờ chuẩn Mỹ và Anh). Mùa hè sớm hơn một giờ. Tokyo không đổi giờ. `bots/_shared/sessions.py` tính từ múi giờ thật, không hard-code bảng này.

## 3. Lịch tin và sự kiện định kỳ

| Sự kiện | Thời điểm | Ảnh hưởng |
|---|---|---|
| Lợi nhuận Big Tech | Sau 21:00 UTC, mùa báo cáo | Gap lớn qua đêm |
| Lợi suất Mỹ 10 năm | Liên tục | Nhạy hơn US500 |
| NFP | thứ Sáu đầu tháng 13:30 UTC (mùa hè 12:30) | Biến động lớn nhất tháng cho USD, vàng, chỉ số |
| CPI Mỹ | giữa tháng 13:30 UTC (mùa hè 12:30) | Lãi suất thực; vàng và USTEC phản ứng mạnh |
| FOMC | 8 lần/năm, quyết định 19:00 UTC, họp báo 19:30 (mùa hè 18:00/18:30) | Cửa sổ 2 giờ spread giãn trên mọi tài sản USD |
| Đơn xin trợ cấp thất nghiệp | thứ Năm 13:30 UTC | Nhỏ, nhưng đều đặn |
| ISM PMI | ngày làm việc đầu tháng 15:00 UTC (mùa hè 14:00) | Chỉ số và USD |
| Bán lẻ, GDP | 13:30 UTC | Chỉ số |
| Mùa báo cáo lợi nhuận | Sau 21:00 UTC và trước 14:30 UTC, tháng 1, 4, 7, 10 | Gap qua đêm |
| Đáo hạn quyền chọn tháng | thứ Sáu thứ ba, 21:00 UTC; triple witching tháng 3, 6, 9, 12 | Khối lượng và pin |
| Mất cân bằng MOC | 20:50 UTC mùa đông (19:50 mùa hè) | Biến động 10 phút cuối |

## 4. Rollover và swap

Swap tính tại giờ rollover của server (thường quanh 22:00 UTC mùa đông, 21:00 mùa hè; xác minh trên tài khoản). Ngày swap ba lần đọc từ `symbol_info(...).swap_rollover3days` (thường thứ Tư cho FX và kim loại). Spread giãn 15 đến 30 phút quanh rollover; không đặt lệnh thị trường trong cửa sổ này.

## 5. Giờ nên tránh vào lệnh thị trường

| Cửa sổ | Thời gian | Lý do |
|---|---|---|
| Giờ đầu tuần | 22:00-23:00 UTC Chủ nhật (mùa hè 21:00-22:00) | Gap cuối tuần, spread giãn, thanh khoản mỏng |
| Giờ cuối tuần | 2 giờ cuối thứ Sáu trước giờ đóng | Đóng vị thế của các quỹ, spread giãn, rủi ro gap sang thứ Hai |
| Rollover | ±15 phút quanh giờ rollover | Spread giãn, swap tính |
| Nghỉ hằng ngày Globex | 22:00-23:00 UTC mùa đông | Không giá; gap nhỏ |
| Ngoài giờ tiền mặt | 23:00-14:30 UTC | Spread CFD rộng hơn 2-4 lần; chỉ tính feature qua đêm, không vào lệnh trừ khi nhãn qua đêm được backtest riêng |
| FOMC | 19:00-20:00 UTC | Spread giãn, biến động hai chiều |
| 30 phút đầu phiên tiền mặt | 14:30-15:00 UTC | Mép phiên; hình chữ U của khối lượng (Ch03 06); chỉ vào sau opening range nếu backtest chứng minh |

Tránh ở đây nghĩa là không đặt lệnh thị trường mới, và spread trong cửa sổ đó lấy phân vị 90 khi tính chi phí. Feature vẫn được tính trên mọi nến. Một cửa sổ chỉ được đưa trở lại lịch quyết định khi backtest theo phiên chứng minh nó sống sót qua chi phí.

## 6. Mốc quyết định và khớp lệnh của bot

| Mốc | Thời điểm | Ghi chú |
|---|---|---|
| Quyết định 1 | Mở phiên tiền mặt 14:30 UTC hoặc 15:00 sau opening range (mùa hè sớm 1 giờ) | Bot `exness_usidx_sess`, nến H1 |
| Quyết định 2 | Đóng phiên 21:00 UTC cho nhãn giữ qua đêm | Tách lợi nhuận qua đêm và trong phiên (Ch08 01) |
| Backtest theo phiên (mức 2) | `SESSION_FILTER`: us_cash, overnight, london | Mỗi bộ lọc một hash |

Quyết định luôn tại giá đóng nến, khớp ở nến kế tiếp. Mọi cửa sổ feature của bot theo phiên bị chặn tại biên phiên.

## 7. Feature từ repo

- Như US500
- Chênh lệch USTEC trừ US500 làm feature cấu trúc (Ch08 03)
- Beta với BTCUSD trong giờ Mỹ

## 8. Bằng chứng và mẫu trong repo

- NQ trong `case_studies/cme_futures`
- QQQ trong `case_studies/etfs`
- `case_studies/nasdaq100_microstructure` — 15 phút, `edge_block`, `decision.cadence_by_label`

## 9. Tương quan và vị trí trong danh mục

Tương quan cao với US500.

## 10. Ghi chú riêng

Case study `nasdaq100_microstructure` là bằng chứng gần nhất về tín hiệu intraday trên rổ này: tín hiệu thô thua chi phí, chỉ hồi phục sau khi sàng lọc và chọn mô hình có kỷ luật. Kỳ vọng tương tự cho CFD.

## 11. Xác minh trên tài khoản Exness trước khi dùng

```python
import MetaTrader5 as mt5
mt5.initialize()
s = "USTEC"  # đổi sang tên đúng trong Market Watch
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
| Tên trong Market Watch | `USTECm` (suffix `m`; `path` = `Standard\Indices\USTECm`) |
| `trade_contract_size` | 1 |
| `volume_min` / `volume_step` / `volume_max` | 0.05 / 0.01 / 500.0 |
| `digits` / `point` | 2 / 0.01 |
| `swap_long` / `swap_short` (`swap_mode` 1 = points/lot/đêm) | -592.7 / 0.0 |
| `swap_rollover3days` | 5 = Friday (swap x3) |
| Tiền tệ base / profit / margin | USD / USD / USD |
| Giờ server | UTC+0 quanh năm (đo `ServerClock.measure` trên `BTCUSDm`: offset 0; tuần FX mở Sun 22:00 server tháng 1 và Sun 21:00 server tháng 7, tức server **không** theo DST New York). Nến D1 đóng 00:00 UTC |
| Lịch sử D1 | 2,191 nến, 2019-07-16 → 2026-09-04 |
| Lịch sử H4 | 11,217 nến, 2019-07-16 → 2026-09-04 |
| Lịch sử H1 | 22,553 nến, 2022-10-25 → 2026-09-04 |
| `terminal_info.maxbars` | 100000 (loader tải theo chunk) |

**Giờ giao dịch (giờ server = UTC), suy ra từ nến H1 của 8 tuần gần nhất** (package Python `MetaTrader5` không có `symbol_info_session_trade`, hàm đó chỉ có trong MQL5; đoạn mã ở mục 11 vì vậy không chạy được nguyên văn):

- mon: 00:00–21:00, 22:00–24:00; tue: 00:00–21:00, 22:00–24:00; wed: 00:00–21:00, 22:00–24:00; thu: 00:00–21:00, 22:00–24:00; fri: 00:00–21:00; sat: đóng; sun: 22:00–24:00

Spread theo phiên: chưa đo cho mã này (B6 chỉ đo 5 cặp FX của `exness_fx_d1`); đo bằng `bots._shared.costs_mt5.measure_spreads` khi bot của mã này mở.
