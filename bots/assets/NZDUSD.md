---
symbol: NZDUSD
asset_class: forex
bot: xau_fx_mt5
template_case_study: "fx_pairs"
timezone: UTC
sessions:
  sydney:
    winter: "21:00-06:00"
    summer: "22:00-07:00"
  tokyo:
    winter: "00:00-09:00"
    summer: "00:00-09:00"
  london:
    winter: "08:00-17:00"
    summer: "07:00-16:00"
  new_york:
    winter: "13:00-22:00"
    summer: "12:00-21:00"
decision_times:
  - name: "Quyết định"
    when: "Đóng nến H1 của server (UTC), mọi giờ trong tuần FX; phiên vào feature và vào bộ lọc chi phí, không phải lịch riêng"
  - name: "Khớp lệnh"
    when: "Nến H1 kế tiếp sau quyết định (`execution_delay: next_bar_open`)"
avoid_windows:
  - name: "Giờ đầu tuần"
    when: "22:00-23:00 UTC Chủ nhật (mùa hè 21:00-22:00)"
  - name: "Giờ cuối tuần"
    when: "2 giờ cuối thứ Sáu trước giờ đóng"
  - name: "Rollover"
    when: "±15 phút quanh 00:00 UTC (server midnight)"
  - name: "Khoảng trống sau New York"
    when: "21:00-22:00 UTC (mùa hè) / 22:00-23:00 UTC (mùa đông): spread giãn 2-5 lần"
news:
  - name: "RBNZ"
    when: "7 lần/năm, 02:00 UTC (mùa hè NZ 01:00)"
  - name: "Số liệu Trung Quốc (PMI, xuất nhập khẩu)"
    when: "01:30-03:00 UTC"
  - name: "Sữa (GDT auction)"
    when: "2 lần/tháng, ~14:00 UTC"
  - name: "NFP"
    when: "thứ Sáu đầu tháng 13:30 UTC (mùa hè 12:30)"
  - name: "CPI Mỹ"
    when: "giữa tháng 13:30 UTC (mùa hè 12:30)"
  - name: "FOMC"
    when: "8 lần/năm, quyết định 19:00 UTC, họp báo 19:30 (mùa hè 18:00/18:30)"
verify_on_mt5: [symbol_info.trade_contract_size, volume_min, volume_step, swap_long, swap_short, swap_rollover3days]
---

# NZDUSD — hồ sơ phiên và giờ giao dịch

**Lớp tài sản**: forex · **Bot**: `xau_fx_mt5` · **Khuôn case study**: fx_pairs

**Vai trò trong danh mục**: Cặp hàng hóa/risk-on; tương quan ~0.9 với AUDUSD (`bots/assets/AUDUSD.md` §9) nên phần lớn là cùng một bet với AUDUSD. Spread rộng hơn EURUSD ~2 lần trên tài khoản này, nên đây là cặp dễ rớt ở cổng chi phí H1 nhất. Bot `xau_fx_mt5` giữ nó trong panel làm nguồn feature và chỉ giao dịch nếu cổng chi phí (`experiments/xau_fx_mt5/reports/01_feasibility.md`) cho phép.

**Hợp đồng**: 100000 NZD mỗi lot (`trade_contract_size` = 100000).

## 1. Giờ giao dịch

Thị trường FX mở 22:00 UTC Chủ nhật đến 22:00 UTC thứ Sáu (mùa đông; mùa hè sớm 1 giờ); phiên Wellington/Sydney mở đầu tuần. Trên server này nến H1 đầu tuần là Chủ nhật 21:00/22:00 UTC (`experiments/xau_fx_mt5/data/data_quality.md` §5c).

## 2. Các phiên (UTC)

sydney 21:00-06:00 (mùa hè 22:00-07:00, DST Úc/NZ ngược mùa với Bắc bán cầu), tokyo 00:00-09:00, london 08:00-17:00 (07:00-16:00), new_york 13:00-22:00 (12:00-21:00). `bots/_shared/sessions.py` tính từ múi giờ thật.

## 3. Rollover và swap

Swap tính tại 00:00 UTC (server midnight). Ngày swap ba lần: `swap_rollover3days` = 3 (thứ Tư). Đây là cặp duy nhất trong 8 mã có swap khác 0 trên demo (`swap_long` -1.9 points/lot/đêm).

## 4. Measured on this account (2026-09-05 / 2026-09-06)

Nguồn: `D:/05_Quant/v9 Continuum/config/swaps.json` (snapshot `symbol_info`, 2026-09-05, demo `Exness-MT5Trial7`) và `experiments/xau_fx_mt5/data/raw/_mt5/history_depth.json` (2026-09-06).

| Trường | Giá trị đo được |
|---|---|
| Tên trong Market Watch | `NZDUSDm` |
| `trade_contract_size` | 100000 |
| `volume_min` / `volume_step` / `volume_max` | 0.01 / 0.01 / 200.0 |
| `digits` / `point` | 5 / 1e-05 |
| `swap_long` / `swap_short` (`swap_mode` 1 = points/lot/đêm) | -1.9 / 0.0 (demo; đọc lại trên tài khoản thật) |
| `swap_rollover3days` | 3 = Wednesday |
| `currency_profit` | USD (`tick_value` 1.0) |
| `commission_per_lot_usd` | 0.0 |
| Giờ server | UTC+0 quanh năm (đo trên `BTCUSDm` 2026-09-06: offset 0) |
| Lịch sử D1 | 3,892 nến, 2014-01-14 → 2026-08-21 (cắt tại mốc demo tape) |
| Lịch sử H1 | dày từ 2017-03-06 (trước đó là backfill 1 nến/ngày), ~59,794 nến tải sâu đến 2026-08-21 |
| Spread trung vị theo phiên | `experiments/xau_fx_mt5/data/data_quality.md` §4 và `reports/01_feasibility.md` §C |
