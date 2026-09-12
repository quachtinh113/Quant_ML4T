# Hồ sơ tài sản Exness cho các bot ML4T

Mười tài sản, mỗi tài sản một file. Phần YAML đầu mỗi file là định nghĩa máy đọc được cho `bots/_shared/sessions.py`
(phiên theo UTC mùa đông và mùa hè, mốc quyết định, cửa sổ tránh, lịch tin). Phần Markdown là mô tả đầy đủ cho người đọc.

| Mã | Lớp | Bot | Phiên chính | Mốc quyết định | File |
|---|---|---|---|---|---|
| EURUSD | forex | `exness_fx_d1` | london, overlap, new_york | Giờ đóng nến D1 của server (một mốc/ngày) | [EURUSD.md](EURUSD.md) |
| GBPUSD | forex | `exness_fx_d1` | london, overlap, new_york | Giờ đóng nến D1 của server (một mốc/ngày) | [GBPUSD.md](GBPUSD.md) |
| USDJPY | forex | `exness_fx_d1` | tokyo, london, new_york | Giờ đóng nến D1 của server (một mốc/ngày) | [USDJPY.md](USDJPY.md) |
| AUDUSD | forex | `exness_fx_d1` | sydney, tokyo, london, new_york | Giờ đóng nến D1 của server (một mốc/ngày) | [AUDUSD.md](AUDUSD.md) |
| USDCAD | forex | `exness_fx_d1` | new_york, overlap | Giờ đóng nến D1 của server (một mốc/ngày) | [USDCAD.md](USDCAD.md) |
| XAUUSD | metals | `exness_gold_sess` | london, overlap, new_york, tokyo | Đầu phiên London: 08:00 UTC mùa đông (07:00 mùa hè), sau block mép phiên 30 phút | [XAUUSD.md](XAUUSD.md) |
| XAGUSD | metals | `exness_gold_sess` | london, overlap, new_york, tokyo | Đầu phiên London: 08:00 UTC mùa đông (07:00 mùa hè), sau block mép phiên 30 phút | [XAGUSD.md](XAGUSD.md) |
| BTCUSD | crypto | `exness_btc_8h` | new_york, us_cash, london | 00:00, 08:00, 16:00 UTC (cadence `8_hour_funding_aligned` của case study) | [BTCUSD.md](BTCUSD.md) |
| US500 | indices | `exness_usidx_sess` | us_cash, new_york, london | Mở phiên tiền mặt 14:30 UTC hoặc 15:00 sau opening range (mùa hè sớm 1 giờ) | [US500.md](US500.md) |
| USTEC | indices | `exness_usidx_sess` | us_cash, new_york, london | Mở phiên tiền mặt 14:30 UTC hoặc 15:00 sau opening range (mùa hè sớm 1 giờ) | [USTEC.md](USTEC.md) |

## Quy ước chung

- Mọi giờ trong các file là **UTC**. Mùa đông là giờ chuẩn Mỹ và Anh (đầu tháng 11 đến giữa tháng 3), mùa hè sớm hơn một giờ.
  Tokyo không đổi giờ; DST Úc ngược mùa. `sessions.py` phải tính từ múi giờ thật (`zoneinfo`), không hard-code.
- Giờ giao dịch, hợp đồng, swap và ngày swap ba lần của **từng mã trên tài khoản của bạn** đọc từ MT5 (`symbol_info`,
  `symbol_info_session_trade`), theo đoạn mã ở cuối mỗi file. Số đo thật ghi đè mọi ước lượng trong file.
- "Tránh" nghĩa là không đặt lệnh thị trường mới trong cửa sổ, và lấy spread phân vị 90 khi tính chi phí cho cửa sổ đó.
  Feature vẫn tính trên mọi nến. Một cửa sổ chỉ quay lại lịch quyết định khi backtest theo phiên chứng minh nó sống sót qua chi phí.
- Quyết định tại giá đóng nến, khớp ở nến kế tiếp (`decision_snapshot: bar_close`, `execution_delay: 1_bar`).
- Chọn phiên tốt nhất trong N phiên là N lần thử; N vào số trial của Deflated Sharpe (Ch16 `12_dsr_validation`).

## Cách dùng với bộ mentor

- Thiết kế chia bot và ba mức backtest theo phiên: `.claude/skills/ml4t-quant-bot-mentor/references/bot-portfolio-exness.md`.
- Adapter và loader MT5: `.claude/skills/ml4t-quant-bot-mentor/references/mt5-exness-broker.md`.
- Giao `ml4t-bot-builder` đọc thư mục này khi viết `bots/_shared/sessions.py` và khối `costs` của từng `setup.yaml`.
