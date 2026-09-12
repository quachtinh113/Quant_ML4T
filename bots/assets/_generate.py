# -*- coding: utf-8 -*-
"""Generate bots/assets/<SYMBOL>.md — one asset spec per Exness instrument."""
from pathlib import Path
import textwrap

OUT = Path(r"D:\05_Quant\machine-learning-for-trading\bots\assets")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- shared session definitions (UTC)
# winter = Nov..Mar (US/UK standard time), summer = late Mar..late Oct (BST/EDT). Tokyo has no DST.
SESSIONS = {
    "sydney":   dict(winter="21:00-06:00", summer="22:00-07:00", note="Mở đầu tuần FX; DST Úc ngược mùa với Bắc bán cầu, biên có thể lệch 1 giờ"),
    "tokyo":    dict(winter="00:00-09:00", summer="00:00-09:00", note="Không có DST"),
    "london":   dict(winter="08:00-17:00", summer="07:00-16:00", note="Phiên thanh khoản lớn nhất của FX và kim loại"),
    "new_york": dict(winter="13:00-22:00", summer="12:00-21:00", note="Tin Mỹ 13:30, FOMC 19:00 (mùa đông)"),
    "overlap":  dict(winter="13:00-17:00", summer="12:00-16:00", note="London giao New York: spread hẹp nhất, biên độ lớn nhất"),
    "us_cash":  dict(winter="14:30-21:00", summer="13:30-20:00", note="Phiên tiền mặt NYSE/Nasdaq; đấu giá mở 14:30 và đóng 21:00 (mùa đông)"),
    "globex":   dict(winter="23:00-22:00 (Chủ nhật → thứ Sáu), nghỉ hằng ngày 22:00-23:00", summer="22:00-21:00, nghỉ 21:00-22:00", note="Lịch CME Globex của vàng, bạc, ES, NQ"),
    "crypto":   dict(winter="24/7", summer="24/7", note="Không có giờ đóng; thanh khoản thấp cuối tuần"),
}

FX_HOURS = ("Thị trường FX mở từ 22:00 UTC Chủ nhật (Sydney) đến 22:00 UTC thứ Sáu (mùa đông; mùa hè sớm hơn 1 giờ). "
            "Trên Exness, giờ giao dịch thực tế của từng mã và giờ nghỉ hằng ngày đọc từ `symbol_info_session_trade`; "
            "không suy từ lịch thị trường.")
GLOBEX_HOURS = ("Tài sản cơ sở giao dịch theo CME Globex: 23:00 UTC Chủ nhật đến 22:00 UTC thứ Sáu, nghỉ hằng ngày 22:00-23:00 UTC "
                "(mùa hè sớm hơn 1 giờ). CFD trên Exness thường theo sát lịch này, có thể lệch vài phút; đọc từ "
                "`symbol_info_session_trade` và ghi vào `sessions.py`.")
CRYPTO_HOURS = ("Giao dịch 24/7. Exness có thể có khoảng bảo trì ngắn hằng tuần; đọc từ `symbol_info_session_trade` cho cả 7 ngày.")

US_NEWS = [
    ("NFP", "thứ Sáu đầu tháng 13:30 UTC (mùa hè 12:30)", "Biến động lớn nhất tháng cho USD, vàng, chỉ số"),
    ("CPI Mỹ", "giữa tháng 13:30 UTC (mùa hè 12:30)", "Lãi suất thực; vàng và USTEC phản ứng mạnh"),
    ("FOMC", "8 lần/năm, quyết định 19:00 UTC, họp báo 19:30 (mùa hè 18:00/18:30)", "Cửa sổ 2 giờ spread giãn trên mọi tài sản USD"),
    ("Đơn xin trợ cấp thất nghiệp", "thứ Năm 13:30 UTC", "Nhỏ, nhưng đều đặn"),
    ("ISM PMI", "ngày làm việc đầu tháng 15:00 UTC (mùa hè 14:00)", "Chỉ số và USD"),
    ("Bán lẻ, GDP", "13:30 UTC", "Chỉ số"),
]
ROLLOVER = ("Swap tính tại giờ rollover của server (thường quanh 22:00 UTC mùa đông, 21:00 mùa hè; xác minh trên tài khoản). "
            "Ngày swap ba lần đọc từ `symbol_info(...).swap_rollover3days` (thường thứ Tư cho FX và kim loại). "
            "Spread giãn 15 đến 30 phút quanh rollover; không đặt lệnh thị trường trong cửa sổ này.")

COMMON_AVOID = [
    ("Giờ đầu tuần", "22:00-23:00 UTC Chủ nhật (mùa hè 21:00-22:00)", "Gap cuối tuần, spread giãn, thanh khoản mỏng"),
    ("Giờ cuối tuần", "2 giờ cuối thứ Sáu trước giờ đóng", "Đóng vị thế của các quỹ, spread giãn, rủi ro gap sang thứ Hai"),
    ("Rollover", "±15 phút quanh giờ rollover", "Spread giãn, swap tính"),
]

def yaml_list(items, indent=2):
    pad = " " * indent
    return "\n".join(f"{pad}- {i}" for i in items)

ASSETS = []

def fx(symbol, base, quote, bot, role, sessions_active, news_extra, avoid_extra, features, notes, corr):
    ASSETS.append(dict(
        symbol=symbol, cls="forex", bot=bot, template="fx_pairs",
        contract="100000 đơn vị tiền cơ sở mỗi lot (xác minh `trade_contract_size`)",
        hours=FX_HOURS, role=role, sessions_active=sessions_active, news=news_extra + US_NEWS,
        avoid=COMMON_AVOID + avoid_extra, features=features, notes=notes, corr=corr,
        decisions=[
            ("Quyết định", "Giờ đóng nến D1 của server (một mốc/ngày)", "Bot `exness_fx_d1`, cadence daily như `fx_pairs`"),
            ("Khớp lệnh", "Nến kế tiếp sau quyết định (`execution_delay: next_bar_open`)", "Không khớp tại giá vừa nhìn thấy"),
            ("Backtest theo phiên (mức 1)", "Ứng viên `decision.snapshot`: đóng nến server, 08:00 UTC London open, 13:00 UTC NY open", "Mỗi mốc một experiment, nhãn tính lại theo mốc, hash riêng"),
        ],
        repo_evidence=[f"`case_studies/fx_pairs/config/setup.yaml` có {symbol[:3]}_{symbol[3:]} trong universe 20 cặp G10",
                       "`02_financial_data_universe/12_fx_pairs_eda.py` — cấu trúc dữ liệu FX daily",
                       "`25_live_trading/11_fx_deployment_loop.py` — vòng lặp triển khai FX (IB), mẫu cho adapter MT5"],
    ))

fx("EURUSD", "EUR", "USD", "exness_fx_d1",
   "Cặp thanh khoản cao nhất thế giới, spread hẹp nhất; neo của cross-section FX",
   ["london", "overlap", "new_york"],
   [("ECB", "8 lần/năm, quyết định 13:15 UTC, họp báo 13:45 (mùa hè 12:15/12:45)", "Cửa sổ 1 giờ"),
    ("PMI flash Eurozone", "~09:00 UTC (mùa hè 08:00)", "Nhỏ đến vừa"),
    ("Số liệu Đức (ZEW, IFO, CPI)", "07:00-10:00 UTC", "Vừa")],
   [("Phiên Á", "00:00-07:00 UTC", "Biên độ thấp, tín hiệu momentum trong ngày yếu; chỉ dùng để tính feature")],
   ["Momentum 1/3/6 tháng trên nến D1 (Ch08 01)", "Carry từ chênh lệch lãi suất ECB và Fed (Ch08 04, FRED)",
    "Chỉ số USD làm feature cấu trúc (Ch08 03)", "Biến động thực hiện và z-score cross-section (`fx_pairs` 03)"],
   "Biến động thấp so với GBP và JPY nên chi phí chiếm tỷ trọng lớn hơn trong lợi nhuận; cổng chi phí là nơi EURUSD dễ rớt nhất dù spread hẹp.",
   "Nghịch với USDCHF rất mạnh; đó là lý do USDCHF không nằm trong 10 cặp.")

fx("GBPUSD", "GBP", "USD", "exness_fx_d1",
   "Biến động cao hơn EURUSD cùng phiên London; nhạy với tin Anh",
   ["london", "overlap", "new_york"],
   [("BoE", "8 lần/năm, 12:00 UTC (mùa hè 11:00)", "Cửa sổ 1 giờ, spread giãn mạnh"),
    ("CPI, GDP, việc làm Anh", "07:00 UTC (mùa hè 06:00)", "Vừa đến lớn"),
    ("PMI Anh", "09:30 UTC (mùa hè 08:30)", "Vừa")],
   [("Phiên Á", "00:00-07:00 UTC", "Biên độ thấp"),
    ("Sự kiện chính trị Anh", "Ngày bầu cử, ngân sách", "Spread giãn kéo dài; đánh cờ sự kiện")],
   ["Momentum và mean reversion nến D1", "Carry BoE và Fed", "Biến động thực hiện cao hơn nhóm, cần chuẩn hóa theo vol (Ch08 01)"],
   "Đuôi phân phối dày hơn EURUSD; stop theo MAE/MFE (Ch07 04) đo riêng cho cặp này.",
   "Tương quan dương với EURUSD nhưng đủ khác để tăng độ rộng.")

fx("USDJPY", "USD", "JPY", "exness_fx_d1",
   "Cặp carry kinh điển; chạy theo lợi suất Mỹ và tâm lý risk-off",
   ["tokyo", "london", "new_york"],
   [("BoJ", "8 lần/năm, khoảng 03:00-04:00 UTC, giờ không cố định", "Cửa sổ có thể kéo dài; spread giãn"),
    ("Can thiệp của Bộ Tài chính Nhật", "Không báo trước, thường trong giờ Tokyo hoặc đầu London", "Biến động 2-4% trong vài phút; breaker phải bắt được"),
    ("Tokyo CPI, Tankan", "23:30 UTC hôm trước / 23:50 UTC", "Vừa"),
    ("Lợi suất trái phiếu Mỹ 10 năm", "Liên tục; đấu giá 18:00 UTC", "Động lực chính")],
   [("Giờ Tokyo mở", "00:00-00:30 UTC", "Spread giãn ngắn"),
    ("Cửa sổ can thiệp", "Khi USDJPY ở vùng đỉnh lịch sử", "Đánh cờ rủi ro sự kiện, giảm size")],
   ["Carry: chênh lệch lãi suất Fed và BoJ là feature mạnh nhất (Ch08 04)", "Momentum D1", "Chênh lệch lợi suất 2 năm và 10 năm Mỹ trừ Nhật (FRED)", "Cờ phiên Tokyo"],
   "Phiên Tokyo có ý nghĩa với cặp này nhiều hơn các cặp EUR/GBP; backtest mức 1 nên thử mốc quyết định 09:00 UTC (đóng phiên Tokyo).",
   "Tương quan thấp với EURUSD; đóng góp độ rộng lớn.")

fx("AUDUSD", "AUD", "USD", "exness_fx_d1",
   "Commodity currency; gắn với kim loại, Trung Quốc và khẩu vị rủi ro",
   ["sydney", "tokyo", "london", "new_york"],
   [("RBA", "thứ Ba đầu tháng, 03:30 UTC (mùa hè Úc 03:30, kiểm tra)", "Cửa sổ 1 giờ"),
    ("Việc làm, CPI Úc", "00:30 UTC", "Vừa đến lớn"),
    ("Số liệu Trung Quốc (PMI, GDP, tín dụng)", "01:30-02:00 UTC, không DST", "Lớn với AUD và kim loại")],
   [("Giờ Sydney mở", "21:00-22:00 UTC", "Thanh khoản rất mỏng, spread giãn"),
    ("Trước số liệu Trung Quốc", "01:00-02:00 UTC", "Đánh cờ")],
   ["Feature liên thị trường với XAUUSD và đồng (Ch08 03)", "Momentum D1", "Carry RBA và Fed", "Cờ phiên Á"],
   "Cặp này chia sẻ động lực với bot vàng; danh mục cấp trên (Ch17) sẽ thấy tương quan và giảm trọng số.",
   "Tương quan dương với NZDUSD gần 0.9, nên NZDUSD bị loại.")

fx("USDCAD", "USD", "CAD", "exness_fx_d1",
   "Gắn với dầu và số liệu Mỹ; động lực khác bốn cặp còn lại",
   ["new_york", "overlap"],
   [("BoC", "8 lần/năm, 14:45 UTC (mùa hè 13:45)", "Cửa sổ 1 giờ"),
    ("Việc làm Canada", "13:30 UTC, thường cùng ngày NFP", "Biến động kép"),
    ("Tồn kho dầu EIA", "thứ Tư 15:30 UTC (mùa hè 14:30)", "Qua kênh dầu"),
    ("OPEC+", "Theo lịch họp", "Qua kênh dầu")],
   [("Phiên Á và đầu London", "00:00-12:00 UTC", "Biên độ thấp, spread rộng hơn EURUSD"),
    ("Cùng lúc NFP và việc làm Canada", "13:30 UTC", "Biến động cực đoan")],
   ["Feature liên thị trường với giá dầu WTI (CL trong `cme_futures`)", "Momentum D1", "Carry BoC và Fed", "Cờ phiên NY"],
   "Nhiều nhất trong 10 cặp phụ thuộc vào một tài sản khác (dầu); feature cấu trúc Ch08 03 là lý do tồn tại của cặp này trong danh mục.",
   "Tương quan thấp với EUR/GBP/JPY.")

# ---------------------------------------------------------------- metals
def metal(symbol, bot, role, contract, news_extra, avoid_extra, features, notes, corr, evid):
    ASSETS.append(dict(
        symbol=symbol, cls="metals", bot=bot, template="fx_pairs pipeline + mô hình chuỗi thời gian từng tài sản (Ch16 04, 05)",
        contract=contract, hours=GLOBEX_HOURS, role=role,
        sessions_active=["london", "overlap", "new_york", "tokyo"],
        news=news_extra + US_NEWS,
        avoid=COMMON_AVOID + [("Nghỉ hằng ngày Globex", "22:00-23:00 UTC mùa đông (21:00-22:00 mùa hè)", "Không có giá; lệnh chờ không khớp; gap nhỏ khi mở lại")] + avoid_extra,
        features=features, notes=notes, corr=corr,
        decisions=[
            ("Quyết định 1", "Đầu phiên London: 08:00 UTC mùa đông (07:00 mùa hè), sau block mép phiên 30 phút", "Bot `exness_gold_sess`, nến H1"),
            ("Quyết định 2", "Đầu phiên New York: 13:00 UTC mùa đông (12:00 mùa hè), hoặc 14:00 sau cửa sổ tin 13:30", "Hai biến thể là hai spec backtest"),
            ("Thoát", "Ba nhãn: đóng cuối phiên, giữ qua đêm cộng swap, triple-barrier trong phiên", "Ch07 03, Ch19 02"),
            ("Backtest theo phiên (mức 2)", "`SESSION_FILTER` trong stage backtest: london, ny, overlap", "Mỗi bộ lọc một hash; N phiên = N trial trong DSR"),
        ],
        repo_evidence=evid,
    ))

metal("XAUUSD", "exness_gold_sess",
      "Tài sản trú ẩn; chạy theo lãi suất thực Mỹ, USD và dòng tiền ETF; nhạy tin Mỹ",
      "100 oz mỗi lot (xác minh `trade_contract_size`)",
      [("LBMA gold price", "10:30 UTC và 15:00 UTC", "Mốc thanh khoản; tăng khối lượng quanh giờ này"),
       ("Số liệu Trung Quốc và giờ Thượng Hải", "01:30-03:00 UTC", "Nhu cầu vật chất; biên độ phiên Á"),
       ("Lợi suất thực Mỹ (TIPS 10 năm)", "Liên tục", "Động lực chính (Ch08 04, FRED)")],
      [("Trước và sau NFP, CPI, FOMC", "±30 phút", "Spread giãn 3-5 lần; chỉ vào lệnh nếu nhãn được thiết kế cho cửa sổ tin")],
      ["Lợi suất thực và USD làm feature cấu trúc (Ch08 03, 04)", "Tỷ số vàng/bạc", "Momentum và mean reversion H1 chặn tại biên phiên",
       "Tách lợi nhuận qua đêm và trong phiên (Ch08 01)", "Cờ phiên và cờ mép phiên (`edge_block` như `nasdaq100_microstructure`)"],
      "Vàng là tài sản mà backtest theo phiên có ý nghĩa nhất trong 10 cặp: biên độ và spread khác hẳn giữa phiên Á, London và NY.",
      "Tương quan dương với AUDUSD và XAGUSD, nghịch với USD.",
      ["GC trong `case_studies/cme_futures/config/setup.yaml`", "GLD và IAU trong `case_studies/etfs`", "`02_financial_data_universe/05_futures_session_aggregation.py` — ranh giới phiên CME"])

metal("XAGUSD", "exness_gold_sess",
      "Cặp đôi của vàng với beta cao hơn và spread rộng hơn; nửa kim loại quý, nửa công nghiệp",
      "5000 oz mỗi lot (xác minh `trade_contract_size`)",
      [("LBMA silver price", "12:00 UTC", "Mốc thanh khoản"),
       ("Số liệu công nghiệp Trung Quốc", "01:30-02:00 UTC", "Kênh nhu cầu công nghiệp")],
      [("Trước tin Mỹ", "±30 phút", "Spread giãn mạnh hơn vàng"),
       ("Phiên Á", "00:00-07:00 UTC", "Thanh khoản mỏng, spread rộng; chỉ tính feature")],
      ["Tỷ số vàng/bạc và độ lệch của nó (mean reversion cấu trúc, Ch08 03)", "Beta với vàng", "Momentum H1 chặn tại biên phiên", "Biến động thực hiện cao, chuẩn hóa theo vol"],
      "Chi phí là rào cản chính: spread tính theo bps cao hơn vàng đáng kể; cổng chi phí có thể loại bạc khỏi bot dù tín hiệu tốt. Chấp nhận kết quả đó.",
      "Tương quan với vàng khoảng 0.8; đóng góp độ rộng thấp nhưng cho feature tỷ số.",
      ["SI trong `case_studies/cme_futures`", "SLV trong `case_studies/etfs`"])

# ---------------------------------------------------------------- crypto
ASSETS.append(dict(
    symbol="BTCUSD", cls="crypto", bot="exness_btc_8h", template="crypto_perps_funding (bỏ feature funding, vì CFD không có funding rate)",
    contract="1 BTC mỗi lot (xác minh `trade_contract_size`, `volume_min` có thể là 0.01)",
    hours=CRYPTO_HOURS, role="Động lực gần như độc lập với FX và chỉ số; chạy 24/7",
    sessions_active=["new_york", "us_cash", "london"],
    news=[("Mốc funding của sàn perp", "00:00, 08:00, 16:00 UTC", "Dòng tiền xoay quanh mốc; đây là mốc quyết định của bot"),
          ("Đáo hạn quyền chọn Deribit", "thứ Sáu 08:00 UTC, lớn nhất cuối tháng và cuối quý", "Biến động và pin quanh strike lớn"),
          ("Đáo hạn hợp đồng tương lai CME BTC", "thứ Sáu cuối tháng 15:00 UTC", "Biến động"),
          ("Dòng tiền ETF BTC giao ngay Mỹ", "Công bố sau phiên Mỹ", "Xu hướng nhiều ngày"),
          ("Sự cố sàn, thanh lý dây chuyền", "Không báo trước, thường ngoài giờ Mỹ", "Breaker phải bắt được")] + US_NEWS[:3],
    avoid=[("Cuối tuần", "thứ Bảy và Chủ nhật", "Thanh khoản mỏng, spread CFD rộng, biến động đột ngột; đánh cờ `weekend`, backtest riêng có và không có cuối tuần"),
           ("Đêm Mỹ", "04:00-07:00 UTC", "Khối lượng thấp nhất trong ngày"),
           ("Rollover swap CFD", "Giờ rollover server, hằng ngày kể cả cuối tuần (xác minh)", "Swap crypto lớn; giữ qua nhiều mốc phải đưa vào nhãn")],
    features=["Momentum và mean reversion nến 8 giờ và D1 (feature của `crypto_perps_funding` 03, trừ funding)", "Biến động thực hiện và khối lượng tương đối",
              "Cờ cuối tuần, cờ giờ Mỹ", "Beta với USTEC trong giờ Mỹ (Ch08 03)"],
    notes="Điểm khác lớn nhất so với case study gốc: không có funding rate, không có basis; chiến lược funding arbitrage của chương 12 không áp dụng được. Chỉ phần momentum, vol, khối lượng còn dùng được. Swap CFD hằng ngày là chi phí giữ lệnh chính.",
    corr="Tương quan với USTEC dương trong giờ Mỹ, gần 0 ngoài giờ; cuối tuần độc lập hoàn toàn.",
    decisions=[("Quyết định", "00:00, 08:00, 16:00 UTC (cadence `8_hour_funding_aligned` của case study)", "Bot `exness_btc_8h`"),
               ("Khớp lệnh", "Nến kế tiếp", ""),
               ("Backtest theo phiên (mức 2)", "`SESSION_FILTER`: chỉ mốc 16:00 (giờ Mỹ), chỉ ngày trong tuần, tất cả", "Ba spec, ba hash")],
    repo_evidence=["BTCUSDT trong `case_studies/crypto_perps_funding/config/setup.yaml`, cadence 8 giờ",
                   "`25_live_trading/09_crypto_funding_deployment_loop.py` — Binance nghiên cứu, Alpaca thực thi; mẫu venue split",
                   "`21_rl_execution_hedging/04_crypto_execution_rl.py` — thực thi crypto"],
))

# ---------------------------------------------------------------- indices
def index(symbol, alias, bot, role, news_extra, features, notes, corr, evid):
    ASSETS.append(dict(
        symbol=symbol, cls="indices", bot=bot, template="cme_futures cho feature, fx_pairs cho pipeline; mô hình chuỗi thời gian từng chỉ số",
        contract=f"Thường 1 đơn vị chỉ số mỗi lot trên Exness; tên mã có thể là {alias}; xác minh `trade_contract_size` và tên trong Market Watch",
        hours=GLOBEX_HOURS, role=role,
        sessions_active=["us_cash", "new_york", "london"],
        news=news_extra + US_NEWS + [("Mùa báo cáo lợi nhuận", "Sau 21:00 UTC và trước 14:30 UTC, tháng 1, 4, 7, 10", "Gap qua đêm"),
                                     ("Đáo hạn quyền chọn tháng", "thứ Sáu thứ ba, 21:00 UTC; triple witching tháng 3, 6, 9, 12", "Khối lượng và pin"),
                                     ("Mất cân bằng MOC", "20:50 UTC mùa đông (19:50 mùa hè)", "Biến động 10 phút cuối")],
        avoid=COMMON_AVOID + [("Nghỉ hằng ngày Globex", "22:00-23:00 UTC mùa đông", "Không giá; gap nhỏ"),
                              ("Ngoài giờ tiền mặt", "23:00-14:30 UTC", "Spread CFD rộng hơn 2-4 lần; chỉ tính feature qua đêm, không vào lệnh trừ khi nhãn qua đêm được backtest riêng"),
                              ("FOMC", "19:00-20:00 UTC", "Spread giãn, biến động hai chiều"),
                              ("30 phút đầu phiên tiền mặt", "14:30-15:00 UTC", "Mép phiên; hình chữ U của khối lượng (Ch03 06); chỉ vào sau opening range nếu backtest chứng minh")],
        features=features, notes=notes, corr=corr,
        decisions=[("Quyết định 1", "Mở phiên tiền mặt 14:30 UTC hoặc 15:00 sau opening range (mùa hè sớm 1 giờ)", "Bot `exness_usidx_sess`, nến H1"),
                   ("Quyết định 2", "Đóng phiên 21:00 UTC cho nhãn giữ qua đêm", "Tách lợi nhuận qua đêm và trong phiên (Ch08 01)"),
                   ("Backtest theo phiên (mức 2)", "`SESSION_FILTER`: us_cash, overnight, london", "Mỗi bộ lọc một hash")],
        repo_evidence=evid,
    ))

index("US500", "US500 / SPX500 / SP500", "exness_usidx_sess",
      "Beta cổ phiếu Mỹ; rộng nhất, rẻ nhất trong nhóm chỉ số",
      [],
      ["Lợi nhuận qua đêm và trong phiên tách riêng (Ch08 01)", "Biến động thực hiện và regime (Ch09)", "Cờ phiên, cờ mép phiên, cờ FOMC và opex",
       "Feature từ VIX nếu tài khoản có (VIX là CFD riêng), cấu trúc kỳ hạn ES nếu lấy được (Ch02 06)"],
      "Chỉ số là nơi tách qua đêm và trong phiên quan trọng nhất: phần lớn lợi nhuận dài hạn của S&P đến từ qua đêm, còn phần trong phiên là nơi mean reversion hoạt động. Hai nhãn, hai spec.",
      "Tương quan với USTEC khoảng 0.9; giữ cả hai vì beta và biến động khác nhau, nhưng danh mục cấp trên sẽ gộp rủi ro.",
      ["ES trong `case_studies/cme_futures`", "SPY trong `case_studies/etfs`", "`26_mlops_governance/04_circuit_breakers.py` chạy trên SPY 2020 H1 — mẫu breaker"])

index("USTEC", "USTEC / NAS100 / NDX", "exness_usidx_sess",
      "Beta công nghệ; biến động cao hơn US500, nhạy lãi suất và lợi nhuận Big Tech",
      [("Lợi nhuận Big Tech", "Sau 21:00 UTC, mùa báo cáo", "Gap lớn qua đêm"),
       ("Lợi suất Mỹ 10 năm", "Liên tục", "Nhạy hơn US500")],
      ["Như US500", "Chênh lệch USTEC trừ US500 làm feature cấu trúc (Ch08 03)", "Beta với BTCUSD trong giờ Mỹ"],
      "Case study `nasdaq100_microstructure` là bằng chứng gần nhất về tín hiệu intraday trên rổ này: tín hiệu thô thua chi phí, chỉ hồi phục sau khi sàng lọc và chọn mô hình có kỷ luật. Kỳ vọng tương tự cho CFD.",
      "Tương quan cao với US500.",
      ["NQ trong `case_studies/cme_futures`", "QQQ trong `case_studies/etfs`", "`case_studies/nasdaq100_microstructure` — 15 phút, `edge_block`, `decision.cadence_by_label`"])

# ---------------------------------------------------------------- render
def render(a):
    s = a["sessions_active"]
    fm = ["---", f"symbol: {a['symbol']}", f"asset_class: {a['cls']}", f"bot: {a['bot']}",
          f"template_case_study: \"{a['template']}\"", "timezone: UTC", "sessions:"]
    for k in s:
        d = SESSIONS[k]
        fm += [f"  {k}:", f"    winter: \"{d['winter']}\"", f"    summer: \"{d['summer']}\""]
    fm += ["decision_times:"]
    for name, when, _ in a["decisions"]:
        fm += [f"  - name: \"{name}\"", f"    when: \"{when}\""]
    fm += ["avoid_windows:"]
    for name, when, _ in a["avoid"]:
        fm += [f"  - name: \"{name}\"", f"    when: \"{when}\""]
    fm += ["news:"]
    for name, when, _ in a["news"]:
        fm += [f"  - name: \"{name}\"", f"    when: \"{when}\""]
    fm += ["verify_on_mt5: [symbol_info.trade_contract_size, volume_min, volume_step, swap_long, swap_short, swap_rollover3days, symbol_info_session_trade]", "---", ""]

    body = [f"# {a['symbol']} — hồ sơ phiên và giờ giao dịch", "",
            f"**Lớp tài sản**: {a['cls']} · **Bot**: `{a['bot']}` · **Khuôn case study**: {a['template']}", "",
            f"**Vai trò trong danh mục**: {a['role']}", "",
            f"**Hợp đồng**: {a['contract']}", "",
            "## 1. Giờ giao dịch", "", a["hours"], "",
            "## 2. Các phiên (UTC)", "",
            "| Phiên | Mùa đông | Mùa hè | Ghi chú |", "|---|---|---|---|"]
    for k in s:
        d = SESSIONS[k]
        body.append(f"| `{k}` | {d['winter']} | {d['summer']} | {d['note']} |")
    body += ["", "Mùa đông là khoảng đầu tháng 11 đến giữa tháng 3 (giờ chuẩn Mỹ và Anh). Mùa hè sớm hơn một giờ. Tokyo không đổi giờ. "
             "`bots/_shared/sessions.py` tính từ múi giờ thật, không hard-code bảng này.", "",
             "## 3. Lịch tin và sự kiện định kỳ", "",
             "| Sự kiện | Thời điểm | Ảnh hưởng |", "|---|---|---|"]
    for name, when, eff in a["news"]:
        body.append(f"| {name} | {when} | {eff} |")
    body += ["", "## 4. Rollover và swap", "", ROLLOVER, "",
             "## 5. Giờ nên tránh vào lệnh thị trường", "",
             "| Cửa sổ | Thời gian | Lý do |", "|---|---|---|"]
    for name, when, why in a["avoid"]:
        body.append(f"| {name} | {when} | {why} |")
    body += ["", "Tránh ở đây nghĩa là không đặt lệnh thị trường mới, và spread trong cửa sổ đó lấy phân vị 90 khi tính chi phí. "
             "Feature vẫn được tính trên mọi nến. Một cửa sổ chỉ được đưa trở lại lịch quyết định khi backtest theo phiên chứng minh nó sống sót qua chi phí.", "",
             "## 6. Mốc quyết định và khớp lệnh của bot", "",
             "| Mốc | Thời điểm | Ghi chú |", "|---|---|---|"]
    for name, when, note in a["decisions"]:
        body.append(f"| {name} | {when} | {note} |")
    body += ["", "Quyết định luôn tại giá đóng nến, khớp ở nến kế tiếp. Mọi cửa sổ feature của bot theo phiên bị chặn tại biên phiên.", "",
             "## 7. Feature từ repo", ""]
    body += [f"- {f}" for f in a["features"]]
    body += ["", "## 8. Bằng chứng và mẫu trong repo", ""]
    body += [f"- {e}" for e in a["repo_evidence"]]
    body += ["", "## 9. Tương quan và vị trí trong danh mục", "", a["corr"], "",
             "## 10. Ghi chú riêng", "", a["notes"], "",
             "## 11. Xác minh trên tài khoản Exness trước khi dùng", "",
             "```python", "import MetaTrader5 as mt5", "mt5.initialize()", f"s = \"{a['symbol']}\"  # đổi sang tên đúng trong Market Watch",
             "info = mt5.symbol_info(s)",
             "print(info.trade_contract_size, info.volume_min, info.volume_step, info.volume_max,",
             "      info.swap_long, info.swap_short, info.swap_rollover3days, info.spread, info.digits)",
             "for day in range(7):",
             "    i = 0",
             "    while (sess := mt5.symbol_info_session_trade(s, day, i)) is not None:",
             "        print(day, sess); i += 1",
             "rates = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_D1, 0, 100000)",
             "print('D1 bars:', len(rates), 'from', rates[0]['time'] if len(rates) else None)",
             "```", "",
             "Kết quả đo được ghi đè lên mọi con số ước lượng trong file này. Cập nhật file khi Exness đổi giờ hoặc điều kiện hợp đồng.", ""]
    return "\n".join(fm + body)

for a in ASSETS:
    (OUT / f"{a['symbol']}.md").write_text(render(a), encoding="utf-8")
    print("wrote", a["symbol"])

# index
rows = ["| Mã | Lớp | Bot | Phiên chính | Mốc quyết định | File |", "|---|---|---|---|---|---|"]
for a in ASSETS:
    rows.append(f"| {a['symbol']} | {a['cls']} | `{a['bot']}` | {', '.join(a['sessions_active'])} | {a['decisions'][0][1]} | [{a['symbol']}.md]({a['symbol']}.md) |")
readme = """# Hồ sơ tài sản Exness cho các bot ML4T

Mười tài sản, mỗi tài sản một file. Phần YAML đầu mỗi file là định nghĩa máy đọc được cho `bots/_shared/sessions.py`
(phiên theo UTC mùa đông và mùa hè, mốc quyết định, cửa sổ tránh, lịch tin). Phần Markdown là mô tả đầy đủ cho người đọc.

""" + "\n".join(rows) + """

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
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
print("wrote README")
