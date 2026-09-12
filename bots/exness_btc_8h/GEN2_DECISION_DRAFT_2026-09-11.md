# Gói quyết định — `exness_btc_8h`, đóng thế hệ 1 (mentor draft, read-only, 2026-09-11)

Trạng thái: DRAFT do `ml4t-mentor` soạn, orchestrator phiên `exness-btc-8h-8d` ghi lại (`reviews.jsonl`
2026-09-10T22:05:46Z). **Không phải bằng chứng, không phải quyết định.** Chỉ những gì người dùng duyệt nguyên văn mới được
chuyển vào `BOT.md`.

Vị trí trên lộ trình: phase 4 (Mô hình), cổng đã qua như một screen (`roadmap.md:146-147`); phase 5 không mở vì
`kill_criteria.r1_early_close` (`setup.yaml:891`). Guard áp dụng: Multiple testing (Ch07 07, `mentor-protocol.md:67`) và
Evidence boundary (holdout 2025-09-01 → 2026-08-31 chưa đọc, `setup.yaml:775-776`; `as_of <= 2025-08-31`).

## 1. Kết quả thế hệ 1 bằng ba con số

| # | Con số | Nguồn | Nói gì | KHÔNG nói gì |
|---|---|---|---|---|
| 1 | **0/356** prediction set qua BH p<0.05; BH-adjusted p nhỏ nhất **0.994581**; HAC p thô nhỏ nhất **0.0501**; **0/356** có \|hac_t\| ≥ 1.96 chưa hiệu chỉnh | `r1_early_close_verdict.json`; `phase4_ic_diagnostics.csv` cột `hac_p`, `hac_t`; `BOT.md:239` | Không mô hình nào trong dân số đã khai (28 linear + 15 GBM × 10 checkpoint × 2 nhãn) có pooled OOF IC phân biệt được với 0 ở HAC bandwidth 9 trên ~4,377 slot validation. BH và sign 4/4 không có ứng viên để tác động (M2 chưa từng là yếu tố quyết định) | Không nói "không có edge", không bác bỏ B1 — kill line của B1 là DSR ở K5 (phase 5) và breakeven (phase 6), chưa rút (`BOT.md:249`). Chuẩn đóng: tiền lệ `xau_fx_mt5` D1 "INCONCLUSIVE by the pre-registered rule; line closed, no re-run" (`bots/xau_fx_mt5/BOT.md:339`) |
| 2 | **+0.0303 = 26.5 % của IC\* 0.1144** (`default_mse`, `fwd_ret_8h`, checkpoint 150; fold IC [+0.045, +0.033, +0.047, −0.004], sign 3/4; hac_t 1.9599) | verdict `best_ic_mean`, `leading_ic_over_ic_star`; IC\* từ `setup.yaml:896` | Ngay cả set dẫn đầu cũng thiếu ~4× so với mức tương quan trả nổi spread + một đêm swap. Cơ chế (ii) của B1 đo được ngắn xa cost-paying IC trước khi cần backtest | Không nói mô hình "vô dụng" ở mức chiến lược (IC\* là ngưỡng trên mọi slot, backtest chỉ trả phí ở lượt giao dịch, `BOT.md:295`). Không được đọc lại 356 dòng để chọn `default_mse` — đó là selection (`BOT.md:253`) |
| 3 | **DSR-K = 0; FDR-n = 73 rồi 356; K5 = 2,136 khai mà chưa rút** | `BOT.md:244-247`; registry 86 training runs / 356 prediction sets / 0 backtests (`BOT.md:233`) | Chưa có Sharpe nào được tính hay chọn, nên chưa có DSR | Không "miễn phí về sau": theo accumulation rule (`BOT.md:247`, `PHASE567_DECLARATION.md:226`), 2,136 nhập K ngay khi một yếu tố thiết kế thế hệ 2 được chọn *vì* một con số đọc từ bảng IC này |

Ghi chú mô tả, không phải bằng chứng (`BOT.md:249`): 56/56 set linear có IC âm trên cả hai nhãn; 10/28 set linear
`fwd_ret_24h` có 4/4 fold âm (hac_t −1.19…−1.48); `lasso_f0.85`/`enet_f0.85` naive t −3.07 nhưng HAC t −1.94 (p 0.053) —
overlap 2 slot làm `hac_se` nhãn 24h (median 0.0241) lớn hơn nhãn 8h (0.0154).

## 2. M2: hai cách hoà giải

Lệch: văn bản B2 đã duyệt nói **chỉ BH** (`BOT.md:109-111`); code cài **BH VÀ sign 4/4** (`12_model_analysis.py:375-377`
`clears_r1 = bh_significant & (sign_consistency >= r1_min_sign_consistency)`, đọc `setup.yaml:904
r1_required_sign_consistency: 1.0`; markdown `:341-347`; thông điệp `:402-403`, `:416-417`; lý do ở `BOT.md:208`).

- **Phương án A — sửa văn bản B2.** Duyệt lại nguyên văn `BOT.md:109-111` thành "...có HAC p < 0.05 sau BH **và** sign
  consistency 4/4 fold theo chiều của chính cấu hình (`_model_reading.py::fold_ic_summary`) thì đóng thế hệ..."; câu tóm tắt
  `BOT.md:101-102` sửa cùng. Code và `setup.yaml` giữ nguyên.
- **Phương án B — thu hẹp code về đúng B2.** `12_model_analysis.py:375-377` → `clears_r1 = bh_significant`; `setup.yaml:904`
  bỏ khoá (hoặc `0.0` kèm comment "reported, not gating"); sửa markdown `:341-347` và hai chuỗi `:402-403`, `:416-417`;
  sign consistency vẫn được in (`:294`, `:309-310`). Test pin khoá này trong `test_phase4_declarations.py` phải đi theo.
- Hôm nay hai phương án cho kết quả giống hệt: `n_bh_significant = 0` nên `clears_r1 = 0` dù có hay không điều kiện sign.
- **Đề xuất: B.** Văn bản đã duyệt là pre-registration; siết quy tắc sau khi đã thấy kết quả — dù chặt hơn — vẫn là
  "threshold change made after seeing generation-1 results" (`bots/xau_fx_mt5/BOT.md:885`). Code phải theo hợp đồng. Nếu
  muốn sign 4/4 làm cổng cho thế hệ 2, ghi vào B2 **mới** trước fit đầu tiên của thế hệ 2.

## 3. MDE trước khi mở thế hệ 2

Cách tính (Ch07 06 `06_ic_inference`): MDE = (z_{α/2} + z_{power}) × SE_HAC, hai đuôi, power 0.8 (1.960 + 0.842). Dưới BH
với một hiệu ứng thật đơn lẻ trong m set, ngưỡng ở hạng 1 là α/m (cận bảo thủ); nếu họ mới nâng đồng thời k set, ngưỡng
nới về α·k/m. **SE đo được** (356 dòng, `hac_se`): median **0.01937** (min 0.0147, max 0.0253; ≈1.28× SE naive
1/√4377 = 0.0151); `fwd_ret_8h` **0.01535**, `fwd_ret_24h` 0.02413. MDE dùng `hac_se` (thang nhiễu), không dùng
`ic_mean` nào → không chọn cấu hình nào → không kích accumulation rule; phải ghi rõ trong pre-registration.

| Dân số m | z ngưỡng | MDE (SE pooled 0.01937) | MDE (SE 8h 0.01535) | Power bắt IC 0.06 (pooled / 8h) | Power bắt IC\* 0.1144 |
|---|---|---|---|---|---|
| (a) 1 set, α 0.05 hai đuôi | 1.960 | **0.0543** | **0.0430** | 0.87 / 0.97 | 1.00 |
| (b) 36 = 3 config × 2 nhãn × 6 checkpoint (minh hoạ; menu a-priori "ols + default_mse + default_mae × 10 ckpt" = 42 cho cùng số) | 3.197 | **0.0782** | **0.0620** | 0.46 / 0.76 | 1.00 |
| (b) 356 = dân số thế hệ 1 | 3.807 | **0.0900** | **0.0714** | 0.24 / 0.54 | 0.98 |

Đọc bảng:
- So với IC\* 0.1144: thiết kế thế hệ 1 **đủ lực** (power 0.98 ở m = 356). Cổng không thiếu lực; nó thiếu tín hiệu.
- So với 0.0303 đã đạt: power 0.35 (một test), 0.05 (m 36), 0.01 (m 356) — không thể qua BH, và theo B1 không trả nổi chi phí.
- Một họ "thêm +0.03 IC" (→ ~0.06): BH ở 356 power 0.24 / 0.54; m = 36: 0.46 / 0.76 — vẫn dưới 0.8. Họ mới phải được
  **dự đoán trước** nâng pooled IC `fwd_ret_8h` lên **≥ 0.062 (m 36) hoặc ≥ 0.071 (m 356)**.
- IC 0.05 tuyệt đối: với 4 fold hiện tại không m nào đạt power 0.8 (m = 1 cần ≈ 5,155 slot pooled; m = 36 ≈ 6,700–10,700;
  m = 356 ≈ 8,900–14,200, vượt cửa sổ phát triển 8,261 slot, `setup.yaml:742`). Đổi hình học fold là thay đổi thiết kế.

## 4. Ba lối đi cho thế hệ 2

**(i) Họ `on-chain and flow`** (`setup.yaml:729-738`, khai PLANNED 2026-09-08 trước mọi IC; B1 nêu đích danh, `BOT.md:49-51`).
- Dữ liệu cần: exchange netflow, miner reserves, US spot-ETF creations (`setup.yaml:732`; `bots/assets/BTCUSD.md:77-81`).
  Repo không có nguồn nào cho ba biến này: `data/crypto/onchain/` chỉ tải DefiLlama TVL và CoinGecko (key-free, CoinGecko
  giới hạn 365 ngày; `data/crypto/onchain/README.md:4-9`, `data/crypto/loader.py:166-215`); `crypto_perps_funding` chỉ nạp
  Binance perp klines + premium index + funding từ `data.binance.vision` (`funding_data.py:19-20`) — chưa tải trên máy này.
  Data lake: không dataset on-chain; nguồn crypto đều bắt đầu sau 2025-08-06. Cần **vendor ngoài repo** (Glassnode /
  CryptoQuant / Coin Metrics cho netflow và miner; ETF flow chỉ tồn tại từ 2024-01-11), **key mới trong `.env`**,
  **backfill 2018-2025** và **vintage** (Ch02 14, `roadmap.md:80`; `setup.yaml:733` cấm bảng gõ tay). Funding/premium
  Binance là nguồn free có sẵn cho cơ chế "dòng quanh mốc funding" của B1, nhưng không nằm trong họ đã khai → họ mới.
- Pre-registration phải ghi: B1 mới nguyên văn; vendor, key, ngày backfill, sha256, cách xử lý vintage; IC dự đoán trước
  ≥ 0.062 (m 36) / ≥ 0.071 (m 356) trên `fwd_ret_8h` với lý do cơ chế; m chọn bằng quy tắc a-priori (ví dụ "ols + hai
  preset GBM mặc định"), không phải "config dẫn đầu thế hệ 1"; `dollar` là "scored AND spent"; không đổi preset /
  checkpoint / horizon / HAC / IC\* / fold.
- Chi phí: FDR-n +m; DSR-K 0 tới `13_backtest`; fit ~8-10 phút; phase 2 (loader, test lookahead, vintage) là phần tốn công;
  tiền vendor. **K:** giữ 0 nếu chỉ dùng họ/nhãn/menu đã khai và prior ngoài; tiêu 2,136 nếu bỏ linear "vì linear âm",
  chọn GBM "vì `default_mse` dẫn đầu", hay chọn nhãn 8h "vì 24h tệ hơn".

**(ii) Lưới quyết định khác, case study riêng.** Ứng viên: `exness_btc_d1` — quyết định tại D1 close 00:00 UTC (bucket
rollover đã đo spread không giãn, `BOT.md:195`), nhãn `fwd_ret_5d`. Theo `bots/README.md:130-131` cadence khác → **bot
mới**: `case_studies/exness_btc_d1/` + `bots/exness_btc_d1/BOT.md`, magic mới. Lý do a-priori hợp lệ duy nhất là số học chi
phí/biến động (IC\* thấp hơn khi σ nhãn dài hơn), không phải mẫu 4/4 âm của bảng IC. Prior fleet yếu: D1 FX majors DSR 0.25
(`exness_fx_d1`), D1 TSMOM `xau_fx_mt5` INCONCLUSIVE. Duyệt: B1/B2/B3 mới. Chi phí: phase 1-4 làm lại (~1 ngày builder),
FDR-n 73 + ~356 mới. **K:** bot mới bắt đầu ở 0, nhưng nếu lưới hoặc nhãn được chọn vì bảng IC của bot này → 2,136 nhập K
của cả hai bot.

**(iii) Dừng và ghi closure.** Record đã có: verdict, ba bộ đếm, "B1 not falsified", điều kiện mở thế hệ 2
(`BOT.md:237-258`). Hàng "retired" sẽ thêm: Phase status row 4 → "CLOSED — generation 2 not planned"; row 9 ghi quy tắc
nghỉ hưu đã kích ("R1 early-close, không có đường dữ liệu cho họ còn lại"); Trials đóng băng FDR-n 73 + 356 / DSR-K 0;
magic 260904 không tái cấp (`bots/README.md:92`); `bots/README.md:120-128` cập nhật. Duyệt: một dòng quyết định.
Chi phí: 0 trial, ~30 phút builder. **K:** 0.

## 5. Đề xuất của mentor

Xếp theo cổng hỏng sớm nhất (`mentor-protocol.md:78-79`): (i) hỏng ở phase 2 hôm nay (không nguồn, không key, không
vintage, không backfill); (ii) hỏng ở phase 1 (chưa có B1, prior fleet yếu); (iii) không hỏng. Vì vậy: **M2 theo phương án
B**, rồi **ghi closure theo (iii) ở dạng "closed, generation 2 conditional"** — chưa "retired" — với (i) là con đường mở
duy nhất đã khai, điều kiện mở là quyết định dữ liệu của người dùng (vendor, key, backfill, vintage) cộng một B1 mới dự
đoán IC ≥ 0.062-0.071 trên `fwd_ret_8h`; không mở (ii) trừ khi người dùng muốn viết B1 mới cho D1 với prior fleet ghi rõ.

Câu hỏi người dùng phải trả lời: **Q1** M2: A hay B? **Q2** Ghi "retired" ngay, hay "closed, generation 2 conditional"?
**Q3** Nếu conditional: có duyệt mua/đăng ký nguồn on-chain (vendor nào, key nào, phủ 2018-2025) không? Nếu không → Q2
thành "retired". **Q4** (chỉ khi muốn (ii)): có muốn mentor soạn B1 mới cho `exness_btc_d1` không?

## 6. Read

- `.claude/skills/ml4t-quant-bot-mentor/references/roadmap.md:146-147`, `:167`, `:255-256`, `:297`.
- `bots/exness_gold_sess/PHASE567_DECLARATION.md:24,29`, `:226`, `:266`.
- `bots/xau_fx_mt5/BOT.md:18`, `:339`, `:885`.
- Tập 2, Ch07 › `06_ic_inference`, `07_multiple_testing` (`knowledge-index.md:198-199`); Ch16 › `12_dsr_validation` (`:453`).
- Tập 1, Ch02 › `14_point_in_time_validation` (`roadmap.md:80`); `data/crypto/onchain/README.md:1-20`.

## Đối chiếu closure record với artefact

Số liệu (`BOT.md:176,231,233,239,249` so với `r1_early_close_verdict.json` và `phase4_ic_diagnostics.csv`): **tất cả khớp**.
Hai điểm wording cần sửa khi ghi closure: (1) `BOT.md:172` và `setup.yaml:733` nói "no free source is wired" — không chính
xác: repo có hai nguồn key-free đã wired (DefiLlama/CoinGecko; Binance funding/premium), chỉ là không nguồn nào cung cấp ba
biến đã khai và không nguồn nào có vintage. (2) `BOT.md:101-102` tóm tắt R1 là "BH alone" trong khi `BOT.md:208` biện minh
"BH AND sign" — chính là M2 (`:112-119`), chưa giải quyết.
