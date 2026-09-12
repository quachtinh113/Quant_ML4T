# DECLARATION — luật Giai đoạn 5, 6, 7 cho `exness_gold_sess`

Ngày 2026-09-08 · Mentor declaration · Repo root `D:/05_Quant/machine-learning-for-trading` (branch `exness-bots`) · Viết **trước** khi `06_linear` fit mô hình đầu tiên, với `case_studies/exness_gold_sess/run_log/` còn rỗng và không tồn tại `registry.db`.

Đây là **declaration**, không phải gợi ý. Builder thực thi từng dòng. Không một quyết định nào dưới đây được sửa sau khi nhìn thấy một con số kết quả; nếu phải sửa, phép sửa là một tuyên bố mới, có ngày, có lý do cơ học, và trial count được tính lại.

Vị trí roadmap: cổng **Giai đoạn 5** (`roadmap.md:167`), **Giai đoạn 6** (`roadmap.md:193`), **Giai đoạn 7** (`roadmap.md:215`).

Guard áp dụng trên toàn văn bản (`mentor-protocol.md:20-33`): **Multiple testing**, **Costs**, **Evidence boundary**, **Capacity**, **Risk actions**, **Parity**.

**Điều kiện tiên quyết**: Giai đoạn 4 chưa chạy. Không dòng nào dưới đây được thực thi trước khi chữ ký nghiệm thu **2a-2i** của tuyên bố thứ hai trong `FOLD_GEOMETRY_DECLARATION.md` được ghi vào `BOT.md` và `tests/test_fold_geometry.py` chuyển **3 xanh**.

---

# GIAI ĐOẠN 5 — Backtest và suy luận

## D5.1 — Sweep chạy trọn vẹn, không chia đợt

**Phán quyết: chạy TOÀN BỘ. Không staging.**

Số học phải được viết ra trước vì `13_backtest` hiện đang in sai:

```
prediction sets: fwd_ret_8h 178 | fwd_ret_24h 178 | dir_tb_8h 63   = 419
engine books   : london, ny                                        = 2
signal specs   : fixed_threshold_0, per_symbol_p80                 = 2
=> backtest ĐƯỢC ĐĂNG KÝ  = 419 x 2 x 2 = 1.676
=> chuỗi pooled (số học)  = 419 x 1 x 2 =   838
=> K khai trước           = 1.676 + 838 = 2.514
```

**Khuyết tật phải sửa trước khi chạy**: `case_studies/exness_gold_sess/13_backtest.py:676-680` in `expected_total` (= **1.676**) kèm câu *"This is the trial count the Deflated Sharpe Ratio divides by"*. Sai **838**. Sửa câu in thành: `1.676 engine backtests; K = 2.514 sau khi cộng 838 chuỗi pooled dựng số học`. Cùng lý do, ước lượng *"~1,9 h engine time"* trong `BOT.md` tính trên 2.514 và **thừa 838 lần chạy không tồn tại**: 1.676 × 2,68 s ≈ **75 phút**, ~2,1 h nếu mỗi lần nạp giá lạnh.

**Thứ tự chạy, cố định và bắt buộc** — đúng thứ tự vòng lặp `13_backtest.py:647-673`, không được đổi:
`for label in sorted(labels)` = `dir_tb_8h → fwd_ret_24h → fwd_ret_8h`; `for book in [london, ny]`; `for spec in signal_specs` theo thứ tự `setup.yaml:380-400`. Thứ tự không phải lựa chọn; **thứ tự cố định là thứ ngăn "chạy cái hứa hẹn trước"**.

**Tham số của lần chạy duy nhất**: `RUN_SWEEP=True`, `LABEL=""`, `SESSION_BOOKS=[]`, `TOP_N_PREDICTIONS=None`, `POPULATION_NAME=""`. Bất kỳ giá trị thu hẹp nào cũng buộc `POPULATION_NAME` (`13_backtest.py:262-270`) và bản thu hẹp đó **không phải bằng chứng canonical**.

**Stopping rule**: không có quy tắc dừng nào dựa trên kết quả. Sweep dừng khi `baseline_population.require_complete()` xanh (`13_backtest.py:794`). Lần chạy chỉ được dừng bởi lỗi hạ tầng, và khi đó population còn dở nên không được công bố. `13_backtest` không có nhánh `except-and-continue` (`:751-769`); giữ nguyên.

**Nếu ai đó vẫn muốn staging**: hình thức trung thực duy nhất là chia theo **nhãn**, theo đúng thứ tự cố định trên, khai trước ngày và **chạy hết cả ba**. Chia đợt rồi quyết định có chạy đợt sau hay không **dựa trên kết quả đợt trước** là điều kiện hóa trên kết quả; khi đó cohort là **hợp** của các đợt, K vẫn 2.514, và DSR phải tính lại trên cohort hợp. Staging không tiết kiệm được một trial nào — nó chỉ tiết kiệm CPU, và CPU ở đây là 75 phút.

**Cổng `dir_tb_8h` đã khai** (`setup.yaml:847-874`) được đánh giá **sau** khi 378 trial của nó đã chạy và chấm; không đạt (2) lẫn (3) thì nhãn đóng, 378 trial ghi là **ĐÃ CHI**, K vẫn 2.514.

## D5.2 — Đăng ký gì cho mỗi backtest, và `19_strategy_analysis` phải báo cáo gì

Mỗi thành viên trong 1.676 phải có, và `_report_phase5.py` phải khẳng định là có:

| Vật thể | Ở đâu | Dùng để làm gì |
|---|---|---|
| hàng `backtest_runs` | `registry.db` | `spec_json` (chứa `signal.session_filter`, `signal.long_short`, `signal.drop_friday`, `risk.position_rules[0].bars`), `stage == "signal"`, `git_commit`, `elapsed_s` |
| hàng `backtest_metrics` | `registry.db` | `sharpe`, `sharpe_ci95_lo/hi`, `psr_pvalue`, `max_drawdown`, `cagr`, `volatility`, `num_trades`, `n_periods`, `total_commission`, `total_slippage` |
| `daily_returns.parquet` | `run_log/backtest/<hash>/` | **chuỗi duy nhất mọi thống kê D5.3 đọc** |
| `trades.parquet`, `fills.parquet` | idem (`registration.py:1377-1383`) | turnover thật, P&L theo symbol (D6.2) |
| `portfolio_state.parquet` | idem | `net_exposure / gross_exposure` — cổng phơi nhiễm |
| `weights.parquet` | idem | **chỉ để kiểm tra**, không dùng tính turnover |

> **TU CHÍNH D5.2, 2026-09-09 (builder, theo phán quyết mentor cùng ngày; câu cũ ở hàng `backtest_runs` phía trên
> giữ nguyên, không xoá).** Cột `stage` của `backtest_runs` **không** phải `"signal"` trên bot này và không thể là:
> registry không nhận nhãn stage từ caller mà **suy từ nội dung spec** (`case_studies/utils/registry/store.py:447-449`:
> spec có khối `risk` mà `risk.name != "baseline"` → `risk_overlay`). Mọi spec baseline của bot này mang hold là position
> rule `time_exit` trong khối `risk` (`PRICE_GRID_DECLARATION.md` §2), nên cả 1.674 hàng được đăng ký `stage = 'risk_overlay'`.
> Đó là **nhãn tầng đăng ký suy từ position rule**, không phải khẳng định một overlay rủi ro đã được quét. Assertion
> `stage == "signal"` ở `13_backtest.py` (chép từ `exness_fx_d1`, nơi spec không có khối `risk`) bị thay bằng assertion
> **nội dung** — với mọi hàng: `spec_json.strategy.risk == {"position_rules": [{"type": "time_exit", "bars": HOLD_BARS[label, book]}]}`
> đúng bằng hold đã derive của (label, book) hàng đó; `strategy.signal.session_filter` ∈ {london, ny} và bằng book của job;
> `stage` lấy đúng MỘT giá trị trên toàn bộ hàng và bằng giá trị `_infer_stage` suy từ chính nội dung ấy (`risk_overlay`).
> Vị từ chỉ đọc `spec_json`, không đọc return hay metric, nên không thể nhận hay loại một thành viên nào. Lối thoát của thư
> viện, `risk: {name: baseline, …}`, dịch hash mọi spec nên thuộc **thế hệ 2**, phải khai trước backtest đầu tiên của thế hệ ấy.
>
> **Xác minh 2026-09-10 (builder).** `13_backtest` chạy lại bằng papermill với đúng tham số D5.1 và
> `SUPERSEDES_SESSION_BOOK_BASELINES=cf0cb9ef5f25` (`FORCE_REBACKTEST` mặc định `False`): 2026-09-09 19:45:39 → 20:17:11 UTC
> (02:45 → 03:17 VN 2026-09-10), exit 0, `0 computed, 1674 served from the registry, 1674 in the population`, assertion
> nội dung xanh trên cả 1.674 hàng với đúng một nhãn stage `risk_overlay` bằng `_infer_stage`, `require_complete()` xanh,
> `Official exness_gold_sess:session-book-baselines population: 2f37b3b51a42`; registry trước và sau giống hệt
> (`backtest_runs` 1.674 / `backtest_metrics` 1.674 / `official_populations` 24 / `official_population_members` 6.851 /
> `prediction_sets` 801, không thêm hàng population). Lần thử 2026-09-09 15:38 UTC bị ngắt ở cell 18/23 trước cell chạy
> engine, không ghi registry, không phải một xác minh.

**Turnover: bot này KHÔNG đọc `avg_turnover` của registry.** Lý do cơ học đã khai `PRICE_GRID_DECLARATION.md`: `avg_turnover` tính từ chuỗi weight (`backtest_runner.py:1610-1620`) còn lệnh thoát ở đây là position rule **không đi qua weight**, nên nó báo thiếu khoảng một nửa. `exness_fx_d1` phát hiện cùng lớp lỗi và chuyển sang dựng lại (`_report_phase5.py:426-438`) — **cách đó cũng không dùng được ở đây**, vì `target_weight_turnover` trên scheduled targets vẫn là chuỗi weight.

**Bot này đọc hai số, cả hai từ sổ fill:**

```
turnover_ledger = (total_slippage / slippage_rate) / initial_cash / n_periods
notional_traded = tổng |qty x price| trên fills.parquet
```

`turnover_ledger` là số của cổng chi phí (nó tuyến tính đúng theo cái mà chi phí tính trên). In `avg_turnover` của registry **bên cạnh**, gắn nhãn `UNDER-REPORTS — do not use`.

**`19_strategy_analysis` phải báo cáo** (ngoài phần template): (a) `cohort_metrics` scoped đúng bằng `ADMITTED_PREDICTIONS` (mẫu `fx_pairs/19_strategy_analysis.py:127-141` — gọi `compute_and_register` không scope sẽ deflate trên toàn registry); (b) **chuỗi pooled** theo D5.4; (c) bảng chẩn đoán mức 3 của D5.6.

## D5.3 — Giao thức Deflated Sharpe

**Cohort của bản ghi = cả 2.514 chuỗi**, không phải 1.676. Định danh:

- 1.676 thành viên đăng ký: tên = `backtest_hash`.
- 838 thành viên pooled: tên = `pooled:<london_hash>+<ny_hash>` (khai dạng chuỗi này, không sinh hash mới).
- `member_digest = cohort_member_digest(sorted(names))` — cùng hàm `case_studies/utils/uncertainty.py:1109-1113`. Ghi digest ấy vào `BOT.md` phase-5 row; **mọi DSR được báo cáo phải kèm digest này**.

**Ràng buộc phải khai ngay**: `cohort_metrics` có FK `leader_hash REFERENCES backtest_runs(backtest_hash) NOT NULL`. Chuỗi pooled **không thể** đăng ký vào `cohort_metrics`. Nên có **hai** con số, cả hai được báo cáo, không cái nào thay cái nào:

1. **DSR của bản ghi (cổng)** — tính trong `case_studies/exness_gold_sess/_report_phase5.py`, bot-local, read-only, trên numpy array, K = **2.514**. Đây là số quyết định cổng.
2. **DSR của registry (mô tả)** — `compute_and_register` trong `19`, K = số thành viên đăng ký. In kèm câu: *"cohort hẹp hơn cohort của bản ghi 838 thành viên; không phải cổng"*.

**Quy ước thống kê**, đúng quy ước `16_strategy_simulation/12_dsr_validation.py` mà `exness_fx_d1` đã dùng:

- `PPY = 252` (`setup.yaml:675`). **Không phải 504** — 504 sống ở `decision.slots_per_year`.
- `confidence_level = 0.95`; `is_significant = probability >= 0.95`.
- `variance_trials` = phương sai mẫu (`ddof=1`) của Sharpe **annualized** trên toàn bộ 2.514 thành viên, **tính trên cùng định nghĩa lợi nhuận** với thống kê đang xét (raw cho raw, active cho active).
- In **cả hai** cách tính: công thức notebook (`dsr_notebook`) và `ml4t.diagnostic`. Hai số lệch nhau là bình thường; báo cả hai, không chọn số đẹp.

**Benchmark, phải khai vì lưới giá là H1 chứ không phải D1**: sổ 1/N long hai kim loại dựng bằng cách lấy **bar cuối cùng của mỗi ngày lịch cho mỗi symbol** trên lưới đã đăng ký, tính close-to-close theo ngày, rồi trung bình hai kim loại. **Không** được `pct_change` trực tiếp trên khung H1 — làm thế cho ra chuỗi 24 lần dày và mọi active return sau đó là rác. Active return = `daily_return − benchmark_daily`, join theo `date`, inner.

**`n_trials_effective_mp` / `_er`** được **in như chẩn đoán, không bao giờ dùng làm cổng**. Trên bot này chúng sẽ nhỏ hơn 2.514 rất nhiều (178 checkpoint của cùng một GBM là các chuỗi gần trùng), và hạ K xuống K_eff là hạ chính cái thanh mà leader đã được rút ra từ đó.

**Thứ tự in bắt buộc trong `_report_phase5.py`** — đây là thiết bị chống peek, không phải thẩm mỹ:

1. K = 2.514, `member_digest`, số thành viên mỗi nhãn và sổ;
2. `variance_trials` (raw, active) và **`expected_max_sharpe` dưới giả thuyết không** cho cả hai;
3. **chỉ sau đó** mới in phân phối Sharpe và tên spec dẫn đầu.

**Chữ đạt và không đạt, viết sẵn — builder chép nguyên văn:**

- **ĐẠT**: *"Tại K = 2.514 (`member_digest <digest>`), spec `<hash>` có DSR trên **active return** so với sổ 1/N long hai kim loại = `<p>` >= 0,95. Giả thuyết 'chưa có lợi thế nào được phân giải' bị mẫu này bác bỏ ở mức đã khai. Đây là một kết quả trên split validation; nó không phải một kỳ vọng lợi nhuận và không phải một quyết định triển khai."*
- **KHÔNG ĐẠT**: *"Tại K = 2.514 (`member_digest <digest>`), `<n>` của 2.514 spec có DSR >= 0,95 trên active return. Với n = 0: không spec nào phân giải được một lợi thế trên mẫu này ở số trial đã chi. Giai đoạn 6 KHÔNG mở, `14`-`16` không chạy, holdout không được đọc."*
- **ĐẠT TRÊN RAW, KHÔNG ĐẠT TRÊN ACTIVE** — trường hợp nhiều khả năng nhất trên mẫu vàng 2017-2025: *"Spec `<hash>` có DSR raw `<p_raw>` >= 0,95 và DSR active `<p_act>` < 0,95. **Cổng là active return.** Kết quả đọc là: sổ này không phân biệt được với việc giữ 1/N long hai kim loại trên cùng cửa sổ. Nó KHÔNG được mang sang Giai đoạn 6."* Kèm **bắt buộc** ba bằng chứng cơ chế: (i) tương quan chuỗi lợi nhuận ngày của spec với benchmark; (ii) tỷ lệ slot có `net_exposure > 0` từ `portfolio_state.parquet`; (iii) **cổng phơi nhiễm** kiểu `exness_fx_d1`: tỷ lệ phiên đầu tư có `|net|/gross > 0,5` — một sổ như thế được báo cáo là **directional**, không bao giờ là dollar-neutral.

## D5.4 — Sổ pooled: công thức đúng một dòng, khai trước

`setup.yaml:361-364` khai `pooled_book: {method: sleeve_sum, weights: {london: 0.5, ny: 0.5}, produced_in: 19_strategy_analysis}`. Thực thi chính xác:

```
ghép cặp theo (label, prediction_hash, signal spec name) -> 419 x 2 = 838 cặp, song ánh
pooled_daily_return(d) = 0.5 x r_london(d) + 0.5 x r_ny(d)
join OUTER theo `date`; thiếu một tay áo -> 0.0 (tay áo đó giữ TIỀN MẶT ngày đó)
```

Khẳng định, không phải hy vọng: (a) số cặp đúng 838 và song ánh (một sổ thiếu thành viên thì dừng); (b) in số ngày chỉ có một tay áo — trên nhãn `fwd_ret_24h` con số này phải khớp tập thứ Sáu đã lọc; (c) chuỗi pooled **không** có hàng `backtest_runs`, **không** là thành viên `OfficialPopulation`, **không** là ứng viên carrier.

**Hệ quả phải khai bây giờ**: vì `resolve_solvent_carrier` chỉ đọc backtest đã đăng ký, **sổ pooled không bao giờ có thể là carrier của holdout**. Nếu sổ pooled là thứ được đưa lên demo, thì thứ được chấm ở phase 7 là **hai carrier riêng của hai sổ**, và chuỗi pooled của chúng được báo cáo như tổng số học đã khai. Ghi câu này vào `BOT.md` ngay hôm nay.

## D5.5 — `_report_phase5.py` của bot này

Fork từ `case_studies/exness_fx_d1/_report_phase5.py`, giữ nguyên khung, đổi đúng những chỗ sau:

| Sửa | Vì |
|---|---|
| `LABELS = ["fwd_ret_8h", "fwd_ret_24h", "dir_tb_8h"]` | nhãn của bot này |
| benchmark: gấp H1 sang D1 trước khi `pct_change` | lưới giá là H1 (D5.3) |
| `_spec_fields`: `spec_key` = `f"{method}/{session_filter}/hold{risk.position_rules[0].bars}"` | `top_k` không tồn tại; sổ và hold **phải** hiện trong khoá nhóm |
| bỏ `turnover_from_weights`, giữ `turnover_from_ledger`, thêm `notional_traded` từ `fills` | D5.2 |
| thêm 838 chuỗi pooled vào `member_stats_df` trước khi tính `var_raw` / `var_active` | K của bản ghi |
| `ONE_LEGGED_THRESHOLD` giữ 0.5, đổi tên báo cáo thành **cổng phơi nhiễm** | mẫu vàng có xu hướng tăng |

Chạy: `ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_gold_sess uv run python case_studies/exness_gold_sess/_report_phase5.py`. Nó **không ghi gì** vào registry.

## D5.6 — Bảng chẩn đoán theo phiên, ba mức — CHỈ ĐỌC

In cho **mọi** spec vào top-5 theo active Sharpe và cho carrier:

- **Mức 1 — sổ**: `london` / `ny` / `pooled`. N slot, N ngày, Sharpe raw, Sharpe active, max DD, `turnover_ledger`, số trade, tỷ lệ phiên `|net|/gross > 0,5`.
- **Mức 2 — sổ x kim loại**: P&L gộp và ròng theo symbol từ `trades.parquet`, số trade, notional giao dịch, hit rate theo slot.
- **Mức 3 — sổ x kim loại x năm validation** (2021-22 / 2022-23 / 2023-24 / 2024-25, đúng bốn fold): hit rate, mean net return mỗi slot, Sharpe, và **pooled panel IC của score carrier so với `fwd_ret_8h`** kèm khoảng HAC.

**Cộng thêm, bắt buộc, do `NY_EXIT_DECLARATION.md` yêu cầu**: trên riêng hàng New York — `corr(fwd_ret_7h_probe, fwd_ret_8h)` và IC của cùng bộ prediction trên probe 7 giờ. Probe dựng từ panel đã niêm phong, không phải nhãn mới, không dịch digest nào.

**Luật cứng, ghi ngay cạnh bảng trong notebook**:

> Bảng này là **chẩn đoán, không phải mặt phẳng để chọn**. Không mô hình, không spec, không sổ, không kim loại nào được chọn hay loại vì một ô trong bảng này. Nó được tính trong `_report_phase5.py`, ghi 0 hàng vào registry, và không `CandidateSet` / `OfficialPopulation` nào được dựng từ nó.

Cưỡng chế cơ học: `_report_phase5.py` mở `registry.db` bằng `sqlite3.connect(...)` ở chế độ đọc và **không import `CandidateSet`, `OfficialPopulation`, `run_backtests`**.

---

# GIAI ĐOẠN 6 — Danh mục, chi phí, rủi ro

**Cổng vào**: chỉ mở khi D5.3 cho ít nhất một spec ĐẠT trên **active return**. Không đạt thì `14`, `15`, `16` không chạy.

## D6.1 — Allocator: đóng chiều này lại

**Phán quyết: chiều allocation ĐÓNG cho thế hệ này. `14_portfolio_management` KHÔNG chạy.** Ba lý do cơ học, mỗi lý do đủ để đóng:

1. **Mọi allocator trong `case_studies/utils/allocation.py` đều xếp hạng cross-section.** Cả ba đi qua `_select_top_bottom` (`allocation.py:22-46`), và ở đó `effective_k = min(top_k, n_assets // 2)`. Với `n_assets = 2` và `long_short=True`, **effective_k = 1**: mỗi slot **bắt buộc** long một kim loại và short kim loại kia, bất kể hai điểm số cùng dấu gì. Sổ trở thành một giao dịch chênh lệch vàng/bạc thường trực — nó **ghi đè** `mapping.class: per_asset_signal_timing` và ghi đè cả ngưỡng của `fixed_threshold_0`.
2. **Stage template sẽ ném, rồi nếu sửa ẩu sẽ ném mất luật của bot.** `14_portfolio_management.py:192-196` `_baseline_top_k` **raise** trên mọi signal không phải `equal_weight_top_k`. Và `:386, :434` dựng lại signal thành `{"method": "equal_weight_top_k", "top_k": ...}`, **vứt bỏ** `session_filter`, `long_short`, `drop_friday` **và khối `risk` mang hold**.
3. **`max_sleeve = n_assets // 2 = 1`** (`:109`) và `:352-356` raise nếu `top_k > 1`.

**Cái gì sẽ có nghĩa trên hai tài sản**: một bộ tỷ lệ theo biến động **giữ nguyên dấu tín hiệu của từng tên** (per-asset vol scaling), tức không xếp hạng. Không tồn tại trong `allocation.py`. Xây nó là **sửa thư viện dùng chung cộng một họ trial mới**.

**Kế toán trial, ghi trước**: công thức K đã khai **không có thừa số allocator**. Nếu `14` từng chạy, nó **cộng thêm** trial ngoài 2.514: `n_allocation_backtests = |labels| x top_n_predictions.allocation(=10) x |allocators|`, và DSR phải tính lại. Đó là lý do thứ tư để đóng.

## D6.2 — Cổng chi phí, đầy đủ

**(a) Đơn vị — chỗ dễ sai nhất và nó sai theo hướng dễ dãi.** `16_costs` chia mỗi điểm lưới thành `commission_bps = total/2` và `slippage_bps = total/2` (`fx_pairs/16_costs.py:395-399`), tức **`cost_grid_bps` là bps MỖI LƯỢT KHỚP**, còn `costs.round_trip_p90_bps` là **khứ hồi**. So thẳng hai số là làm cổng dễ đi **2 lần**. Luật:

```
breakeven_round_trip_bps = 2 x breakeven_per_crossing_bps
CỔNG: breakeven_round_trip_bps > 1,5 x round_trip_p90_bps
```

| | round trip p90 | ngưỡng khứ hồi (x1,5) | ngưỡng mỗi lượt |
|---|---|---|---|
| XAUUSD | 1,2 bps | **> 1,8 bps** | > 0,90 bps |
| XAGUSD | 9,4 bps | **> 14,1 bps** | > 7,05 bps |
| sổ (bị tính giá bạc) | 9,4 bps | **> 14,1 bps** | > 7,05 bps |

Lưới `[0,1,2,3,5,7,10,15,20,30,50]` có 7, 10, 15 nên nó **bao** được 7,05 — lưới không cần đổi và **không được đổi**.

**(b) Breakeven tính thế nào.** Trên **một** spec đã khai trước (carrier; `top_n_predictions.cost_sensitivity = 1`), **nội suy tuyến tính trên chính đường Sharpe(cost) của spec đó**, giữa điểm lưới cuối còn dương và điểm lưới đầu <= 0. **Không** dùng:

- `strategy_analysis.py:1520-1537` (`plot_cost_decay`) như nó đứng: nó nội suy trên **envelope max-over-config**, tức một phép chọn cực đại nữa chồng lên phép chọn đã có. Được phép vẽ, không được lấy số.
- `strategy_analysis.py:2213-2221`: quy tắc bậc thang lấy **điểm lưới đầu tiên có Sharpe <= 0**, tức **cao hơn** breakeven thật tới một bước lưới — sai theo hướng dễ dãi. In kèm, gắn nhãn `step rule, over-states by up to one grid step`.

**(c) Chạy per-symbol mà không thêm một trial nào.** `apply_universe_filter` chỉ nhận `liquid` / `cost_feasible` / `full` — **không có đường chạy một kim loại**. Đừng thêm giá trị mới vào thư viện dùng chung. Dùng dạng đóng, vì chi phí tuyến tính theo notional:

```
mean_net(c) = mean_gross − c x (notional_traded_per_period / equity)
Sharpe = 0  <=>  mean_net = 0   (std > 0)
=> breakeven_per_crossing[symbol] = P&L gộp của symbol / notional đã giao dịch của symbol
```

Cả hai vế lấy từ `trades.parquet` / `fills.parquet` của **chính carrier đã đăng ký**: không lần chạy nào, không hash nào, không trial nào. **Falsifier bắt buộc**: công thức này áp ở mức **sổ** phải khớp đường 11 điểm mà `16_costs` chạy, trong phạm vi một bước lưới. Không khớp thì công thức sai, dừng, báo cáo.

**(d) Luật loại XAGUSD, viết đủ.**

> Nếu `breakeven_round_trip_bps[XAGUSD] <= 14,1` **và** `breakeven_round_trip_bps[XAUUSD] > 1,8`, thì XAGUSD **rời universe giao dịch**. Việc đó là **một thế hệ mới và một thí nghiệm mới**: `universe.symbols` đổi nên mọi label digest, feature digest và backtest hash đổi. **2.514 trial cũ vẫn được đếm**, K của thế hệ mới cộng dồn lên chúng, và Trials table ghi một hàng có ngày nói rõ điều đó.
>
> **Họ feature tỷ số vàng/bạc KHÔNG mất.** `setup.yaml:461` đã khai `features.reference_symbols: [XAGUSD]` đúng cho tình huống này: cả hai chân của tỷ số phải **quan sát được**, chỉ một chân phải **giao dịch được**. `gsr`, `gsr_z_252d`, `gsr_beta_63d` vẫn được dựng, và `_features.gold_silver_ratio` **raise** nếu chân bị mất khỏi cả hai khoá — nó không trả về ba cột null.
>
> Nếu **cả hai** trượt cổng: bot dừng ở Giai đoạn 6. Không có "nới safety_margin". `safety_margin: 1.5` đã khai trước sweep; sửa nó sau khi nhìn breakeven là điều kiện hóa trên kết quả.

**(e) Bối cảnh phải in cạnh cổng, không phải để nới nó**: IC\* của bạc trên nhãn 8h là **0,0397**, còn sai số chuẩn của IC trên mẫu này khoảng **0,0216**, nên IC tối thiểu phát hiện được khoảng **0,054 > 0,0397**. Nghĩa là **trên mẫu này, một IC vừa đủ trả spread của bạc không phân biệt được với 0.** Đó không phải lý do bỏ cổng breakeven — đó là lý do cổng breakeven là phép thử duy nhất còn nói được điều gì về chân bạc.

## D6.3 — Swap: cái gì chặn `16_costs`

**Cổng, chép nguyên từ tiền lệ**: port `case_studies/exness_fx_d1/16_costs.py:159-178` vào `case_studies/exness_gold_sess/16_costs.py` **nguyên văn** — stage **từ chối chạy** khi `costs.swap.measured_on_account` không phải `real`/`live`, và từ chối khi có swap khác 0 mà mô hình phần trăm không diễn đạt được.

**Phải xảy ra trước khi `16_costs` được phép chạy**, theo thứ tự:

1. Đọc read-only trên login Pro **thật**: `symbol_info` cho XAUUSD/XAGUSD, cộng một cửa sổ tick 30 ngày qua `bots/_shared/costs_mt5.py::measure_spreads`.
2. Ghi vào `setup.yaml::costs.swap` kèm `measured_on_account: real` và ngày; ghi bảng spread mới **nối thêm**, không ghi đè.
3. Nếu swap khác 0: chuyển sang points qua `holding_cost_points` và đưa vào lưới **một cách tường minh**; mô hình phần trăm không tự làm được.

**Phép đo gỡ chặn một phần, phải chạy trước bước 1 và có khả năng cao trả về 0**:

> Đếm số vị thế mà **cửa sổ giữ chứa mốc 00:00 UTC của server**, theo (label, book), từ `trades.parquet` của một lần `register=False`. London 09:00 đến 17:00 UTC và New York 13:00/14:00 đến 20:00/21:00 UTC đều nằm gọn trong một ngày server (server là UTC+0), và hold của New York đã là **7 bar** nên không còn Friday carry. Nếu số đếm là **0** cho `fwd_ret_8h` và `dir_tb_8h`, thì **swap của hai nhãn ấy bằng 0 theo cấu tạo, độc lập với tài khoản**.

Nếu phép đo trả 0: khai `costs.swap.not_applicable_labels: [fwd_ret_8h, dir_tb_8h]` **kèm số đếm đo được và ngày**, nới cổng của `16_costs` **theo nhãn** (chỉ `fwd_ret_24h` bị chặn), và ghi vào `BOT.md`. Nếu số đếm khác 0: không nới gì cả.

**Nếu không đọc được tài khoản thật trong phiên này**: builder **không chạy `16_costs`**, không sửa cổng, không đặt `measured_on_account: real`. Thay vào đó, cả ba: (1) ghi một hàng Open questions; (2) chạy phép đo mốc 00:00 UTC — nó không cần tài khoản thật; (3) nếu (2) trả 0 cho hai nhãn 8 giờ, chạy `16_costs` **chỉ** cho hai nhãn ấy dưới khai báo `not_applicable_labels`, và ghi rõ nhánh `fwd_ret_24h` còn treo.

## D6.4 — Overlay rủi ro: đo tỷ lệ kích hoạt trước, rồi mới quét

Sáu arm đã khai: `stop_loss_3pct`, `stop_loss_5pct`, `trailing_1pct`, `trailing_2pct`, `trailing_3pct`, `hold_2h`.

**Bối cảnh cơ học**: hold đã bị chặn thời gian (8 bar London, 7 bar NY), và mức dịch chuyển tuyệt đối trung vị của một phiên là **44,3 bps vàng và 85,7 bps bạc**. Một stop 3 % = 300 bps là khoảng **7 lần** mức dịch chuyển trung vị của cả phiên vàng. Một quy tắc không bao giờ kích hoạt vẫn là một trial mà DSR chia cho — đúng lý do `time_exit_10` đã bị xóa.

**Luật, thực thi trước `15_risk_management`:**

> **Pre-check không tốn trial**: với carrier của mỗi sổ, từ đường giá H1 giữa fill vào và fill ra, tính **tỷ lệ vị thế mà mỗi arm sẽ kích hoạt**. In bảng đầy đủ (arm x sổ x kim loại). Arm có tỷ lệ kích hoạt **bằng 0** trên cả hai sổ là **suy biến** và bị **loại khỏi sweep trước khi chạy**, ghi số đo và ngày.
>
> Đây **không** phải peek: tỷ lệ kích hoạt là tính chất của **đường giá**, không phải của P&L hay Sharpe. **Cấm tuyệt đối** loại một arm vì Sharpe của nó, hoặc giữ một arm vì Sharpe của nó.

**Dự đoán để phép đo có thể bác bỏ**: `stop_loss_3pct` và `stop_loss_5pct` kích hoạt khoảng 0 trên vàng; `trailing_1pct` có khả năng kích hoạt trên bạc; `hold_2h` là arm duy nhất khác biệt thật sự vì nó cắt hold còn một phần tư.

**Kế toán trial**: mỗi arm sống sót x mỗi nhãn x `top_n_predictions.risk_overlay = 1` là một backtest **đăng ký ở stage `risk_overlay`**, và `resolve_solvent_carrier` xếp hạng `signal`, `allocation`, `risk_overlay` cùng nhau — **chúng là ứng viên, nên chúng là trial**. Ghi `K_final = 2.514 + n_risk_backtests` vào Trials table **trước** khi `15` chạy. Ngược lại, biến thể chi phí **không** là trial.

**Mỗi overlay phải gắn một hành động định trước**. Arm nào sống sót phải có một dòng trong `bots/exness_gold_sess/deploy/risk_config.yaml` nói nó làm gì khi kích hoạt trong live — nếu không có, arm đó không được quét.

---

# GIAI ĐOẠN 7 — Holdout

## D7.1 — Checklist tiên quyết trước khi `17_holdout_predictions` được phép chạy

Cả **mười một** mục phải xanh. Thiếu một mục thì stage không chạy; không ngoại lệ, không "chạy thử".

1. D5.3 ĐẠT: ít nhất 1 spec có DSR >= 0,95 trên **active return** tại K = 2.514, kèm `member_digest` đã ghi vào `BOT.md`.
2. **User đã duyệt** Hypothesis và Kill criteria (hiện là DRAFT) **và** quy ước DSR. Chưa duyệt thì không chạy.
3. D6.2 ĐẠT: breakeven khứ hồi > 1,5 x p90 khứ hồi, trên tài khoản **Pro thật** cho nhãn có swap, hoặc dưới khai báo `not_applicable_labels` đã đo cho nhãn không có swap.
4. `16_costs` đã chạy xanh — không có đường tắt từ `13` sang `17`.
5. `15_risk_management` đã chạy và đã ghi `CandidateSet` **`exness_gold_sess:holdout-candidates`**. Đây là tập mà `17`/`18`/`19` giải; nó là sản phẩm của `15`, không phải của `13`.
6. `resolve_solvent_carrier(CASE_STUDY_ID, admitted=ADMITTED)` trả **đúng một** carrier, và carrier ấy **thuộc một engine book** — không phải chuỗi pooled.
7. Registry khẳng định **0 hàng `prediction_sets` có `split == "holdout"`**.
8. `evaluation.holdout_start/end` vẫn đúng `2025-09-01` và `2026-08-31`, chưa từng bị sửa.
9. `assert_no_holdout` xanh trên mọi frame mà `13`/`15`/`16` đã đọc.
10. Buffer huấn luyện của `17` là **buffer rộng nhất** case study khai (`labels.variant_buffers.fwd_ret_24h = 2D`), không phải buffer của nhãn chính.
11. `bots/exness_gold_sess/tests/` toàn xanh, gồm `test_fold_geometry.py` **3 xanh** và `test_backtest_grid.py` với mệnh đề NY 7 bar.

## D7.2 — Chấm gì, một lần, và báo cáo phải chứa gì

**Chấm đúng một đối tượng**: carrier do `resolve_solvent_carrier` giải, được **fit lại** trên toàn bộ dữ liệu tới trước holdout (`17`), predict trên holdout, rồi chạy **đúng spec ấy** trên holdout (`18`). `19` khẳng định hash replay khớp; lệch thì **không báo cáo**, chạy lại `18`.

Báo cáo bắt buộc chứa, theo thứ tự:

1. `member_digest` của cohort validation, **K = 2.514**, và spec carrier (family, config, checkpoint, label, book, hold_bars).
2. Sharpe holdout **raw** và **active** so với sổ 1/N long hai kim loại **trên cửa sổ holdout**, cùng n, skew, excess kurtosis.
3. PSR so với 0 và PSR so với benchmark; **DSR tại K = 2.514** trên cả raw và active — số K **không** giảm vì holdout chỉ có một spec: K đếm phép tìm kiếm đã sinh ra carrier.
4. So sánh validation với holdout: suy giảm Sharpe, suy giảm IC, `turnover_ledger`, chi phí thực trả, max DD.
5. Bảng chẩn đoán ba mức của D5.6, tính lại trên holdout — **vẫn chỉ đọc**.
6. **Cờ nhiễm bẩn, in nguyên văn**: *"Bot v9 legacy đã giao dịch XAUUSD trên chính login này tới 2026-09-05, trùng cửa sổ holdout. Holdout này được phép **XÁC NHẬN**, không được phép **CHỌN**. Không tham số nào của bot được chọn vì nó giống một luật của v9."*

**Chữ cho từng kết cục — chép nguyên văn, cấm ứng biến:**

- **A. Xác nhận**: *"Trên holdout 2025-09-01 đến 2026-08-31, chấm đúng một lần, carrier `<hash>` cho Sharpe active `<x>` và DSR tại K = 2.514 là `<p>`. Kết quả holdout **không mâu thuẫn** với kết quả validation. Đây không phải bằng chứng rằng chiến lược sinh lợi; nó là bằng chứng rằng kết quả validation chưa bị bác bỏ bởi một cửa sổ chưa từng dùng để chọn. Holdout **đã cháy**. Bước tiếp: Giai đoạn 8 shadow, `armed: false`."*
- **B. Không xác nhận, chưa đảo dấu**: *"Sharpe active holdout `<x>` dương nhưng dưới mức đã khai; DSR `<p>` < 0,95. Kết quả **không được xác nhận**. Không chọn lại, không quét lại, không đổi ngưỡng. Holdout **đã cháy**."*
- **C. Bác bỏ**: *"Sharpe active holdout `<x>` <= 0. Kết quả validation **không sống sót** qua cửa sổ giữ lại. Bot quay về Giai đoạn 1. Holdout **đã cháy**."*
- **D. Không có gì sống sót ở Giai đoạn 5 hoặc 6** (nhánh nhiều khả năng nhất, và là nhánh **tốt** về mặt bằng chứng): *"Cổng Giai đoạn `<5|6>` không đạt tại K = 2.514 (`member_digest <digest>`). `17`, `18`, `19` **không chạy**. Cửa sổ holdout 2025-09-01 đến 2026-08-31 **chưa từng được đọc bởi bất kỳ stage hay chẩn đoán nào** và **vẫn chưa cháy**; registry khẳng định 0 hàng prediction có `split == 'holdout'`. Prior của repo, 'chưa có lợi thế nào được phân giải', đứng vững. Tiền lệ: `exness_fx_d1` chi 3.204 trial trên 5 cặp FX cùng broker, 0 spec sống sót."*

**Cấm tuyệt đối**: một câu chứa "hứa hẹn", "có triển vọng", "gần đạt", "chỉ cần thêm", hoặc bất kỳ con số lợi nhuận kỳ vọng nào.

## D7.3 — "Holdout đã cháy" nghĩa là gì về mặt vận hành

**Định nghĩa**: cửa sổ 2025-09-01 đến 2026-08-31 đã cháy kể từ khoảnh khắc `17_holdout_predictions` ghi hàng `prediction_sets` đầu tiên có `split == "holdout"`. Từ đó trở đi, **không** con số nào từ cửa sổ ấy còn có thể là bằng chứng ngoài mẫu **vô điều kiện** cho bất kỳ thiết kế nào của bot này, kể cả một thiết kế chưa tồn tại — vì người ra thiết kế tiếp theo đã nhìn thấy nó.

Nhánh:

- **Đã cháy, kết cục A** thì sang Giai đoạn 8. Bằng chứng ngoài mẫu tiếp theo là **tape shadow tiến về phía trước** (demo, `shadow_mode: true`, `armed: false`, magic 260903), chứ không phải một lát cắt khác của lịch sử. Kill criteria phải được user duyệt **trước lệnh live đầu tiên**.
- **Đã cháy, kết cục B hoặc C** thì về Giai đoạn 1. Thế hệ mới **phải khai một holdout mới**, và trên bot này chỉ còn một lựa chọn trung thực: **holdout tương lai** kiểu `xau_fx_mt5`, vì lịch sử đã hết chỗ chưa nhìn. Ghi rằng bot đã đổi quy ước holdout và vì sao.
- **Chưa cháy (kết cục D)** thì ghi rõ **chưa cháy** vào `BOT.md` phase-7 row, kèm khẳng định 0 hàng holdout trên đúng số hàng `backtest_runs` đã có. Một holdout chưa cháy là **tài sản của thế hệ sau**; đừng tiêu nó để có một dòng báo cáo.

Trong mọi nhánh: một hàng Trials table có ngày, số K tại thời điểm chấm, `member_digest`, và câu nói rõ holdout đã cháy hay chưa.

---

# XUYÊN SUỐT

## X.1 — Ba thứ dễ hỏng nhất, riêng cho bot này

**(1) Cross-section hai tên biến mọi cơ chế xếp hạng thành một giao dịch chênh lệch vàng/bạc, âm thầm.** `_select_top_bottom` (`allocation.py:34`) cho `effective_k = min(top_k, 2 // 2) = 1`, nên mọi allocator bắt buộc long một kim loại và short kim loại kia mỗi slot; `14_portfolio_management.py:386` còn dựng lại signal thành `equal_weight_top_k` và **vứt** `session_filter`, `long_short`, `drop_friday` cùng khối `risk` mang hold. Cùng lớp lỗi đã ăn `ic_mean` của registry (`min_obs=5` trên panel rộng 2 nên null toàn bộ) và đã ăn `auc_mean_daily`. **Phòng thủ**: D6.1 đóng chiều allocation; `_report_phase5.py` khẳng định `spec_key` của mọi thành viên chứa `session_filter` và `hold_bars`. **Dấu hiệu nhận biết khi nó xảy ra**: `net_exposure` của mọi phiên gần 0 trong khi tín hiệu lẽ ra phải cho những phiên cả hai tên cùng dấu.

**(2) IC đo trên nhãn 8 giờ, P&L kiếm trên cửa sổ 7 giờ ở New York.** Đã khai, đã đo (lệch **60 phút, 100 % hàng NY**), và không sửa được bằng một số bar nào vì giờ đóng phiên New York **chính là** đầu giờ nghỉ Globex ở cả hai mùa. Nguy hiểm: 1.257 của 2.514 trial (sổ `ny` cộng phần NY của pooled) được **xếp hạng** bằng một thống kê đo trên cửa sổ mà chúng **không giao dịch**. Nếu giờ cuối phiên New York mang phần lớn tín hiệu, sổ NY sẽ trông tốt trên IC và tệ trên P&L; nếu ngược lại, nó ăn may. **Phòng thủ**: chẩn đoán `corr(fwd_ret_7h_probe, fwd_ret_8h)` cộng IC trên probe 7 giờ, riêng hàng NY, ở D5.6 — **báo cáo, không bao giờ chọn theo nó**. Nếu IC suy giảm đáng kể, câu trả lời trung thực là **niêm phong lại nhãn NY ở chân trời giao dịch được**, và đó là identity mới, khai trước, K cộng dồn.

**(3) Xu hướng tăng của vàng 2017-2025 làm mọi sổ thiên mua đẹp trên raw return.** `fixed_threshold_0` phát tín hiệu long bất cứ khi nào điểm số dương; trên một chuỗi có drift dương, một mô hình gần như vô dụng vẫn cho Sharpe raw dương. `analytics.py:376-377` đã ghi rằng thống kê pooled *"also rewards a model for the base rate moving through time"*. **Phòng thủ**, ba lớp, đều bắt buộc: (a) cổng là **active return**, không phải raw; (b) **cổng phơi nhiễm** in ra tỷ lệ phiên `|net|/gross > 0,5`; (c) `expected_max_sharpe` dưới giả thuyết không được in **trước** tên spec nào. Bối cảnh: `xau_fx_mt5` chạy 3 trial trên XAUUSD cùng tài khoản và đóng cả hai đường H1 lẫn D1.

## X.2 — Builder KHÔNG được làm

1. **Không** chạy `14_portfolio_management` (D6.1) — và nếu ai bỏ cổng đó thì **không** để nó dựng lại signal dict.
2. **Không** đọc `avg_turnover` của registry, và **không** dựng lại turnover từ `weights.parquet` (D5.2).
3. **Không** so `breakeven_bps` mỗi lượt với `round_trip_p90_bps` khứ hồi (D6.2a).
4. **Không** lấy breakeven từ envelope max-over-config của `plot_cost_decay`, cũng không từ quy tắc bậc thang `strategy_analysis.py:2213-2221`, làm số của cổng.
5. **Không** chạy `16_costs` khi `costs.swap.measured_on_account` chưa là `real`, trừ nhánh `not_applicable_labels` đã **đo** (D6.3) và đã ghi.
6. **Không** thêm, bớt hay đổi một điểm nào của `cost_grid_bps`, `safety_margin`, `confidence_level`, `session_books`, `signal_specs` **sau khi** đã nhìn thấy một kết quả.
7. **Không** giữ hay loại một risk arm vì Sharpe của nó; chỉ vì **tỷ lệ kích hoạt bằng 0** (D6.4).
8. **Không** để chuỗi pooled trở thành carrier, thành viên population, hay hàng `cohort_metrics` (D5.4).
9. **Không** hạ K xuống `n_trials_effective_mp` hay `_er` để qua cổng (D5.3).
10. **Không** chạy `17`/`18`/`19` khi bất kỳ mục nào trong checklist D7.1 còn đỏ — đặc biệt mục 2 (user chưa duyệt Hypothesis, Kill criteria và quy ước DSR).
11. **Không** viết một câu kết luận nào ngoài bốn mẫu A/B/C/D ở D7.2, và **không** đưa bất kỳ con số lợi nhuận kỳ vọng nào vào bất kỳ tài liệu nào của bot.
12. **Không** sửa `_CADENCE_CALENDAR_DAYS_PER_PERIOD`, `metrics.py::min_obs`, hay `apply_universe_filter` để chiều bot này — chúng thuộc các case study có run log **không** rỗng.

---

**Next step**: chép chữ ký nghiệm thu D5.1 đến D5.3 (K = 2.514 tách 1.676 cộng 838, thứ tự chạy cố định, `member_digest` sẽ ghi ở đâu) vào `bots/exness_gold_sess/BOT.md` **trước**, rồi sửa đúng một câu in sai ở `case_studies/exness_gold_sess/13_backtest.py:676-680`.

**Read**: `case_studies/utils/uncertainty.py:1085-1201` (`member_digest` ở `:1109-1113`, hiệu chỉnh MP/ER ở `:1168-1179`) cùng `case_studies/exness_fx_d1/_report_phase5.py:271-340, 390-408` — Tập 1, Ch16 notebook `12_dsr_validation` và `14_cost_sensitivity`; và `case_studies/utils/allocation.py:22-46` cùng `case_studies/fx_pairs/14_portfolio_management.py:192-196, 386` — Tập 1, Ch20 notebook `06_cost_survival` và `04_signal_to_strategy`.
