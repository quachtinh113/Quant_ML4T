# GEN2_DECLARATION_2026-09-10 — thế hệ 2 của `exness_gold_sess`

Ngày 2026-09-10 · Mentor declaration · Repo root `D:/05_Quant/machine-learning-for-trading` (branch `exness-bots`) · Viết **trước** lần fit đầu tiên của thế hệ 2; registry hiện tại: `training_runs` 216, `prediction_sets` 801 (toàn `validation`, 0 `holdout`), `backtest_runs` 1.674, `cohort_metrics` 0, `candidate_sets` 0 (`bots/exness_gold_sess/BOT.md:104`).

Trạng thái thế hệ 1 mà tuyên bố này kế thừa, không bàn lại: cổng D5.3 **KHÔNG ĐẠT** tại K = 2.514, `member_digest 765830e9fe95a8e06167ca89317282c02e6f9dcaf4e00685bf1bdcc5817e287b`, 0/2.514 dưới cả `dsr_notebook` lẫn `dsr_library` (`BOT.md:104`, `PHASE567_DECLARATION.md:126`); kết cục D7.2-D, holdout 2025-09-01 → 2026-08-31 **chưa cháy** (`BOT.md:106`).

Vị trí roadmap của cả tuyên bố: đây là vòng lặp về **Giai đoạn 1** mà `roadmap.md:215` bắt buộc sau một cổng không đạt, đi lại các cổng 2-5 với K cộng dồn. Guard toàn văn (`mentor-protocol.md:20-33`): **Multiple testing** (K cộng dồn, không cắt theo kết quả), **Evidence boundary** (holdout không đọc), **Parity** (nhãn = cửa sổ giao dịch), **Point-in-time** (lịch biết trước), **Costs**, **Risk actions**, **Safety**, **Decay**.

---

## 0. Luật chung thế hệ 2

**L0.1 — Cùng case study, cùng experiment, cùng `registry.db`. Phán quyết dứt khoát: KHÔNG tạo case study mới.**
Cơ chế: mọi identity là content-addressed — `backtest_hash = SHA256(canonical_json({prediction_hash, strategy_spec, identity_version}))` (`case_studies/RUN_LOG.md:339-368`), nên một spec mới sinh identity mới còn identity cũ được **phục vụ lại, không tính lại** (đã đo: `13_backtest` chạy lại "0 computed, 1674 served", `BOT.md:879`). Lineage đi theo **tên population**: `population_supersedes` chỉ cho phép hash đã khai khi nó là `current.supersedes` hoặc `current.hash` (`population.py:411`); `superseded_members` hỏi theo tên (`population.py:576-628`). Tiền lệ: `exness_fx_d1` cộng dồn K = 1.068 → 2.136 → 3.204 trong **một** registry, "K accumulates over the bot, not over a population" (`bots/exness_fx_d1/BOT.md:147, :169`), và `_report_phase5.py` của nó đi qua mọi generation của mọi population (`case_studies/exness_fx_d1/_report_phase5.py:110-134`). Một case study mới sẽ khởi động lại K ở 0 — chính điều FOLD_GEOMETRY §3 ("hoãn không phải là xoá") và D6.2(d) ("2.514 trial cũ vẫn được đếm") cấm.
`setup_version` được nâng `v1 → v2` **chỉ như tài liệu**, kèm comment "read by no code, enters no hash" (đo 2026-09-10: 0 reader trong `case_studies/`, `utils/`).

**L0.2 — K2 khai TRƯỚC lần fit đầu tiên; tổng K = 2.514 + K2.**
```
nhãn mô hình hoá thế hệ 2 : fwd_ret_sess                          = 1
prediction sets            : 28 linear + 15 GBM x 10 checkpoint  = 178   (menu không cắt, L0.5)
engine books               : london, ny                          = 2
signal specs               : fixed_threshold_0 (signed), per_symbol_p80 = 2
=> engine backtests đăng ký = 178 x 2 x 2 = 712
=> chuỗi pooled (số học)    = 178 x 1 x 2 = 356
=> K2                       = 712 + 356   = 1.068
=> K tại cổng D5.3 thế hệ 2 = 2.514 + 1.068 = 3.582
```
K2 được viết ở đây, ngày 2026-09-10, với `backtest_runs` = 1.674 và 0 hàng thế hệ 2 tồn tại. Cùng công thức D5.1: `K = |books+pooled| × Σ_label |prediction sets| × |signal specs|`. Nếu Giai đoạn 6 mở: `K_risk` cộng thêm (khối 6), khai trước khi `15` chạy. Mọi lựa chọn "giữ nhãn X" ở khối 2 có giá trial ghi ngay cạnh; giá đó cộng vào K2 **chỉ khi được chọn trước lần fit đầu tiên**.

**L0.3 — Digest nào được phép dịch.** Ba ngoại lệ §5 của `PRICE_GRID_DECLARATION.md` (fwd_ret_24h A-prime, kalman_smoothness) đã dùng hết cho thế hệ 1 vì thế hệ 1 lúc đó chưa có hàng đăng ký. Thế hệ 2 có registry không rỗng, nên luật mới là: **một digest chỉ được dịch ở đúng chỗ tuyên bố này nói nó dịch, và mọi cú dịch khác = dừng, báo cáo.**

| Artifact | Giá trị hiện hành | Thế hệ 2 |
|---|---|---|
| `session_panel` `526e4afb5f1679cc` | bất động | **KHÔNG dịch** (khối 1 không đụng panel) |
| `market_data` `6d63567e7b83eea2` | bất động | **KHÔNG dịch** (`02_labels.py:179-181` lấy trên hàng có endpoint; nhãn mới không đổi tập đó) |
| `features/financial.parquet` `0557461a95579969` | bất động | **KHÔNG dịch** trong thế hệ 2 (khối 3 không thêm cột — L0.5); dịch **chỉ** khi `FRED_API_KEY` tới trước lần fit đầu (khối 3, có ngày) |
| `labels/*.parquet` | `9fbf73e65699693c / 1a96d0952eb9d1de / 3b76fd3d690b7fa9` | **dịch** — `02_labels` publish nhãn mới `fwd_ret_sess`; ba parquet cũ **không xoá**, không republish |
| `features/model_based.parquet` `881f249635f32deb` | bất động | `digest` **KHÔNG dịch**, chỉ `inputs["labels:<primary>"]` dịch — đây là **falsifier** của khối 2 (cùng chữ ký PRICE_GRID §5: "vintage của timeline nhãn dịch, giá trị được fit thì không") |
| mọi `prediction_hash` thế hệ 1 | 419 current | **giữ nguyên hàng**, retire theo lineage (khối 4) |
| mọi `backtest_hash` thế hệ 1 | 1.674 | **giữ nguyên**, không xoá, không chạy lại |

**L0.4 — Holdout.** `evaluation.holdout_start/end` = `2025-09-01 / 2026-08-31` **không sửa** (`setup.yaml:643-644`), chưa cháy (`BOT.md:106`); cờ nhiễm bẩn `legacy_v9_contaminated_window` (`setup.yaml:656-657`) vẫn áp: holdout **XÁC NHẬN**, không **CHỌN**. D7.3: một holdout chưa cháy là tài sản; thế hệ 2 dùng **cùng** cửa sổ, không khai cửa sổ mới.

**L0.5 — Những gì KHÔNG được làm vì đã nhìn thấy kết quả thế hệ 1** (`RULINGS_2026-09-08.md:179-185`, `PHASE567_DECLARATION.md` X.2):
1. **Không cắt menu** `config/training/*.yaml` — không bỏ họ `huber` (41 khoảng âm trên 24h), không bỏ checkpoint, không giữ riêng `default_huber@450` (leader IC thế hệ 1). Menu 28 + 15×10 chạy nguyên.
2. **Không dịch ngưỡng**: `threshold: 0.0`, `long_q: 0.80`, `lookback_days: 63` giữ nguyên (`setup.yaml:385, :402`). Chỉ **thêm** khoá quy ước `threshold_convention` — một defect đã ghi ngày ở thế hệ 1 (`BOT.md:907` mục 6, `:925`).
3. **Không đảo dấu** `fwd_ret_24h` thành short, không thêm spec `short_only` (`BOT.md:104` (ii)).
4. **Không** giữ/loại risk arm theo Sharpe (D6.4); **không** hạ K xuống `n_trials_effective_*` (D5.3); **không** đụng `cost_grid_bps`, `safety_margin`, `confidence_level`, `session_books` (X.2 mục 6); **không** quét trọng số sleeve (`setup.yaml:361-362`).
5. **Không** đưa `dir_tb_8h` trở lại vì "feature mới có thể giúp" — nhãn đã CLOSED theo `setup.yaml:873-876`.
6. **Bốn khoá pin** (`labels.rebalance_step`, `labels.classification_eval_label`, `universe.cost_feasible`, `backtest.sweep.htm_cost_cascade.liquid_quantile`, `.claude/skills/ml4t/references/case-studies.md:141-146`): ba giá trị hiện có của `rebalance_step` **không sửa**; `classification_eval_label` **không sửa** (entry `dir_tb_8h` trở thành trơ). Thế hệ 2 chỉ **THÊM** `rebalance_step.fwd_ret_sess: 1` — thêm cho một nhãn chưa từng có hàng nào không thể "trộn hai phương pháp trong một registry" (`case-studies.md:151-152`), và phải được viết **trước** hàng đăng ký đầu tiên của nhãn ấy rồi không bao giờ sửa.

**L0.6 — Thứ tự xây và ranh giới "khối xong".** Xây theo thứ tự 1 → 2 → 3 → 4 → 5 → 6 → 7 (mã + test), rồi mới ghép (mục 8). "Khối xong" = test của khối **xanh**, mọi lần chạy engine là `register=False`, và **số hàng registry đếm trước và sau bằng nhau** (`training_runs` 216 / `prediction_sets` 801 / `backtest_runs` 1.674 / `official_populations` 24), theo đúng cách `tools/presweep_measurements.py::registry_counts` (`:99`) đã làm. Khối 4 là khối duy nhất có ranh giới đặc biệt: "xong" = menu, lineage, K2 khai, `request.resolve()` khô — **0 fit**; fit là bước 1 của mục 8.

**L0.7 — Mỗi khối một tiên đoán viết trước** (falsifier, mục "Tiên đoán" trong từng khối), cùng chuẩn D6.2(e) thế hệ 1: số ghi trước, run là phép thử; sai thì ghi là sai, không sửa số.

---

## 1. Khối 1 — Dữ liệu + lịch

**Mục đích**: một lịch đóng cửa/đóng sớm **có provenance** cho (i) kill criterion (g) tinh chỉnh (`BOT.md:72-85`), (ii) tàn dư early-close 1,6 % sổ NY mà `NY_EXIT_DECLARATION.md:166` hoãn "khi có file lịch", (iii) Christmas-2024 carry (`BOT.md:943`); cộng assertion tập khoá nhãn = panel (`BOT.md:947`). Guard: **Point-in-time** (lịch phải biết trước), **Parity**. Roadmap: Giai đoạn 1-2, cổng `roadmap.md:93-94` ("mọi feature tính lại được tại thời điểm quyết định; có test lookahead").

**Nguồn provenance, phán quyết**: dùng gói lịch repo đã pin, `exchange-calendars>=4.5` (`pyproject.toml:149`), qua đúng đường `utils/cv_splits.py:55-71` (`_CALENDAR_MAP`) mà thang fold của bot đã đi. Provenance = tên gói + version + id lịch + ngày đối chiếu. **Tape là sự thật cho quá khứ, lịch là tuyên bố cho tương lai**: file bot-local `bots/exness_gold_sess/calendar/closures_metals.parquet` (cột `date, symbol, kind ∈ {closed, early_close}, close_utc, source, source_version, reconciled_on`) sinh bởi `bots/exness_gold_sess/tools/build_closure_calendar.py`, chỉ đọc; **không** gõ tay ngày nào — mỗi hàng hoặc đến từ gói lịch, hoặc đến từ tape (`_features.decision_grid` báo cáo `unresolved_dates` per symbol, `_features.py:295`) và ghi rõ nguồn nào.

**Đối chiếu bắt buộc, in ra**: 18 lỗ Mon-Fri và 34 early-close/kim loại cửa sổ validation (`data_census_2026-09-08.md` §3, `BOT.md:508`) so với lịch; ngày tape có mà lịch không (Christmas Eve loại này) in tên, không lọc.

**File tạo/sửa**: `tools/build_closure_calendar.py` (mới); `calendar/closures_metals.parquet` (mới, artifact có sidecar digest qua `write_artifact`); `_features.py` thêm `declared_closures(...)` đọc file này (không đụng `decision_grid`, `session_panel`, `build_features` — panel bất động); `tests/test_calendar.py` (mới); `tests/test_fold_geometry.py` thêm một test.

**`setup.yaml`**: thêm khối tài liệu `decision.closure_calendar: {path, source: exchange_calendars, source_version: <đo>, reconciled_on: 2026-09-XX, tape_wins_for_history: true}`. **Không đụng** 4 khoá pin, `universe.history_start`, `decision.snapshots`.

**Digest dịch**: không. `session_panel 526e4afb5f1679cc` phải tái tạo (test).

**Trial accounting**: **+0**. Lịch là dữ liệu; luật lọc theo lịch (khối 5) là luật biết trước, cùng lớp `drop_friday` (`PRICE_GRID_DECLARATION.md` §3.3).

**Tests bắt buộc**
- `test_calendar.py::test_every_mon_fri_hole_of_the_census_is_a_declared_closure` — 18 lỗ ⊂ lịch.
- `::test_early_close_rows_of_the_panel_are_declared_or_named` — với mọi hàng `label_end_ts` null không phải stale-decision: ngày nằm trong lịch **hoặc** được in vào bảng "tape-only"; số tape-only ≤ số ghi trong BOT.md sau lần chạy đầu (pin lại có ngày).
- `::test_the_daily_break_is_not_a_closure` — không hàng nào của lịch trùng giờ nghỉ 21:00/22:00 UTC (đó là trạng thái đã khai, `BOT.md:77-85`).
- `::test_no_closure_row_reads_a_price` — file dựng từ lịch + tập `unresolved_dates`, không từ return nào.
- `test_fold_geometry.py::test_label_key_set_equals_the_decision_grid` — `set(keys(labels/*.parquet)) == set(keys(session_panel dev))` (một dòng, `BOT.md:947`; cơ chế `02_labels.py:713`).
- Chạy lại nguyên: `test_data_quality.py::test_the_sparse_prefix_is_outside_the_declared_history` (`:197`), `::test_zero_spread_bars_are_counted_and_not_priced_as_free` (`:379`) — nhánh pre-2017 và spread==0 đã có test, không thêm.

**Definition of done**: 6 test xanh; `session_panel` digest bất động; 0 hàng registry.

**Phụ thuộc**: không. **Chặn bởi user**: không. **Chi phí**: < 5 phút CPU (đọc parquet + lịch).

**Tiên đoán**: lịch gói tái tạo **cả 18** lỗ Mon-Fri; tập early-close NY của tape khớp tập ngày lễ Mỹ của lịch trên ≥ 30/34 ngày mỗi kim loại, phần lệch (≤ 4/kim loại) là Christmas Eve/đóng sớm không chuẩn hoá và được in tên; 2018-01-31/02-01 xuất hiện là **tape-only** (outage, `BOT.md:993`), không phải ngày lễ. Test tập khoá xanh ngay lần đầu (A′ đã bảo đảm, `BOT.md:861` (h)).

---

## 2. Khối 2 — Nhãn

**Mục đích**: nhãn chính đo **đúng cửa sổ được giao dịch** trên cả hai venue. Guard: **Parity** (Ch25 §25.1, `roadmap.md:302` sai lầm 5: hai đường code phân kỳ), **Point-in-time**, **Multiple testing**. Roadmap: Giai đoạn 1, cổng `roadmap.md:72-73` (nhãn niêm phong tại endpoint; fold + holdout khai trước fit).

**Phán quyết 2.1 — một nhãn, một luật, hai hằng số: `fwd_ret_sess`.** Không phải "hai nhãn 8h và 7h" (sẽ chéo `fwd_ret_7h` với sổ London và tạo tàn dư ngược chiều, +1.068 trial vô ích), không phải "7h cả hai" (đưa tàn dư vào London, nơi đang lệch 0 phút). Cùng hình dạng `NY_EXIT_DECLARATION.md` §1.2-1.3: *bất đối xứng của hằng số, không của luật*.
Luật, trên panel: `tradable_exit_ts = max{ c ≤ label_end_ts : tồn tại bar H1 mở tại c }` — bar cuối mà lệnh thoát `next_bar_open` còn khớp được (`PRICE_GRID_DECLARATION.md` §1.1: fill tại open của hàng khoá `c+60` = giá tại `c`). `fwd_ret_sess = close(tradable_exit_ts) / close(decision_ts) − 1`. Trên London nó **bằng** `fwd_ret_8h`; trên NY nó **bằng** `fwd_ret_7h_probe` đã dựng ở `_report_phase5.py:1219-1228` (`label_end_ts − 60min`). Không có luật mới — đây là probe thế hệ 1 được niêm phong thành nhãn.
Điểm-thời-gian: `label_end_ts` là giờ đóng venue theo lịch; bar tồn tại hay không chỉ biết tại `label_end_ts` — hợp lệ cho một **nhãn** (niêm phong tại endpoint); còn hằng số **hold** của backtest vẫn do `_hold.derive_hold_bars` (`_hold.py:287`) khẳng định hằng trong mỗi sổ.

**Phán quyết 2.2 — `fwd_ret_8h` rời menu mô hình hoá.** Không phải vì kết quả (0/178): vì cửa sổ NY của nó không được giao dịch, khuyết tật đã khai **trước** sweep (`setup.yaml:312-318`, X.1(2)) với lối thoát đã viết sẵn ("re-seal at a tradable horizon"). 1.068 trial của nó ĐÃ CHI và vẫn đếm.

**Phán quyết 2.3 — `fwd_ret_24h`: ba lựa chọn, giá trial, và lựa chọn.**

| Lựa chọn | Trial cộng vào K2 | Ghi chú |
|---|---|---|
| Giữ nguyên (fit lại trên matrix thế hệ 2) | **+1.068** (178×2×2 + 178×2) | Sổ Mon-Thu (`BOT.md:1017-1021`), swap đúng 1 đêm/vị thế 2.131/2.131 (`BOT.md:105`), `16_costs` **bị chặn** trên Pro thật (D6.3) — ngay cả khi qua D5.3 nó không thể mở Giai đoạn 6 |
| Bỏ | 0 | **Cấm** nếu lý do là 41 khoảng âm — đó là chọn theo kết quả |
| Đổi | — | Không có gì để đổi: exit của nó đã là quyết định cùng venue kế tiếp, backstop 23 dẫn xuất |

**Phán quyết: HOÃN (không phải bỏ), lý do cơ học không phụ thuộc kết quả**: một nhãn mà chi phí không định giá được (chờ user đọc Pro thật, D6.3) không thể đi hết Giai đoạn 6, nên 1.068 trial cho nó ở thế hệ 2 là trial chi cho một sổ không thể mở cổng kế tiếp. Nó **trở lại như một cohort riêng (+1.068 tại menu lúc đó)** ngay khi `costs.swap.measured_on_account: real` — không cắt preset, không đảo dấu, không chọn vì số thế hệ 1. Ghi vào Trials có ngày.

**Phán quyết 2.4 — `dir_tb_8h`: CLOSED, 0 trial.** `setup.yaml:873-876` `on_failure` là hệ quả tiền đăng ký; cổng (2) đã đo và FAILED (`BOT.md:480`); cổng (3) chỉ đánh giá được khi có breakeven từ `16_costs`. Thứ tự trung thực: thế hệ 2 hồi quy trước; nếu Giai đoạn 6 mở, `dir_tb` là câu hỏi thế hệ 3 với cổng (3) đánh giá được và giá **+378** (63×3×2) tại menu hiện tại. Khoá pin `classification_eval_label` **không sửa** — entry `dir_tb_8h` trơ.

**File tạo/sửa**: `02_labels.py` — thêm `fwd_ret_sess` theo luật 2.1 (đặt cạnh §C.2, `02_labels.py:217-248`), `LABEL_NAMES` vẫn bind từ `setup.yaml` (`:100-102`); publish trên tập khoá lưới quyết định (`:713`) — **không** đổi. `_features.py`: không đổi. `04_model_based_features.py`: không đổi mã; chạy lại **đúng một lần** để `inputs` dịch. `05_evaluation.py`: chạy lại.

**`setup.yaml`**: `labels.primary: fwd_ret_sess`; `labels.variants: []`; `labels.horizons.fwd_ret_sess: 8H` (cận trên, London 8 bar; buffer resolver đọc khoá này chứ không parse tên — `BOT.md:27`); thêm `labels.rebalance_step.fwd_ret_sess: 1` (**THÊM**, ba entry cũ giữ); thêm khối tài liệu `labels.tradable_exit: {rule: last_h1_bar_open_at_or_before_label_end, by_book_bars: {london: 8, ny: 7}, derived_by: _hold.derive_hold_bars, equals: {london: fwd_ret_8h, ny: fwd_ret_7h_probe}}`; `backtest.exit_gap_minutes_by_book` thêm dòng `against_fwd_ret_sess: {london: 0, ny: 0}` (giữ `by_book` cũ vì nó là sự thật về tape so với giờ đóng phiên). Giữ nguyên `variant_buffers`, `triple_barrier`, `classification_scoring` (trơ, tài liệu). **Builder phải kiểm bằng config-load khô** rằng `variants: []` với các khoá trơ ấy không làm `research/labels.py`/`load_model_configs` raise; nếu raise, xin phán quyết — không xoá khối pin.

**Digest dịch**: `labels/fwd_ret_sess.parquet` mới; `04` `inputs["labels:fwd_ret_sess"]` mới; `04` `digest 881f249635f32deb` **PHẢI tái tạo**; `market_data 6d63567e7b83eea2`, `session_panel`, `financial` bất động.

**Trial accounting**: **+1.068** (nhãn duy nhất của K2); `fwd_ret_24h` +0 (hoãn), `dir_tb_8h` +0 (closed).

**Tests bắt buộc** (`tests/test_labels_gen2.py`, mới)
- `test_fwd_ret_sess_equals_fwd_ret_8h_on_every_london_row` — max|diff| = 0.0 trên 100 % hàng London có endpoint.
- `test_fwd_ret_sess_equals_the_seven_hour_probe_on_every_ny_row_the_folds_reach` — max|diff| = 0.0 so với công thức `_report_phase5.py:1224-1228`; hàng NY trước 2018-08-30 (102 ngoại lệ 2017, `BOT.md:852`) in riêng.
- `test_the_tradable_exit_reproduces_the_derived_hold` — `bars(tradable_exit_ts − decision_ts)` == `HOLD_BARS[book]` từ `_hold.derive_hold_bars` trên 100 % hàng trong tầm fold.
- `test_the_new_label_is_sealed_at_its_own_endpoint` — mở rộng `test_lookahead.py::test_session_close_reproduces_the_sealed_label` (`:173`) cho nhãn mới.
- `test_fold_geometry.py` 3 test cũ vẫn xanh + test tập khoá của khối 1.
- Chữ ký `04` sau chạy lại, viết trước: `digest == 881f249635f32deb`, 24.740 hàng, 9 cột, 5 fold, 20 trường `fold_geometry` **không đổi một trường**, ba seal `0.00e+00`, purge 3-5 slot.

**Definition of done**: tests xanh; `04` tái tạo digest; `05` chạy lại và verdict được **ghi**, ngưỡng không đụng (luật 2(i) `RULINGS §2.3`); 0 hàng registry.

**Phụ thuộc**: khối 1 (assertion tập khoá). **Chặn bởi user**: không. **Chi phí**: `02` vài phút; `04` ≈ 13 phút (`BOT.md:258`); `05` vài phút.

**Tiên đoán**: Pearson NY giữa `fwd_ret_sess` và `fwd_ret_8h` = **0,9852 / 0,9886** đúng bốn chữ số (đó chính là probe); `04` digest tái tạo; `05` trên nhãn mới: 33 cột financial đổi IC (nhãn đổi trên hàng NY) nhưng **0** cột qua Benjamini-Hochberg, số STOP vẫn 4 (staleness không phụ thuộc nhãn).

---

## 3. Khối 3 — Feature

**Mục đích**: hai họ PLANNED có mã + test chạy được trên vintage tổng hợp (mẫu carry của `exness_fx_d1`, `bots/exness_fx_d1/BOT.md:146, :182`), screen staleness không phụ thuộc thang đã tiền đăng ký (`BOT.md:736-772`), quyết định về ba cột trùng và `kalman_slope_zscore`. Guard: **Point-in-time** (macro theo ngày công bố, Ch08 §8.4), **Multiple testing** (feature-set change là chọn theo số?). Roadmap: Giai đoạn 3, cổng `roadmap.md:118-119` (register khai trước; feature sống sót là bộ lọc, không phải bằng chứng).

**Phán quyết 3.1 — hai họ PLANNED: xây mã, test trên vintage tổng hợp, GIỮ trạng thái PLANNED trong `setup.yaml` cho tới khi user mở khoá.** `real yields and the dollar` cần `FRED_API_KEY` (`setup.yaml:610-619`); `CPI and FOMC release windows` cần lịch công bố có provenance — đó là **khối 1 mở rộng**: cùng file `closures_metals.parquet` KHÔNG chứa lịch tin; lịch tin là file thứ hai `calendar/releases_us.parquet` với nguồn phải là dữ liệu có ngày công bố (FRED/ALFRED release calendar qua `data/macro` schema, `bots/_shared/macro_config.yaml` như fx_d1), nên cũng chặn trên `FRED_API_KEY`. Không gõ tay 90 ngày (`setup.yaml:604`).
Kế toán: nếu khoá tới **trước** lần fit đầu của thế hệ 2, hai họ vào matrix thế hệ 2 với **+0** (K2 = 1.068 đã bao 178 set trên matrix nào cũng vậy) — nhưng `financial` digest dịch và phải ghi. Nếu tới **sau**, mỗi họ là một cohort mới **+1.068** tại menu lúc đó (`bots/exness_fx_d1/BOT.md:169`: "carry +1,068 planned when the data arrives").

**Phán quyết 3.2 — screen mới đứng cạnh, không thay** (`BOT.md:751-756`, ngưỡng cố định 2026-09-08: `n_unique/n < 0.01` STOP; CV median < 1e-6 STOP, 1e-6..1e-4 REVISE; modal share > 0.20 REVISE, > 0.50 STOP; zero-mean safe). Cài ở `05_evaluation.py` như một bảng **thứ hai** in cạnh bảng cũ (`:333-349` không sửa — dùng chung 4 case study); ledger thêm cột `scale_free_verdict`. Nó là screen trước IC, không đọc IC.

**Phán quyết 3.3 — ba cột `slot_ret_1 / kalman_innovation / arima_residual`: GIỮ cả ba.** Lý do: register đã dự đoán (`BOT.md:1030-1032`), ledger `05` đã gắn cờ nhóm redundancy tại cut 0,70 đã khai (`setup.yaml:538`), và một threshold "≥ 0,99 thì gộp" đặt ra **sau** khi thấy 1,000/0,995 là ngưỡng dịch sau kết quả — đúng lớp edit `RULINGS §2.1` cảnh báo. Đa cộng tuyến là việc của ridge/lasso/GBM, không phải của K. **`kalman_slope_zscore`: GIỮ, gắn cờ** — phán quyết `RULINGS §2.2` đứng; không chặn `observation_noise` (một lựa chọn sau khi thấy bất ổn, `BOT.md:871`). Screen 3.2 áp lên cả hai.

**File tạo/sửa**: `_features.py` thêm `real_yields_dollar(...)`, `release_windows(...)` (đọc vintage có `vintage_date` + `available_at = vintage_date + lag ≥ 1 ngày`, từ chối panel aligned — y hệt `exness_fx_d1/_features.py::load_carry_vintages`), nối vào `build_features`/`features_as_of` (`:1001`, `:1094`) qua tham số tuỳ chọn, mặc định None → **không đổi matrix**; `warmup_expectations` (`:1035`) thêm entry; `05_evaluation.py` thêm bảng screen 3.2; `bots/_shared/macro_config.yaml` thêm series vàng (`DFII10`, `DTWEXBGS` hoặc tương đương — id do `_check_carry_series`-style script xác nhận **trước** khi fetch; **không** sửa `data/macro/config.yaml`); `tests/test_macro_pit.py` (mới, vintage tổng hợp), `tests/test_scale_free_screen.py` (mới).

**`setup.yaml`**: hai họ vẫn `pattern: PLANNED`; thêm `features.windows.real_yield_zscore: 252`, `features.windows.release_window_minutes: 30` (đọc từ `bots/assets/XAUUSD.md:38-39` ±30 phút) — tài liệu tới khi bật. Thêm `features.scale_free_screen: {n_unique_ratio_stop: 0.01, cv_stop: 1e-6, cv_revise: 1e-4, modal_share_revise: 0.20, modal_share_stop: 0.50, declared_on: 2026-09-08}`.

**Digest dịch**: **không** trong thế hệ 2 (`financial 0557461a95579969` bất động vì họ mới không bật). Dịch chỉ khi khoá tới trước fit, ghi ngày.

**Trial accounting**: **+0** nay; **+1.068/họ** nếu bật sau lần fit đầu.

**Tests**: `test_macro_pit.py` — (a) observation-dating vs publication-dating khác nhau trên > 0 hàng (không rỗng), (b) cắt bỏ vintage sau `t` không đổi giá trị trước `t` (max|diff| 0), (c) frame thiếu `vintage_date` bị từ chối, (d) lag 0 bị từ chối, (e) `features_as_of` tái tạo batch trên 2 instant/mùa/venue; `test_scale_free_screen.py` — (f) áp lên artifact `04` **thế hệ 1 đã niêm phong** với cột `kalman_smoothness` được dựng lại tổng hợp từ 3 số đo (`1e10×(1−5,55e-6,1]`, 22,04 % đúng `1e10`): STOP; (g) cột z-score tâm 0 không bị CV test đánh; (h) screen không đọc nhãn (import ban tương tự `_report_phase5.py:135`).

**Definition of done**: tests xanh; `03` chạy lại tái tạo `0557461a95579969` byte-identical (họ mới tắt); 0 hàng registry.

**Phụ thuộc**: khối 1 (lịch), khối 2 (`05` chạy trên nhãn mới). **Chặn bởi user**: `FRED_API_KEY` (cả hai họ). Builder làm được: toàn bộ mã + test tổng hợp + script kiểm id series. **Chi phí**: `03` vài phút; test giây.

**Tiên đoán**: screen 3.2 trên 9 cột `04` hiện hành: **0 STOP, ≤ 1 REVISE** (ứng viên `kalman_slope_zscore`); trên 33 cột financial: 0 STOP. Nếu nó STOP một cột financial, đó là phát hiện, ghi lại, ngưỡng không đụng.

---

## 4. Khối 4 — Mô hình

**Mục đích**: menu, lineage và K2 sẵn sàng để lần fit đầu tiên của mục 8 là một phép thử, không phải một quyết định. Guard: **Multiple testing**, **Leakage across folds** (`assert_variant_folds_are_out_of_sample`, giờ 0 variant), **Evidence boundary**. Roadmap: Giai đoạn 4, cổng `roadmap.md:146-147`.

**Phán quyết 4.1 — menu KHÔNG cắt.** Cắt sau khi thấy kết quả thế hệ 1 là chọn theo kết quả (L0.5.1). `config/training/fwd_ret_sess.yaml` = bản sao nguyên văn `fwd_ret_8h.yaml` (28 linear + 15 GBM). Các mục `deep_learning`, `tabular_dl`, `causal_dml` trong menu là **trơ** vì case study không có stage 08-11 (thư mục chỉ có 01-07, 12, 13); K2 đếm những gì stage tồn tại fit. Nếu một stage DL được fork sau này: +trial, khai trước.

**Phán quyết 4.2 — `detect_label_type` đã sửa (`utils/modeling.py:502, :541`), không có nhãn phân loại trong thế hệ 2, nên `test_label_task_type.py` (3 test) chỉ cần **vẫn xanh** với `variants: []`.**

**Phán quyết 4.3 — lineage**: ba population prediction thế hệ 1 (`exness_gold_sess-linear-validation-v1`, `-linear-validation-v2`, `-gbm-validation-regression-only-v1`, `BOT.md:463-465`) phải được **supersede** để "current" = thế hệ 2 (nếu không, catalog của `13_backtest` sẽ gộp 419 + 178 = 597 set và kế hoạch 597×2×2 ≠ K2). Cơ chế: `06`/`07` thế hệ 2 publish dưới **cùng tên** với `SUPERSEDES_POPULATION = <hash tip hiện hành của tên đó>` (`population.py:411`); tip đọc read-only bằng `OfficialPopulation.one(study, name=...)` (`population.py:398`) — **không** lấy từ trí nhớ (BOT.md chỉ ghi hash bị supersede, không ghi hash tip). Hàng thế hệ 1 **giữ nguyên**, retire theo lineage (`population.py:37-79`).

**File tạo/sửa**: `config/training/fwd_ret_sess.yaml` (mới); `06_linear.py`/`07_gbm.py`: tham số `SUPERSEDES_POPULATION` (`06:118`, `07:99`) nhận hash tip — mỗi stage/tên một hash; `12_model_analysis.py`: completeness guard phải in `178 × 3 books × 2 specs = 1.068` từ chính số học của nó (`BOT.md:452`).

**`setup.yaml`**: không thêm gì ngoài khối 2; `modeling.gbm` không đổi (`num_threads` chỉ ở bản experiment, `setup.yaml:917-930`).

**Digest dịch**: 0 ở khối này (không fit).

**Trial accounting**: **K2 = 1.068 = 178 × (2 + 1) × 2**, khai ở L0.2. Fit KHÔNG là trial; backtest là (D5.1).

**Tests**
- `tests/test_gen2_menu.py::test_the_menu_resolves_to_exactly_178_prediction_identities_and_writes_nothing` — `load_model_configs` + `model_requests(...).resolve()` khô cho `fwd_ret_sess`: 43 training request (28 + 15), 178 prediction identity (`gbm_checkpoint_iterations` = 10), `registry_counts` trước == sau.
- `::test_the_three_generation_one_prediction_tips_resolve_for_supersession` — `population_supersedes(study, name, declared=tip)` trả về `tip` (không None) cho cả ba tên.
- `::test_no_label_and_no_eval_label_in_feature_names` — `feature_names` ∩ {`fwd_ret_sess`, `*_right`} = ∅ (tái dùng `test_eval_label_not_a_feature.py` cho nhãn mới).
- `test_label_task_type.py` 3 xanh; `test_fold_geometry.py` 3 xanh (0 variant → `assert_variant_folds_are_out_of_sample` trả 0 hàng, test phải nói rõ trường hợp này thay vì skip).

**Definition of done**: tests xanh, **0 fit**, `registry_counts` bất động, K2 + hàng Trials "K2 = 1.068 khai 2026-09-10, 0 chi" đã ghi vào `BOT.md`.

**Phụ thuộc**: khối 2 (nhãn), khối 3 (matrix ổn định). **Chặn bởi user**: không. **Chi phí**: giây (resolve khô); fit (mục 8) ước lượng theo tỷ lệ thế hệ 1 ba nhãn `06` ≈ 4 phút / `07` ≈ 9 phút (`BOT.md:103`) → một nhãn ≈ **1,5 + 3 phút**, cộng `12` ≈ 2 phút.

**Tiên đoán**: 43 request / 178 identity / 0 hàng; ba tip resolve; sau fit (mục 8): 178 prediction set, 43 training run, 172 fold fit (43 × 4), `12` xanh trên menu đầy đủ, và — viết trước — **0 của 178** có khoảng HAC loại trừ 0 trên pooled panel IC so với `fwd_ret_sess` (IC tối thiểu phát hiện được vẫn ≈ 0,054, D6.2(e); bar không đổi vì N không đổi).

---

## 5. Khối 5 — Backtest / signal

**Mục đích**: sửa ba defect ghi ngày thế hệ 1 **bằng khoá tường minh**, giữ vị từ plan-time, `_report_phase5.py` cộng dồn hai thế hệ. Guard: **Multiple testing** (K cộng dồn, digest hợp), **Evidence boundary** (D5.3 print order), **Costs** (fill nào bị tính). Roadmap: Giai đoạn 5, cổng `roadmap.md:167-168`.

**5.1 — `lower_threshold = 1.0 − threshold`.** Khoá thư viện đã có: `threshold_convention: signed` (`signals.py:33`, `:86-89`; dispatcher `:639` đọc từ config, mặc định `probability_mirror` để hash cũ không đổi). Khai trong `setup.yaml:385`: `{name: fixed_threshold_0, method: fixed_threshold, threshold: 0.0, threshold_convention: signed}`. Hệ quả: score hằng 0,0 giờ **flat** (không còn sổ short cố định `fd6221c0a770`-class) → engine từ chối (`backtest_runner.py:2503-2546`) → vị từ plan-time (5.3) phải áp cho **cả hai họ spec** ở thế hệ 2, mỗi họ theo đúng cơ chế của nó: `per_symbol_rolling_percentile` — `n_unique(score) == 1`; `fixed_threshold` signed — `n_unique == 1` **và** giá trị đó `== threshold`.

**5.2 — `risk: {name: baseline, position_rules: [...]}`.** `store.py:447-449`: `risk.get("name") != "baseline"` → `risk_overlay`; với `name: baseline` → `signal`, và `COVERAGE_STAGE = "signal"` / `PLAN_STAGE_KEYS` (`case_studies/research/selection_field.py:48, :118`) nhìn thấy sweep. Điều kiện: **một** lần `run_backtest(..., register=False)` với và không có `name` → `trades.parquet` byte-identical (max|diff| 0 trên `entry_time, exit_time, qty, exit_price`), `_build_position_rules` (`backtest_runner.py:2527-2551`) không raise. Rồi sửa `13_backtest.py:670` và assertion nội dung `:1048-1051` (`expected_risk` thêm `name`).

**5.3 — vị từ plan-time giữ** (`13_backtest.py:740-792`), nhưng hằng số `UNRUNNABLE_ENGINE_EXPECTED = 2` là số thế hệ 1. Thế hệ 2: số đo tại plan-time trên **score** (`n_unique`, 0 P&L), **in ra và ghi vào BOT.md trước `RUN_SWEEP=True`**, rồi stage raise nếu khác. Tên `unrunnable:<prediction_hash>/<book>/<spec>` và `unrunnable:pooled:...` giữ nguyên.

**5.4 — luật lọc early-close theo lịch khối 1** (`NY_EXIT_DECLARATION.md:166`): `signal.drop_declared_early_closes: true` cho **cả hai** sổ (London không có hàng nào trong tập, nên vô hiệu ở đó — luật một, không hai), thực thi trong `backtest_runner::apply_session_filter` (seam duy nhất, `BOT.md:843`) đọc file lịch; **cấm** lọc theo `label_end_ts` null (PRICE_GRID §3.3). Không phải trial: luật biết trước, một giá trị, khai trước sweep. Falsifier in ra: tập hàng bị lọc ⊆ tập `label_end_ts` null của NY; hàng null không có trong lịch in tên.

**5.5 — `_report_phase5.py` thế hệ 2**: tổng quát hoá theo `exness_fx_d1/_report_phase5.py:86-134`: đi qua **mọi generation** của `exness_gold_sess:session-book-baselines`; `K_DECLARED_BY_GENERATION = {1: 2514, 2: 1068}`, `UNRUNNABLE_BY_GENERATION = {1: 3 tên hiện có, 2: <đo ở 5.3>}`; `names_total = names_gen1 ∪ names_gen2`; **assert** `len(names_gen1) == 2514`, `cohort_member_digest(names_gen1) == 765830e9fe95a8e06167ca89317282c02e6f9dcaf4e00685bf1bdcc5817e287b` (falsifier: cohort thế hệ 1 nguyên vẹn), `len(names_total) == 3582` (không va chạm tên — `cohort_member_digest` dedup nên va chạm sẽ **giảm K trong im lặng**, `uncertainty.py:1053`); `member_digest_total = cohort_member_digest(names_total)`. `variance_trials` trên mọi thành viên có chuỗi (3.582 − unrunnable), cùng định nghĩa lợi nhuận. Hai DSR (`dsr_notebook`, `dsr_library`) đã có (`:396`, `:425`) — giữ. Print order D5.3 giữ. Benchmark H1 gấp D1 (`:301`) không đổi vì lưới không đổi.

**5.6 — Population thế hệ 2**: cùng tên `exness_gold_sess:session-book-baselines`, generation 2, `SUPERSEDES_SESSION_BOOK_BASELINES = "2f37b3b51a42"` (tip hiện hành, `BOT.md:104`). Cùng `method`, cùng sổ, spec sửa khoá + nhãn mới = "cùng thí nghiệm trả lời lại" → supersede (tiêu chí `bots/exness_fx_d1/BOT.md:147`). `SUPERSEDES_VALIDATION_PREDICTIONS` = tip prediction population thế hệ 2 (do khối 4 tạo) — đọc từ registry.

**File tạo/sửa**: `setup.yaml:385` (khoá quy ước), `setup.yaml:382-402` không đổi giá trị nào khác; `13_backtest.py` (`:670`, `:740-792` tổng quát hai họ, `:1048-1051`, tham số `:148-149`); `backtest_runner.py::apply_session_filter` (mở rộng có gate theo khoá — không đổi hành vi case study khác); `_report_phase5.py` (5.5); `tests/test_backtest_grid.py` bổ sung; `tests/test_gen2_specs.py` (mới).

**Digest dịch**: mọi `backtest_hash` thế hệ 2 là identity mới (nhãn mới + hai khoá mới); 1.674 hash thế hệ 1 bất động.

**Trial accounting**: **+0 ngoài K2** — khối này không thêm nhãn, sổ hay spec; ba khoá mới là defect fix khai trước (cùng lập luận `NY_EXIT §3`: tham số cố định bởi luật, không sinh Sharpe để chọn).

**Tests bắt buộc** (`register=False`, 0 hàng)
- `test_gen2_specs.py::test_signed_convention_makes_a_constant_zero_score_flat_not_short` — score hằng 0,0 → 0 vị thế dưới `fixed_threshold_0` signed (thế hệ 1: short 100 %).
- `::test_risk_name_baseline_changes_the_stage_label_and_nothing_else` — `_infer_stage(spec) == "signal"`, `trades.parquet` byte-identical với spec không có `name`.
- `::test_the_plan_time_predicate_names_every_unrunnable_identity_per_family` — trên prediction frame tổng hợp có 1 set hằng: 2 engine + 1 pooled dưới percentile **và** 2 + 1 dưới signed-threshold khi hằng == 0,0; 0 khi hằng ≠ 0.
- `::test_declared_early_closes_are_filtered_through_the_seam` — hàng bị lọc là tập lịch ∩ sổ; engine không thấy hàng ấy (rebalance schedule không chứa instant).
- `test_backtest_grid.py` 12 test cũ xanh trên nhãn mới; `::test_both_books_exit_at_the_last_fill_instant_before_their_label_endpoint` (`:518`) khẳng định thêm: exit gap so với `fwd_ret_sess` = **0** cả hai sổ.
- `_report_phase5.py` chạy trên registry **hiện tại** (chưa có thế hệ 2): in `member_digest_gen1 = 765830e9…`, thế hệ 2 `NOT COMPUTABLE`, tổng `NOT COMPUTABLE` — degrade trung thực như `BOT.md:104` (c).

**Definition of done**: tests xanh; `_report_phase5.py` tái tạo digest thế hệ 1; `registry_counts` bất động; số unrunnable thế hệ 2 chưa đo (đo tại mục 8 bước 3).

**Phụ thuộc**: khối 1 (lịch), 2 (nhãn), 4 (catalog). **Chặn bởi user**: không. **Chi phí**: ~10 lần `register=False` × 2,68 s; report ~3 phút.

**Tiên đoán**: `trades.parquet` với/không `name` **giống hệt**; `_infer_stage` → `signal`; digest thế hệ 1 tái tạo; số unrunnable thế hệ 2 = **0** dưới cả hai họ (menu hồi quy không có L1 logistic; `lasso_f0.85` cho intercept ≠ 0 nên n_unique = 1 nhưng ≠ threshold → chạy được dưới signed, **unrunnable dưới percentile** — nếu một lasso nào triệt tiêu hết hệ số, số đo là 2 + 1 cho mỗi set như thế; viết trước rằng số này được ĐO chứ không tiên đoán bằng 0 tuyệt đối).

---

## 6. Khối 6 — Chi phí + rủi ro

**Mục đích**: `16_costs` và `15_risk_management` tồn tại, guard đúng, sẵn sàng chạy **nếu** D5.3 thế hệ 2 đạt. Guard: **Costs** (D6.2 đơn vị mỗi lượt vs khứ hồi), **Capacity**, **Risk actions** (mỗi arm gắn hành động trong `risk_config.yaml`). Roadmap: Giai đoạn 6, cổng `roadmap.md:193-194`.

**6.1 — `16_costs`**: fork `case_studies/exness_fx_d1/16_costs.py`, port **nguyên văn** `:159-180` (từ chối khi `costs.swap.measured_on_account ∉ {real, live}`; từ chối swap ≠ 0 mà mô hình phần trăm không diễn đạt) cộng nhánh D6.3: cho phép **theo nhãn** khi nhãn ∈ `costs.swap.not_applicable_labels` **đã đo** (0 crossing 00:00 UTC). Cổng D6.2 nguyên: `breakeven_round_trip = 2 × breakeven_per_crossing > 1,5 × round_trip_p90` (XAUUSD > 1,8 bps, XAGUSD > 14,1 bps), nội suy trên đường Sharpe(cost) của **một** carrier, không envelope, không bậc thang; per-symbol bằng dạng đóng từ `trades/fills.parquet` với falsifier mức sổ. `cost_grid_bps` không đổi.

**6.2 — `not_applicable_labels` cho `fwd_ret_sess`**: thế hệ 1 đo `[fwd_ret_8h, dir_tb_8h]` (`BOT.md:105`); nhãn mới **phải đo lại** bằng `tools/presweep_measurements.py::midnight_crossings` (`:242`) trên vị thế `register=False` của `fwd_ret_sess` (hold 8/7 bar, cùng cửa sổ) rồi mới khai `costs.swap.not_applicable_labels: [fwd_ret_sess]` kèm số đếm và ngày. `fwd_ret_24h` (hoãn) không cần.

**6.3 — `15_risk_management`**: fork từ `exness_fx_d1`, 6 arm đã pre-check (`BOT.md:105` D6.4: 1.000 / 0.495 / 0.135 / 0.036 / 0.0194 / 0.0019, không arm nào suy biến); **đo lại** activation trên vị thế `fwd_ret_sess` (`presweep_measurements.py::arm_activation`, `:277`) — cùng luật D6.4: chỉ loại arm activation **= 0 trên cả hai sổ**, không bao giờ theo Sharpe. Luật đã tiền đăng ký: nếu XAGUSD rời universe ở D6.2(d), `stop_loss_5pct` suy biến (0 trên XAUUSD cả hai sổ) và bị loại **lúc đó**.

**6.4 — `14_portfolio_management`: vẫn ĐÓNG** (D6.1, ba lý do cơ học `allocation.py:22-46`, `14:192-196, :386`). **Per-asset vol scaling: KHÔNG xây ở thế hệ 2** — là sửa thư viện dùng chung cộng một họ trial mới (D6.1); nếu xây ở thế hệ sau: `K += |labels| × top_n_predictions.allocation(10) × |books| + pooled`, khai trước.

**6.5 — Công cụ user chạy trên Pro thật (read-only)**: `bots/exness_gold_sess/tools/measure_metals_costs.py` (đã có; `:167` hiện **từ chối** `trade_mode != 0` — tức từ chối tài khoản thật; builder thêm cờ `--allow-real-account` chỉ đổi guard từ "phải demo" thành "in trade_mode và login, không `order_send`", ghi vào `spreads_by_session` **nối thêm** với `measured_on_account: real`, và `read_swaps` (`costs_mt5.py:410`) → `setup.yaml::costs.swap.points_per_lot_per_night` + `measured_on_account: real` + ngày). Đây là bước duy nhất `fwd_ret_24h` chờ.

**File tạo/sửa**: `16_costs.py`, `15_risk_management.py` (mới, fork; `tests/overrides.yaml` thêm `skip: true` như fx_d1 `BOT.md:149`); `tools/presweep_measurements.py` tham số nhãn; `tools/measure_metals_costs.py` cờ; `deploy/risk_config.yaml`: mỗi arm sống sót một dòng hành động khi live (D6.4 câu cuối) — hiện đã có 10 breaker theo kill criteria, thêm mục `position_rules_live: {stop_loss_*, trailing_*, hold_2h}` mapping sang lệnh MT5 (SL/TS server-side hay client-side) — tài liệu trước, mã sau.

**`setup.yaml`**: `costs.swap.not_applicable_labels: [fwd_ret_sess]` + `measured_crossings: {fwd_ret_sess: {london: 0/N, ny: 0/N}, measured_on: ...}` — **chỉ sau khi đo**; mọi thứ khác trong `costs`, `backtest.sweep.cost_grid_bps`, `breakeven_gate`, `risk_controls` **không đổi**.

**Digest dịch**: không (khối này không đụng nhãn/feature).

**Trial accounting**: `K_risk = |arms sống sót| × |labels| × |books| × top_n.risk_overlay(1) + |arms| × |labels| × 1 (pooled) = 6 × 1 × 2 × 1 + 6 × 1 = 18` (nếu 6 arm sống, Giai đoạn 6 mở). Ghi `K_final = 3.582 + 18 = 3.600` vào Trials **trước** khi `15` chạy. Biến thể chi phí: **không** là trial (D6.4). Nếu D5.3 không đạt: +0, `14`-`16` không chạy.

**Tests** (`tests/test_backtest_costs.py` mở rộng; `tests/test_stage16_guard.py` mới)
- `::test_16_costs_refuses_a_demo_swap_for_a_label_not_in_not_applicable_labels` — import module với `measured_on_account: demo`, nhãn ngoài danh sách → `RuntimeError` nguyên văn `:161-168`.
- `::test_16_costs_admits_only_measured_not_applicable_labels` — khai không có `measured_crossings` → từ chối.
- `::test_breakeven_units_are_round_trip_not_per_crossing` — D6.2(a): 2× trước khi so 1,5× p90.
- `::test_midnight_crossings_are_zero_on_fwd_ret_sess` — `register=False`, cả hai sổ.
- `::test_no_arm_is_dropped_on_a_sharpe` — `arm_activation` không đọc cột return (kiểm import/cột).
- Test cũ `test_exness_fx_d1_costs_did_not_move` vẫn môi trường (không sửa ở đây, `BOT.md:797`).

**Definition of done**: hai stage fork, `.ipynb` sync 0 drift, **không chạy**; tests xanh; 0 hàng registry.

**Phụ thuộc**: khối 2 (nhãn cho đo lại), khối 5 (`_hold`, seam). **Chặn bởi user**: đọc Pro thật (chỉ chặn `fwd_ret_24h`; `fwd_ret_sess` đi bằng đo 6.2). Builder làm được: toàn bộ. **Chi phí**: đo lại D6.3/D6.4: 4 lần `register=False` ≈ 1 phút; `16_costs` khi chạy: 11 điểm × 2 carrier ≈ 22 × 2,68 s ≈ 1 phút; `15`: 12 backtest ≈ 1 phút.

**Tiên đoán**: crossing 00:00 UTC trên `fwd_ret_sess` = **0/N cả hai sổ** (cấu tạo: 09-17 và 13/14-20/21 UTC, cùng lập luận `BOT.md:105`); activation D6.4 trong ±0,02 của thế hệ 1 (vị thế gần trùng); `stop_loss_5pct` = 0 trên XAUUSD cả hai sổ.

---

## 7. Khối 7 — Deploy + monitor

**Mục đích**: làm cho bước 5-7 của `deploy/deployment_loop.py` **chạy được như harness** không cần carrier, breaker (g) đọc lịch khối 1, `--arm` vẫn từ chối. Guard: **Parity** (`25_live_trading/08_pipeline_verification`), **Safety** (`SafeBroker`, shadow, reconciliation), **Decay** (Ch26). Roadmap: Giai đoạn 8-9, cổng `roadmap.md:237-238, :255-256`.

**7.1 — Harness parity với "mô hình" là một prediction set ĐÃ ĐĂNG KÝ** (mẫu `exness_fx_d1` `--model none`, `bots/exness_fx_d1/BOT.md:152`): `MODEL_SELECTOR = "replay:<prediction_hash>"` đọc score đã đăng ký (read-only, đã đếm trong K) tại các decision instant validation ≤ 2025-08-28, đi qua **cùng** đường live: `features_as_of` (`_features.py:1094`) → `build_targets` (`deployment_loop.py:381`) → `floor_to_placeable` (`:401`) → `replay_parity` (`:455`) qua `ml4t.backtest.Engine` với **đúng spec** thế hệ 2 (signed, `risk.name: baseline`, hold dẫn xuất) → so với `weights.parquet` của `backtest_hash` tương ứng. Bước 3 `retrain` (`:323`) **vẫn stub** (không có carrier; không "mô hình giả" — `:39-40`); bước 4 `persist` ghi provenance replay. `assert_no_holdout` trên mọi frame (`_hold.py:86`). Docstring `:462-474` của `replay_parity` đã lỗi thời (nói price grid chưa landing) — sửa đúng sự thật.

**7.2 — Breaker (g)** (`monitor/circuit_breakers.py:353-400`): thêm đầu vào `declared_closures` (file khối 1) và `session_calendar` (giờ nghỉ theo mùa, `BOT.md:74-76`): kêu chỉ khi `held_symbols` ∩ {symbol có gap qua **một khoảng đóng KHÔNG có trong lịch và không phải giờ nghỉ hằng ngày**}; **im** ở 21:00-22:00 / 22:00-23:00 UTC (hai sổ 8h không giữ qua đó, đo 0 vị thế). `monitor_config.yaml` thêm `overnight_gap.declared_closures_path`.

**7.3 — `--arm`**: `deployment_loop.py:494-495`, `:799` từ chối khi `breakers.pending_user_approval: true` (`risk_config.yaml:261`) — **không đổi**; `build_manager()` (`circuit_breakers.py:463-468`) từ chối — **không đổi**. Test khẳng định cả hai vẫn từ chối.

**7.4 — Test bằng `FakeMT5`** (`bots/_shared/testing/fake_mt5.py`): preflight, reconcile (magic 260903), contract geometry (`:529`), floor (73 oz XAG → 50; 40 → rejected), replay parity, breaker (g).

**File tạo/sửa**: `deploy/deployment_loop.py` (7.1), `monitor/circuit_breakers.py` + `monitor_config.yaml` (7.2), `tests/test_parity.py` (mới — `bot-template.md:90` yêu cầu, hiện chưa có), `tests/test_breakers_gold.py` (mới).

**`setup.yaml`**: không đổi. **Digest dịch**: không. **Trial accounting**: **+0** (replay của identity đã đếm; không chọn gì).

**Tests**
- `test_parity.py::test_replay_of_a_registered_validation_set_reproduces_the_registered_weights` — max|diff| = 0 trên mọi decision instant của một quý validation, cả hai sleeve.
- `::test_the_live_row_equals_the_batch_row_at_eight_instants` — tái dùng 8 instant của `test_lookahead.py:212`.
- `::test_lot_flooring_is_recorded_not_hidden` — 73 → 50 dropped 23; 40 → rejected.
- `::test_arm_is_refused_while_criteria_are_draft` và `::test_build_manager_refuses_draft_criteria`.
- `test_breakers_gold.py::test_g_is_silent_on_the_seasonal_daily_break_and_on_every_declared_closure` — 0 alarm trên toàn cửa sổ validation với tape sổ 8h.
- `::test_g_fires_exactly_once_on_the_christmas_2024_carry_when_the_closure_is_undeclared_and_never_when_declared` — fixture từ vị thế `BOT.md:943` (XAUUSD 2024-12-24 15:00 → 12-26 18:00); có 2024-12-25 trong lịch → 0; không có → 1.
- `::test_no_account_breaker_is_duplicated_here` (mẫu fx_d1 `BOT.md:153`).

**Definition of done**: tests xanh trên `FakeMT5`; `--preflight` smoke read-only trên demo không đổi kết quả (`BOT.md:107`); 0 hàng registry.

**Phụ thuộc**: khối 1 (lịch), khối 5 (spec thế hệ 2 cho replay). **Chặn bởi user**: duyệt Hypothesis/Kill criteria/DSR convention (D7.1 mục 2) — mở `--arm` và `build_manager`; xác nhận `start_bot.bat`/`watchdog.bat` không tự khởi động v9 (`BOT.md:919`); tài khoản demo riêng cho paper (fx_d1 open question 31: legacy magic 202500 còn vị thế trên login này). Builder làm được: toàn bộ 7.1-7.4. **Chi phí**: giây.

**Tiên đoán**: replay trùng `weights.parquet` **0 sai lệch** (cùng code path; nếu lệch, nguyên nhân đầu tiên nghi là `floor_to_placeable` 50 oz trên XAG — engine dùng `share_type: integer` không biết grain 50, `setup.yaml:128-131` — và khi đó test phải **in** lệch chứ không nới); breaker (g): 0 alarm trên tape 8h; đúng 1 trên fixture Christmas khi lịch thiếu.

---

## 8. Khối hoàn chỉnh (ghép)

**Điều kiện ghép**: 7 khối "xong" theo L0.6; `pytest bots/exness_gold_sess` không `ML4T_OUTPUT_DIR` 0 failed, với `ML4T_OUTPUT_DIR` chỉ còn failure môi trường `test_exness_fx_d1_costs_did_not_move`; `registry_counts` = (216 / 801 / 1.674 / 24) **chưa đổi**; hàng Trials "K2 = 1.068, 0 chi" đã ghi; bản này lưu ở `bots/exness_gold_sess/GEN2_DECLARATION_2026-09-10.md` và `BOT.md` Phase status trỏ tới nó.

**Thứ tự chạy cố định, một lần, không staging** (D5.1 nguyên):
1. `02` → `04` (một lần, khẳng định `881f249635f32deb`) → `05` (khối 2 đã làm; nếu làm lại thì digest phải tái tạo).
2. `06` → `07` với `SUPERSEDES_POPULATION` = tip đọc từ registry; `12` xanh trên menu đầy đủ. Ghi: 43 / 178 / 172. **Chưa đọc IC nào trước khi hàng Trials "fit xong, K2 không đổi" được ghi** (IC không phải cổng; `RULINGS §3` lý do 1: IC bằng 0 cũng là một kết quả, không dừng).
3. `13_backtest` cell plan-time: đo số unrunnable thế hệ 2 → ghi vào BOT.md → `UNRUNNABLE_*_EXPECTED` cập nhật → freeze population generation 2 (`SUPERSEDES_SESSION_BOOK_BASELINES=2f37b3b51a42`) → `RUN_SWEEP=True`, `LABEL=""`, `SESSION_BOOKS=[]`, `TOP_N_PREDICTIONS=None`, `POPULATION_NAME=""`, `EXECUTION_TIER="canonical"`, `SPLIT="validation"`, `FORCE_REBACKTEST=False`, `SEED=42`. Vòng lặp `for label in sorted(...)` (một nhãn) → `[london, ny]` → spec theo thứ tự file. Không `except-and-continue`. Dừng chỉ khi `require_complete()` xanh (`13_backtest.py:1105`).
4. `_report_phase5.py` read-only, print order D5.3: (i) K = 3.582, `member_digest_gen1` (phải = `765830e9…`), `member_digest_total`, thành viên theo thế hệ/nhãn/sổ; (ii) `variance_trials` raw/active và `expected_max_sharpe` dưới H0 tại K = 3.582; (iii) rồi mới phân phối Sharpe và tên spec.

**Chữ ký nghiệm thu D5.3 thế hệ 2, viết sẵn — builder chép nguyên văn**:
- **ĐẠT**: *"Tại K = 3.582 = 2.514 + 1.068 (`member_digest_total <digest>`; thế hệ 1 `765830e9fe95a8e06167ca89317282c02e6f9dcaf4e00685bf1bdcc5817e287b` tái tạo), spec `<hash>` (thế hệ `<1|2>`) có DSR trên **active return** so với sổ 1/N long hai kim loại = `<p>` >= 0,95 dưới `<dsr_notebook|dsr_library>` (giá trị kia: `<p'>`). Giả thuyết 'chưa có lợi thế nào được phân giải' bị mẫu này bác bỏ ở mức đã khai. Đây là một kết quả trên split validation; nó không phải một kỳ vọng lợi nhuận và không phải một quyết định triển khai."*
- **KHÔNG ĐẠT**: *"Tại K = 3.582 = 2.514 + 1.068 (`member_digest_total <digest>`; thế hệ 1 `765830e9…` tái tạo), `<n>` của 3.582 spec có DSR >= 0,95 trên active return. Với n = 0: không spec nào phân giải được một lợi thế trên mẫu này ở số trial đã chi. Giai đoạn 6 KHÔNG mở, `14`-`16` không chạy, holdout không được đọc. Tiền lệ cộng dồn: `exness_fx_d1` 3.204, bot này 3.582, 0 spec sống sót."*
- **ĐẠT RAW, KHÔNG ĐẠT ACTIVE**: nguyên văn `PHASE567_DECLARATION.md:127` với K = 3.582, kèm ba bằng chứng cơ chế (tương quan với benchmark, tỷ lệ `net_exposure > 0`, cổng phơi nhiễm `|net|/gross > 0,5`).

Kỳ vọng ghi trước (không phải mong muốn): nhánh **KHÔNG ĐẠT** — IC tối thiểu phát hiện được vẫn 0,054 và không có chiều thông tin mới trong matrix thế hệ 2; thế hệ 2 sửa **parity**, không thêm **tín hiệu**. Nếu ĐẠT, câu hỏi đầu tiên là cổng phơi nhiễm (X.1(3)).

**Checklist D7.1 cập nhật** (11 mục, `PHASE567_DECLARATION.md:274-288`): mục 1 → "≥ 1 spec DSR ≥ 0,95 active tại **K = 3.582**, `member_digest_total` trong BOT.md"; mục 3 → "dưới `not_applicable_labels: [fwd_ret_sess]` **đã đo**"; mục 5 → `CandidateSet exness_gold_sess:holdout-candidates` từ `15`; mục 6 → carrier thuộc engine book, **hai carrier, một mỗi sổ** (D5.4); mục 10 → buffer huấn luyện của `17` = buffer rộng nhất **đang khai** (`variants: []` → `buffer: 1D`; `variant_buffers` trơ — ghi rõ); mục 11 → thêm `test_labels_gen2.py`, `test_calendar.py`, `test_parity.py`, `test_breakers_gold.py`. Mục 2 (user duyệt), 7, 8, 9 nguyên văn.

**Vẫn cần user, không builder nào thay được**: (a) duyệt Hypothesis, Kill criteria (kể cả (g) tinh chỉnh), quy ước DSR (`BOT.md:915-916`) — chốt chặn sống lên Giai đoạn 7 và `--arm`; (b) đọc Pro thật read-only (`BOT.md:917`) — mở `fwd_ret_24h`; (c) `FRED_API_KEY` (`BOT.md:918`) — mở hai họ PLANNED; (d) xác nhận scheduler v9 và tài khoản demo cho paper.

---

## 9. Bảng tóm tắt

| Khối | Phụ thuộc | Trial cộng thêm | Digest dịch | Chặn bởi user? | Thời gian ước lượng |
|---|---|---|---|---|---|
| 1 Dữ liệu + lịch | — | +0 | không (`session_panel` bất động) | không | < 5 phút |
| 2 Nhãn `fwd_ret_sess` | 1 | **K2 = +1.068** (24h hoãn +0; dir_tb closed +0; nếu giữ 24h: +1.068; reopen dir_tb sau: +378) | `labels/fwd_ret_sess` mới; `04` `inputs` dịch, `digest 881f249635f32deb` **không** | không | `02` ~3 phút, `04` ~13 phút, `05` ~3 phút |
| 3 Feature | 1, 2 | +0 nay; +1.068/họ nếu bật sau fit đầu | không (`financial 0557461a95579969` bất động) | **có** (`FRED_API_KEY`) — mã/test làm được | `03` ~3 phút; test giây |
| 4 Mô hình | 2, 3 | 0 (fit không là trial); K2 khai ở đây | không (0 fit) | không | resolve giây; fit (mục 8) ~4,5 phút + `12` ~2 phút |
| 5 Backtest/signal | 1, 2, 4 | +0 ngoài K2 | mọi `backtest_hash` thế hệ 2 mới; 1.674 cũ bất động | không | ~10 × 2,68 s; sweep (mục 8) 712 × 2,68 s ≈ **32 phút** (papermill thế hệ 1 đo 6,46 s/run → ~77 phút); report ~3 phút |
| 6 Chi phí + rủi ro | 2, 5 | +18 nếu Giai đoạn 6 mở (K_final 3.600); cost variants 0 | không | **một phần** (Pro thật chỉ chặn `fwd_ret_24h`) | đo lại ~1 phút; `16` ~1 phút; `15` ~1 phút |
| 7 Deploy + monitor | 1, 5 | +0 | không | **có** cho `--arm`/`build_manager` (duyệt tiêu chí); harness làm được | giây (`FakeMT5`) |
| **Ghép** | 1-7 | K tại cổng = **3.582** | — | mục 2 D7.1 | ≈ 1,5-2 giờ CPU tổng |

Không có con số lợi nhuận kỳ vọng nào trong tuyên bố này và không được thêm vào.

---

**Next step** (builder, khối 1): viết `bots/exness_gold_sess/tools/build_closure_calendar.py` đọc `exchange_calendars` qua `utils/cv_splits.py:55-71` và `_features.decision_grid` report `unresolved_dates` (`_features.py:295`), in bảng đối chiếu 18 lỗ Mon-Fri + 34 early-close/kim loại (`data_census_2026-09-08.md` §3) so với lịch, ghi `calendar/closures_metals.parquet` kèm `source_version`, rồi `tests/test_calendar.py` + `test_fold_geometry.py::test_label_key_set_equals_the_decision_grid` — tất cả với `registry_counts` trước == sau.

**Read**: `case_studies/research/population.py:356-411` (`population_supersedes`: ba tình huống clean clone / re-run / refit — cơ chế duy nhất giữ "current" = thế hệ 2 mà không xoá thế hệ 1) cùng `case_studies/exness_fx_d1/_report_phase5.py:86-134` (đi qua mọi generation để K cộng dồn) — Tập 3, `RUN_LOG.md` "identity" (`:247-372`) và Tập 2, Ch16 `12_dsr_validation` ("Correcting a Sharpe ratio for the search that found it", `knowledge-index.md:453`); và `case_studies/utils/signals.py:28-97` (`threshold_convention`, khoá tường minh đã có) — Tập 2, Ch16 §16.2.

---

## Phụ lục: ghi chú mở đầu của mentor (nguyên văn)

Tôi đã đọc đủ nguồn. Ba phát hiện quyết định hình dạng tuyên bố, nêu trước để user thấy vì sao vài ranh giới khối được siết lại:

1. `case_studies/utils/signals.py:33, :86-89, :639` **đã có** khoá tường minh `threshold_convention: "signed"` (do `exness_usidx_sess` thêm vào thư viện). Khối 5 không cần đặt khoá mới `short_threshold`; dùng khoá thư viện đã có.
2. `setup_version` **không được dòng code nào đọc** (grep toàn repo: 0 hit ngoài `PHASE1_SPEC_MENTOR.md:75`) nên nó không vào hash; ranh giới thế hệ do lineage population (`case_studies/research/population.py:356-411`) và bảng Trials mang, không phải con số này.
3. Repo **đã có** nguồn lịch có provenance: `pyproject.toml:149` pin `exchange-calendars>=4.5`, và `utils/cv_splits.py:55-71` đã dùng nó (`FX → CME_FX`) cho thang fold. Khối 1 không cần "gõ tay 90 ngày".

Dưới đây là tuyên bố, viết để lưu nguyên văn.

---

## Tu chính 1, 2026-09-10 — khối 1 đóng hai gap; chữ ký khối 2 nhận `526e4afb5f1679cc`

Phán quyết mentor 2026-09-10, thi hành bởi builder cùng ngày. Không câu nào ở trên bị xoá hay sửa; mục này ghi (i) chỗ tiền đề của khối 1 sai và số đo thay thế, (ii) hai gap được đóng và bằng chứng, (iii) LUẬT đọc lịch cho khối 5 và 7, (iv) các falsifier của khối 2 viết **trước** khi `02` chạy. Registry khi viết: `training_runs` 216 / `prediction_sets` 801 / `backtest_runs` 1.674 / `official_populations` 24 / `official_population_members` 6.851 (đếm bởi tool, trước và sau mỗi lần chạy, bằng nhau).

**T1.1 — Tiền đề sai: `CME_FX` không phải id của `exchange_calendars`.** Mục 1 viết "qua đúng đường `utils/cv_splits.py:55-71` (`_CALENDAR_MAP`)". Id đó (`FX → CME_FX`) là tên *pandas_market_calendars* mà `ml4t.diagnostic.splitters.calendar` mở; `exchange_calendars 4.13.2` raise `InvalidCalendarName`. Tool đã thử hai id cùng nhà theo tiêu chí cơ học **in ra** (`choose_calendar`): `CMES` 2/26 ngày đóng bị tape bác bằng một phiên đủ, `XNYS` 56/80; đóng sớm tape xác nhận trên cả hai kim loại: `CMES` 62/62, `XNYS` 12/18 → **`CMES`**. `setup.yaml::decision.closure_calendar.source_calendar_id: CMES`.

**T1.2 — Nguồn thứ hai của tape: `tape:mon_fri_holes`.** Mục 1 chỉ nêu `unresolved_dates` của `decision_grid`. Một ngày **không có bar nào** không bao giờ vào `unresolved_dates` (không có quyết định để phân giải), nên `2018-02-01` (ngày thứ hai của outage feed) sẽ bị **lọc trong im lặng** thay vì được in. Tool thêm nguồn `tape:mon_fri_holes` (luật §3 của `data_census_2026-09-08.md`); mỗi hàng ghi rõ nguồn.

**T1.3 — Ba tiên đoán của mục 1 SAI, số đo giữ nguyên (L0.7).** (a) "lịch gói tái tạo cả 18 lỗ": cửa sổ dev có **17** lỗ/kim loại (lỗ thứ 18 nằm trong holdout, không đọc); gói khai **16/17**; `2018-02-01` tape-only. (b) "≥ 30/34 early-close NY cửa sổ validation khớp gói": **29/34** cả hai kim loại. (c) "lệch ≤ 4/kim loại, là Christmas Eve": **5/kim loại**, gồm `2021-12-31` + bốn Juneteenth `2022-06-20, 2023-06-19, 2024-06-19, 2025-06-19`; Christmas Eve `2024-12-24` **có** trong `CMES` và khớp. Hai tiên đoán còn lại ĐÚNG (outage 2018-01-31/02-01 tape-only; test tập khoá xanh lần đầu).

**T1.4 — Gap A1 đóng: `close_utc` của `tape:unresolved_dates`.** Trước: `max(bar_close)` của ngày → trên **104/159** hàng là bar 23:00 UTC in **sau** khi Globex mở lại (Juneteenth 2024-06-19 → `2024-06-20 00:00`). Sau: close của bar cuối **tại hoặc trước giờ đóng venue New York** của ngày (`_features._venue_window(d, "ny")[1]`, cùng cửa sổ `session_of` dẫn xuất), chỉ đọc timestamp. Chạy lại `--reconciled-on 2026-09-10`: digest **`fb0833fb21c03673` → `e831fb7f48b49235`**, 421 hàng, các đếm đối chiếu (17/16, 62, 34/29/5, 15/20 tape-only) không đổi. Phân phối giờ UTC của 159 hàng: trước `{0: 104, 15: 2, 17: 3, 18: 11, 19: 18, 20: 2, 21: 19}`, sau `{15: 2, 17: 18, 18: 64, 19: 35, 20: 20, 21: 20}`, null 0. Tiên đoán mentor: "0 hàng ở giờ 0" **ĐÚNG**; "19 hàng thứ Sáu 2019-2020 vẫn 21:00 UTC" **ĐÚNG** (19 thứ Sáu, 2019-11-08 → 2020-01-31, mùa đông = 16:00 New York). **Phát hiện ngoài tiên đoán**: hàng thứ 20 ở 21:00 UTC là **XAGUSD 2017-05-29** (Memorial Day 2017, mùa hè): tape có bar 00–18, 20, 22, 23 UTC — thiếu bar 19:00 nên phiên NY **off-horizon** chứ không đóng sớm, và bar cuối ≤ giờ đóng NY là bar đóng 21:00 = **đúng 17:00 New York**, tức thời điểm giờ nghỉ bắt đầu. Cơ chế: `unresolved_dates` = `early_close` ∪ `off_horizon` (XAGUSD 70 + 14, XAUUSD 71 + 8) và không tách hai loại. Xử lý: **không nới** `test_the_daily_break_is_not_a_closure`; hàng này được **ghim đích danh** (`AT_BREAK_PINNED = {("2017-05-29", "XAGUSD")}`, assert bằng tập, không phải `<=`), nằm ở vùng tiền-fold không fold nào chạm; một hàng thứ hai là phát hiện phải ghi, không phải pin để nới. `test_no_closure_row_reads_a_price` vẫn xanh (tái dựng từ tape nhiễu giá = file trên đĩa).

**T1.5 — Gap A2 đóng: hai test rỗng được ghim nội dung.** `test_every_mon_fri_hole_of_the_census_is_a_declared_closure`: `sum(d in by_package for d in holes) == 16` và `[d for d in holes if d not in by_package] == ["2018-02-01"]`. `test_early_close_rows_of_the_panel_are_declared_or_named`: `len(dates) - len(tape_only) == 62` mỗi kim loại. Cả hai ghi ngày đo 2026-09-10 và guard gói (`PINNED_PACKAGE = {CMES, 4.13.2}` so với sidecar; khác → skip nói rõ "re-measure and re-pin", không nới) theo mẫu `test_calendar.py` kiểm `xc.__version__` với sidecar. Trước khi ghim, hai assertion là tautology vì mọi lỗ/mọi ngày unresolved đều có hàng `tape:*` theo cấu tạo.

**T1.6 — A3: hai hằng số tay trong tool, ghi nhận.** (1) `bars_on >= 12` (`choose_calendar`): "phiên đủ" = ≥ nửa số bar H1 một ngày thường (23–24). Đo 2026-09-10: hai ngày `closed` của `CMES` bị tape bác (2018-12-05, 2025-01-09 — hai ngày quốc tang) có trọn ngày bar, tám ngày còn lại tape chạm (Christmas/New Year 2022-2025, Good Friday 2017/2018) có **đúng 1** bar 00:00 UTC, nên mọi ngưỡng trong 2..22 cho cùng 2 — hằng số không quyết định gì. (2) thứ tự `CANDIDATE_IDS = ("CMES", "XNYS")` chỉ là tie-break; 2 ≠ 56 nên không có hoà. **Juneteenth**: `exchange_calendars 4.13.2` (`.venv` WSL, `exchange_calendar_cmes.py`) không có Juneteenth ở cả `regular_holidays` lẫn `special_closes`; `us_holidays.USJuneteenth` (từ 2022-01-01) chỉ được `exchange_calendar_xnys.py` và `exchange_calendar_xcbf.py` dùng; dist-info không có changelog, không cài gói mới, không sửa `pyproject.toml` → **không xác định được** bản nào (nếu có) thêm cho `CMES`. Ghi: **known undeclared early close: Juneteenth 2022-2025 (tape-only), breaker (g) kêu là thiết kế.** Cấm ghép Juneteenth từ `XNYS`, cấm gõ tay.

**T1.7 — LUẬT đọc lịch (ràng buộc khối 5 và khối 7).**
1. Khối 5 (`signal.drop_declared_early_closes`, seam `apply_session_filter`) và khối 7 (breaker (g)) chỉ đọc **`declared_closures(package_only=True, kinds=["early_close"])`**. Hàng `tape:*` không bao giờ vào bộ lọc hay breaker: `tape_wins_for_history` chỉ dùng cho **đối chiếu và in**.
2. Falsifier khối 5, viết trước: tàn dư sau lọc trên sổ NY, cửa sổ validation 2021-09-01 → 2025-08-28, = **đúng 5 ngày/kim loại = 10 hàng** (`2021-12-31` + bốn Juneteenth), in tên, không lọc.
3. Hai ngày gói khai `closed` mà tape giao dịch đủ (2018-12-05, 2025-01-09) **không được lọc** — `kinds=["early_close"]` loại chúng theo cấu tạo, và không consumer nào đọc `kind == closed` để lọc hàng.
4. Breaker (g) **sẽ kêu** trên Juneteenth kế tiếp (2027-06-18) nếu gói chưa khai; đó là thiết kế, không phải lỗi, và là lý do (g) cần user duyệt trước `--arm`.

**T1.8 — Digest `526e4afb5f1679cc` chuyển sang chữ ký khối 2.** Khối 1 chỉ khẳng định giá trị **đã ghi** trong sidecar `04` (frame `04` băm kéo vào holdout, không dựng lại ở khối 1). Khối 2 phải: `04` chạy lại **đúng một lần**, **in** `inputs`, và assert `inputs["session_panel"] == "526e4afb5f1679cc"` (`tests/test_fold_geometry.py::SEALED_STAGE04_INPUTS`, `tests/test_labels_gen2.py::test_stage_04_reproduced_its_digest_on_the_new_label`) cùng `digest == 881f249635f32deb`.

**T1.9 — Falsifier khối 2, viết trước khi `02` chạy (bổ sung mục 2, không thay).** (a) `labels/fwd_ret_sess.parquet`: 9.810 khoá (= lưới), **9.629 non-null** (= `fwd_ret_8h`: mọi endpoint niêm phong có một exit khớp được tại-hoặc-trước nó, không hàng nào khác), `inputs.market_data == 6d63567e7b83eea2`. (b) Ba digest thế hệ 1 tái tạo byte-identical từ sidecar **và** từ nội dung (`value_digest`). (c) Pearson NY `fwd_ret_sess` vs `fwd_ret_8h` được đo **hai cách** và ghi cả hai: trên mọi hàng NY cửa sổ dev (định nghĩa của `ny_probe`, gồm 102 hàng 2017 nơi exit khớp = endpoint) và trên hàng trong tầm fold (≥ `train_start` sớm nhất, dẫn xuất bằng `generate_cv_splits`, không gõ); tiên đoán 0,9852/0,9886 của mục 2 được đối chiếu với **cả hai**. (d) `04`: `digest 881f249635f32deb`, 24.740 hàng, 9 cột, 5 fold, 20 trường `fold_geometry` không đổi (ghim trong `test_labels_gen2.py`), purge 3–5 slot. (e) Hai sai lệch cơ học so với chữ nghĩa mục 2, ghi trước: luật 2.1 được đặt trong `_hold.tradable_exit` (cùng module với luật 1.2 của `derive_hold_bars`) và `02` import nó — để test và stage không có hai bản cài đặt; `04` nhận **một guard** cho trường hợp `variants: []` (`assert_variant_folds_are_out_of_sample` trả 0 hàng → `pl.DataFrame([])["gap"]` raise), guard không đọc dữ liệu nên không thể chạm digest — nếu digest dịch, guard không phải nguyên nhân và khối 2 DỪNG theo L0.3.

**T1.10 — Kết quả khối 2, đo 2026-09-10 sau khi T1.1–T1.9 được viết: DỪNG tại falsifier của chính khối.** `02` (papermill, 72 s): `labels/fwd_ret_sess.parquet` digest **`225a35341f81c7f0`**, 9.810 khoá, **9.629 non-null** (T1.9(a) ĐÚNG), `market_data 6d63567e7b83eea2` bất động, ba digest thế hệ 1 bất động (sidecar và nội dung, T1.9(b) ĐÚNG), digest tái tạo giữa hai lần chạy. Hình học exit in bởi luật: London 8 bar / gap 0 trên 4.384 hàng dev; NY 7 bar / gap 60 trên 4.129 hàng; NY 8 bar / gap 0 trên 102 hàng chế độ nghỉ 2017 (fold −2). Pearson NY (T1.9(c)): probe-vs-8h tái tạo **đúng 0,9852 / 0,9886**; `fwd_ret_sess`-vs-8h **0,9861 / 0,9891** (mọi hàng dev) và 0,9860 / 0,9893 (tầm fold) → tiên đoán "đúng bốn chữ số" của mục 2 **SAI**, "chính là probe" **ĐÚNG** trên 100 % hàng trong tầm fold (max|diff| 0,0). `04` (papermill, một lần, 275 s): `inputs` in và assert `{labels:fwd_ret_sess: 225a35341f81c7f0, session_panel: 526e4afb5f1679cc}` (T1.8 ĐÚNG); **digest `2af8a7e4025bf870` ≠ `881f249635f32deb` → DỪNG**, `05` không chạy, không số nào bị sửa. Cái đã dịch, đo: 24.744 hàng (+4), fold 4 (vintage holdout) `train_end 2025-08-27 → 2025-08-28` (1/20 trường), purge fold 4 5 → 3; fold 0–3 byte-identical, 9 cột, 5 fold. Cơ chế, đọc trong mã: `case_studies/research/holdout.py::build_holdout_cv` kết thúc huấn luyện fold holdout một **buffer rộng nhất** trước holdout, `widest_label_buffer` đọc `labels.primary + labels.variants`; thế hệ 1 rộng nhất là 2D của `fwd_ret_24h`, `variants: []` đưa về 1D. Mục 8 (D7.1 khoản 10) đã thấy đúng hiệu ứng này cho stage `17` nhưng mục 2 tiên đoán `04` bất động — tiên đoán **SAI** vì cùng cơ chế. Hai lựa chọn chờ phán quyết (không lựa chọn nào được lấy ở đây): (a) chấp nhận `2af8a7e4025bf870` bằng tu chính có ngày (fold validation 0–3 mà mọi hàng thế hệ 1 được chấm không đổi; chỉ vintage giai đoạn 7 dịch một phiên, purge 3 slot ≥ tối thiểu 2 mà `04` assert), ghim lại `STAGE04` trong `test_labels_gen2.py`, rồi chạy `05`; (b) revert bằng cách chạy `04` dưới `setup.yaml.bak_gen2block2_2026-09-10` (tất định), tái tạo `881f249635f32deb` nhưng `inputs` sẽ mang `labels:fwd_ret_8h` trong khi primary là `fwd_ret_sess`. Test: có `ML4T_OUTPUT_DIR` 88 xanh / 2 đỏ (`test_exness_fx_d1_costs_did_not_move` môi trường; `test_stage_04_reproduced_its_digest_on_the_new_label` là falsifier, đỏ theo thiết kế), không env 67 xanh / 23 skip / 0 đỏ. Registry 216 / 801 / 1.674 / 24 / 6.851 trước và sau mọi lần chạy. Trial chi: 0.

---
