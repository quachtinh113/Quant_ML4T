# DECLARATION — price grid và hold cho `exness_gold_sess`

Ngày: 2026-09-08 · Mentor declaration, giải quyết defect B · Repo root `D:/05_Quant/machine-learning-for-trading` (branch `exness-bots`)

Đây là một **declaration**, không phải gợi ý. Builder thực thi từng dòng, không phải phán đoán lại. Bổ sung và **sửa** `PHASE1_SPEC_MENTOR.md` §2.d ở những chỗ ghi rõ.

Vị trí roadmap: cổng **Giai đoạn 5** (`roadmap.md:151-168`), nhưng vật thể phải sửa là artefact của **Giai đoạn 1** (`setup.yaml`, `config/backtest/base.yaml`, `_PRICE_CONFIG`). Vì `run_log/` còn rỗng nên đây vẫn là *authoring*, không phải *revision*.

Guard áp dụng (`mentor-protocol.md:20-33`): Costs (quy ước fill quyết định crossing nào bị tính), Multiple testing (mỗi book / mỗi arm risk là một trial), Parity (vòng lặp live phải thoát đúng 8 bar H1 như backtest), Point-in-time (không được lọc hàng giao dịch theo `label_end_ts`, một giá trị chỉ biết được sau).

**Phán quyết về việc hoãn**: builder hoãn đúng. Nhưng bản đo của builder mới đúng một nửa: chuyển sang grid H1 **không** làm `periods_per_year` thành ~6.084, và `slot_strategy` **không** thực thi được lệnh thoát cho bot này. Cả hai được chứng minh dưới đây.

---

## 1. Price grid

### 1.1 Đăng ký gì

`_PRICE_CONFIG["exness_gold_sess"]` (`case_studies/utils/backtest_loaders.py:728-740`) đổi `loader` sang một nhánh mới trả về **khung H1 thô của hai kim loại, đánh khoá theo BAR CLOSE**:

```python
"exness_gold_sess": {
    "entity_col": "symbol", "time_col": "timestamp", "close_col": "close",
    "ohlcv": True, "loader": "exness_gold_sess_h1", "drop_cols": [],
}
```

Nhánh loader (`backtest_loaders.py:881-902`, thay cho lời gọi `load_session_panel(..., for_backtest=True)`):

- đọc `bots/_shared/mt5_loader.load_mt5_bars("1h", symbols=sorted(universe.symbols), start_date=universe.history_start, end_date=...)`;
- `dt.replace_time_zone(None)` (đúng quy ước panel, xem `bots/exness_gold_sess/tests/test_backtest_grid.py:93`);
- **cộng 60 phút vào `timestamp`** để khoá theo *bar close*, đúng quy ước `session_panel` đang dùng cho `decision_ts` (`_features.py:390-392, 427-431, 447`: `bar_close = timestamp + 60m`, và panel `timestamp = decision_ts` là một **bar close**);
- **không** cắt giờ Á, **không** cắt giờ ngoài phiên: các bar giữa quyết định và điểm thoát chính là thứ stop/trailing đọc. Việc không giao dịch ngoài phiên được bảo đảm bởi lịch rebalance (mục 3), không bởi việc xoá bar.
- `load_session_panel(for_backtest=True)` **giữ nguyên việc raise** (`_features.py:457-509`) — nó trở thành chốt chống ai đó đăng ký lại panel làm grid, chứ không còn nằm trên đường chạy.

`config/backtest/base.yaml` (repo-pinned, `case-studies.md:145-147`): `calendar.data_frequency: session` → **`1h`**. `execution.execution_price: open` và `execution_mode: next_bar` **giữ nguyên** — và chỉ giữ nguyên được vì grid khoá theo bar close: weight tại 09:00 khớp hàng "bar đóng lúc 09:00", `next_bar` fill tại **open của bar đóng lúc 10:00 = giá 09:00**, đúng bằng cột `open` mà `session_panel` đang dùng làm giá fill (`_features.py:410-421`). Nếu khoá theo bar open, cùng cấu hình đó sẽ fill lúc 10:00 — trễ một giờ so với quy ước hiện hành, và mọi con số chi phí sẽ không so sánh được với `exness_fx_d1`.

### 1.2 Predictions ở đâu

**Predictions ở nguyên cadence phiên** — không thêm hàng session-close vào panel. Đây là mẫu nasdaq (`nasdaq100_microstructure/config/setup.yaml:260-263`), và ở đây thậm chí không cần asof: mọi decision instant vốn đã là một bar close H1, tức **tập con** của grid mới.

**Panel không được mọc thêm hàng session-close**, vì ba lý do:

1. Engine chỉ hành động tại `timestamp in rebalance_schedule`, và tập đó dựng từ `predictions["timestamp"]` chứ không từ giá (`backtest_runner.py:1408-1410`, chốt chặn `:1500-1501`). Một hàng 17:00 trong panel mà không có prediction 17:00 thì engine vẫn không đổi được vị thế ở đó. Muốn engine hành động ở 17:00 thì phải có *prediction* ở 17:00 — tức phải có feature row và một điểm số mô hình tại một thời điểm bot **không ra quyết định**. Đó là bịa ra một quyết định.
2. Nó làm dịch mọi digest nhãn và feature vì một lý do không liên quan tới thông tin.
3. Với hai slot có độ dài giữ tự nhiên khác nhau, **thời gian giữ là thuộc tính của vị thế, không phải của khoảng cách lưới**. Trên grid H1 cả London lẫn New York đều giữ đúng 8 bar tính từ điểm vào của chính nó; cơ chế "khoảng cách lưới" cho ra 5 h và 19 h. Khi phase 6 muốn đổi riêng New York (ví dụ thoát trước giờ nghỉ 21:00 UTC) thì đó là đổi một hằng số, không phải đăng ký lại lưới.

---

## 2. Các giá trị, kèm số học

| Khoá | Giá trị mới | Số học / nguồn | Pinned? |
|---|---|---|---|
| `decision.cadence` | **`session`** (giữ nguyên) | Cadence resolve trên **prediction timestamps**, không bao giờ trên giá (`backtest_loaders.py:1408`, `1580-1581`). `"session"` không parse thành cadence intraday (`_intraday_cadence_interval`, `:1323-1336`) nên trả về mọi decision instant — đúng. **Tuyệt đối không đổi sang token giờ**: với `1_hour` + step 2, `_on_the_clock(ts, 2h)` giữ 14:00 (50400 % 7200 = 0) và **loại 09:00** (32400 % 7200 = 3600 ≠ 0) — cả sổ London biến mất trong im lặng. | Không pinned nhưng `case-studies.md:135` buộc đi kèm `rebalance_step` tương thích |
| `decision.cadence_by_label` | **không khai** | Không token nào diễn đạt được "hai slot phiên"; khai token khác chỉ tách schedule và tách hash chứ không sửa gì. `build_backtest_spec` raise nếu có khoá này mà thiếu `label=` (`backtest_presets.py:450-456`). | — |
| `labels.rebalance_step.fwd_ret_8h` | **1** | Hold 8 bar H1 do exit rule kết thúc; schedule tiến 1 slot. | **PINNED — viết lại ngay** |
| `labels.rebalance_step.fwd_ret_24h` | **2 → 1** | **Đây là lỗi thứ hai.** Step đếm slot của schedule mà *lần chạy* resolve, và `13_backtest` lọc theo `session` trước. Sổ `london`: schedule 1 slot/ngày → 24 h = 1 slot; step 2 giao dịch cách ngày (48 h cho nhãn 24 h). Sổ gộp: schedule là [London d1, NY d1, London d2, …] và `gather_every(2)` (`backtest_loaders.py:1581`) giữ index 0,2,4… = **chỉ một venue**; sổ 24 h lặng lẽ thành sổ một phiên. Không giá trị nào của một khoá per-label đúng trên cả ba sổ trừ **1**. | **PINNED — viết lại ngay** |
| `labels.rebalance_step.dir_tb_8h` | **1** | Cùng chân trời với `fwd_ret_8h`. | **PINNED** |
| `evaluation.periods_per_year` | **504 → 252** | **Tiền đề của builder sai.** Engine ghi chuỗi lợi nhuận **theo NGÀY**: `extract_daily_returns_frame(...)` → `to_daily_pnl()` (`backtest_runner.py:1582-1586`, `backtest_loaders.py:2075-2096`); ghi chú `backtest_runner.py:56-63` đo thẳng: "Measured grid frequencies are 252/yr for … nasdaq100_microstructure despite … intraday signals". nasdaq chạy feed 1 phút và khai `periods_per_year: 252` (`nasdaq100_microstructure/config/setup.yaml:600`). Grid H1 **không** đưa con số lên 6.084. Nhưng 504 cũng sai cho chuỗi ngày: nó thổi Sharpe lên √(504/252) = **1,414×**, và `reconcile_periods_per_year` **không bắt được**: lưới ngày 24/5 quan sát ≈ 5 × 52,18 = **260,9/năm**, tỷ lệ 261/504 = 0,518 ≥ 0,5 → giữ nguyên giá trị khai (`backtest_runner.py:142-143`). 252 khớp `exness_fx_d1` (`setup.yaml:350`), cùng lịch FX, cùng MTM ngày; 261/252 = 1,036 nên reconciler giữ nguyên. **Không per-label**: annualisation là thuộc tính của lưới lợi nhuận (ngày), không phải của tần suất giao dịch — sổ 24 h vẫn có một hàng mỗi ngày làm việc. | Không pinned (không vào hash) — **nhưng phải đúng trước backtest đầu tiên**, vì nó viết lại mọi metric đã đăng ký mà không đổi hash |
| `decision.slots_per_year` | **504** (khoá MỚI) | 2 snapshot × 252 phiên = 504. Đây là nơi 504 thuộc về: thống kê mức slot (N, N_eff, IC, sàn dữ liệu huấn luyện). `04_model_based_features.py:116` phải đọc khoá này thay vì `evaluation.periods_per_year` — giá trị **không đổi**, nên digest 04 không dịch (mục 5). Sổ 24 h: 252 × 4/5 × 2 = **403,2 slot/năm** — số của bản kiểm đếm giao dịch, không phải số annualisation. | Không pinned |
| `execution.allocator_lookback` | **126 → 1512** | Allocator đọc **khung giá** (`backtest_runner.py:2187-2265`, `_apply_allocation(prices=…)`), nay là H1. 63 ngày giao dịch × 24 bar = **1.512**, đúng tiền lệ `xau_fx_mt5/config/setup.yaml:73`. Mật độ đo được 115-120 bar/tuần ≈ 23,5/ngày, nên 1.512 bar ≈ 64 ngày — lệch dưới một ngày so với 63 ý định. Phụ chú: `_calendar_days_per_period` không biết token `session` nên rơi về 1,5 ngày/bar (`backtest_loaders.py:1237-1239`) → prefix warmup ceil(1512 × 1,5) = 2.268 ngày. Đó là **đọc thừa**, không bao giờ đọc thiếu; **không** sửa bảng dùng chung `_CADENCE_CALENDAR_DAYS_PER_PERIOD` bây giờ (nó thuộc các case study khác, không được `run_log/` rỗng của bot này bảo vệ). | Vào spec/hash qua `research/strategy.py:304-306` → đổi sau chỉ tạo identity mới |
| `backtest.price_grid_expresses_primary_label` | `false` → **`true`** | Sau khi test mục 6 xanh, không trước. | — |
| `backtest.price_grid_fix` | → **`register_h1_close_keyed_grid_and_hold_with_position_time_exit`** | Sửa chính chỉ dẫn cũ ở `PHASE1_SPEC_MENTOR.md` §2.d. Lý do ở mục 3. | — |
| `backtest.sweep.session_books` | `[london, ny, both]` → **`[london, ny]`** + khoá mới `pooled_book: {method: sleeve_sum, weights: {london: 0.5, ny: 0.5}, produced_in: 19_strategy_analysis}` | Lý do ở mục 3.4. | — |
| `backtest.sweep.risk_controls.position` | bỏ `time_exit_2`, `time_exit_10`; thêm `{name: hold_2h, type: time_exit, bars: 2}` | Đơn vị `bars` nay là bar H1. `time_exit_10` (10 bar) **không bao giờ kích hoạt** vì hold nền cắt ở 8 — một trial vứt đi mà DSR vẫn tính. Năm arm stop/trailing giữ nguyên (ngưỡng %, không phụ thuộc lưới). | — |

**Phải viết lại ngay khi `run_log/` rỗng**: `labels.rebalance_step` (cả ba), `config/backtest/base.yaml::calendar.data_frequency`, và `_PRICE_CONFIG` + nhánh loader (là code, nhưng `case-studies.md:145-147` cảnh báo đúng trường hợp này: nó đổi *cái được nạp* mà không đổi *cái được hash*).

**An toàn đổi sau** (vào hash → chỉ sinh identity mới): `allocator_lookback`, `session_books`, `risk_controls`.

**Không vào hash nhưng phải đúng trước lần đăng ký đầu tiên**: `evaluation.periods_per_year`, `decision.slots_per_year`.

---

## 3. Cách 8 giờ được thực thi

### 3.1 Tại sao KHÔNG phải `slot_strategy`

Cơ chế giữ trong `case_studies/utils/slot_strategy.py` là chân max-hold của bộ mô phỏng: `if i - slot["entry_i"] >= hold_bars: to_close.append((sym, "maxhold"))` (`:106-107`), đếm **hàng của lưới mà bộ mô phỏng đi**, mà lưới đó là **giá** (`_align_predictions_to_bars(predictions, bar_grid=prices…)`, `:195-205`, `:343`). Trên giấy đúng. Trên đường chạy thì không, vì ba lý do đo được:

1. **Lệnh thoát không bao giờ được thực thi.** `slot_strategy` diễn đạt thoát bằng cách *ngừng phát* hàng weight; engine chỉ hành động tại `timestamp in rebalance_schedule`, dựng từ `predictions["timestamp"]` (`backtest_runner.py:1408-1410`, `1500-1501`). nasdaq thoát được vì panel prediction của nó dày 15 phút nên **mọi thời điểm thoát cũng là một decision instant**; panel của bot này không có hàng nào lúc 17:00 hay 22:00.
2. Cơ chế slot **từ chối long_short** (`slot_strategy.py:277-281`) trong khi `mapping.position_state_space: long_short`.
3. Trên hai tài sản, `max_slots` suy biến — đúng lý do `equal_weight_top_k` đã bị loại (`setup.yaml:276-280`).

### 3.2 Cơ chế đúng: position rule mức broker

`TimeExit(max_bars=N)` là **rule của broker**, đặt một lần ở bar đầu (`broker.set_position_rules(position_rules)`, `backtest_runner.py:1473-1476`), đánh giá trên `bars_held` của vị thế (`19_risk_management/10_ml4t_backtest_risk_demo.py:137,178-186`) và **không đi qua chốt lịch rebalance**. Đó là cơ chế duy nhất trong repo này đóng được một vị thế tại một thời điểm không phải decision instant.

Builder phải truyền, **trong MỌI spec** của `13_backtest` (kể cả baseline Ch16, không riêng arm Ch19):

```python
risk={"position_rules": [{"type": "time_exit", "bars": HOLD_BARS}]}
```

qua `Strategy(risk=…)` (`case_studies/research/strategy.py:222`) → `build_backtest_spec(risk=…)` → `_build_position_rules` (`backtest_runner.py:2426-2450`). Vì khối `risk` nằm trong spec, hold đi vào `backtest_hash`: bản ghi nói rõ sổ này giữ 8 bar, chứ không phải một mặc định ngầm. Các arm Ch19 chồng thêm stop/trailing qua `RuleChain` (`:2450`).

**Một phép đo, một tiêu chí đạt, không phải một phán đoán**: viết `HOLD_BARS = 8` (= `labels.horizons.fwd_ret_8h` = 480 phút / 60). Test ở mục 6 đo hold thực trên fills. Nếu hold thực đo được là 9 bar (off-by-one của `bars_held` tính từ bar mở vị thế), sửa hằng số thành 7, **ghi con số đo được và lý do vào BOT.md Decisions log**, chạy lại test. Tiêu chí đạt duy nhất: *hold fill-to-fill = 8 bar H1 = 480 phút*, khớp `labels.horizons`.

### 3.3 Ngày endpoint null (early close) và Thứ Sáu của nhãn 24 h

**Endpoint null (79-93 phiên/kim loại)**: **giao dịch bình thường, KHÔNG lọc.** Lọc hàng giao dịch theo `label_end_ts` là point-in-time violation (`mentor-protocol.md:24`): lúc 14:00 ngày 3/7 bot không biết tape sẽ tắt lúc 18:00. `TimeExit` đếm **bar**, nên khi tape đóng sớm, bar thứ 8 sau điểm vào là bar sáng hôm sau và vị thế đi qua khoảng đóng cửa — đúng cái sẽ xảy ra thật. `13_backtest` phải **in ra** tập này (số hàng, ngày, hold thực theo giờ đồng hồ) và test mục 6 miễn trừ đúng tập đó khi khẳng định "8 giờ", trong khi vẫn khẳng định "8 bar" trên **100 %** vị thế.

**Thứ Sáu, `fwd_ret_24h`**: đây là **luật ngày trong tuần**, biết trước hàng tuần, không phải giá trị đo được — nên lọc là hợp lệ. `13_backtest` lọc hàng prediction của sổ 24 h theo `dow != Friday` **trước** khi dựng weight, và khẳng định tập bị lọc **trùng khít** tập nhãn null (đã đo: Mon 0,4 % / Tue 0,8 % / Wed 0,6 % / Thu 2,6 % / **Fri 100 %**, 0 bất đồng trên 4.903 instant). Sổ giữ **tiền mặt** ngày thứ Sáu; không được để weight thứ Năm trôi qua cuối tuần trả ba đêm swap. Điều này **không** đổi `periods_per_year` và **không** là một trial: nó được khai trước sweep, từ chính luật của nhãn.

### 3.4 Sổ gộp `both` — tại sao không chạy được nguyên bản

Broker giữ **một vị thế ròng cho mỗi symbol**. Trên sổ gộp, lệnh New York lúc 14:00 rơi vào giữa lần giữ của London (còn 3 giờ). Vị thế bị điều chỉnh chứ không mở mới, nên `bars_held` vẫn đếm từ điểm vào London: `TimeExit` sẽ đóng **toàn bộ** vị thế lúc 17:00, giết chân New York sau 3 giờ thay vì 8. Không cơ chế nào trong repo diễn đạt được hai lần giữ chồng nhau của cùng một symbol trên một tài khoản ròng — kể cả `slot_strategy` (`slot_strategy.py:272-275`).

Vậy: `both` trở thành **tổng hai tay áo**, mỗi tay 50 % vốn, dựng trong `19_strategy_analysis` từ hai chuỗi lợi nhuận đã đăng ký. Nó **chính xác** (mỗi tay giữ đúng 8 bar), **bảo thủ về chi phí** (khi hai tay ngược chiều, tổng trả hai spread trong khi sổ ròng sẽ tự triệt tiêu — sai lệch về phía đắt hơn, đúng hướng cho một bot đang xét cổng breakeven), và **giữ nguyên `execution_delay: next_bar_open`**.

**Cảnh báo metric**: exit do rule không đi qua chuỗi weight, nên cột `avg_turnover` của registry (tính từ weight, `backtest_runner.py:1610-1620`) sẽ **báo thiếu khoảng một nửa**. Turnover trung thực đọc từ `fills.parquet` / `trades.parquet` (`case_studies/utils/registry/registration.py:1378-1383`). `16_costs` không bị ảnh hưởng vì nó chạy lại backtest trên `cost_grid_bps` và engine tính phí trên fill thật.

---

## 4. Ba nhãn, một grid, một stage

**Một price grid H1 duy nhất, một stage `13_backtest` duy nhất, không có `cadence_by_label`.** Ba nhãn khác nhau ở đúng ba chỗ, tất cả đã khai: `HOLD_BARS`, bộ lọc thứ Sáu (chỉ 24 h), và `labels.buffer / variant_buffers` (không đổi).

Với `fwd_ret_24h`, hold không phải 8 bar mà là "đến decision cùng venue ngày hôm sau" ≈ 24 giờ = **24 bar H1** trên grid mới, nhưng số bar giữa hai decision cùng venue **không hằng số** (DST làm nó là 23 hoặc 25 bar bốn lần một năm, ngày lễ làm lệch thêm). Nên với nhãn này `HOLD_BARS = 24` là **backstop**, còn điểm thoát thực tế là decision kế tiếp cùng venue, nơi weight mới được ghi đè hợp lệ (đó là hàng có prediction, nên engine hành động được). Test mục 6 khẳng định hold của sổ 24 h nằm trong `[23, 25]` bar và bằng đúng khoảng cách decision-tới-decision cùng venue, chứ không khẳng định 24 cứng.

**Trial count: không đổi.** K vẫn là con số đã khai trước, `K = |books| × Σ_labels(|prediction sets|) × |signal specs|`.

> **SỬA 2026-09-08 (builder, theo `FOLD_GEOMETRY_DECLARATION.md` §3, mục này xác nhận phép đo của builder).** Con số `≈ 3 × 3 × 178 × 2 = 3.204` viết trong bản gốc của đoạn này **SAI** — nó giả định 178 prediction set trên *mọi* nhãn. Menu phân loại là **13 logistic + 5 multiclass**, không phải 28 + 15. Số học đúng, đo bằng cách nạp menu chứ không bằng cách đọc:
>
> ```
> fwd_ret_8h  = 28 linear   + 15 GBM × 10 checkpoint = 28 + 150 = 178
> fwd_ret_24h = 28          + 15      × 10           = 178
> dir_tb_8h   = 13 logistic +  5      × 10           = 13 +  50 =  63
> tổng = 419
> K = 3 books × 419 × 2 signal specs = 2.514   (1.068 | 1.068 | 378)
> ```
>
> **K = 2.514**, không phải 3.204. Chênh 690 là menu phân loại, **không** phải menu bị cắt: mọi preset đã khai vẫn được fit. Con số ~3.204 ở mục 5 ("sweep có ~3.204 backtest") đọc là **2.514** vì cùng lý do; ước lượng thời gian engine ~1,9 h đã được tính lại trên 2.514 và ghi trong `BOT.md` phase-5.

Sổ gộp vẫn là một spec cho mỗi (label × prediction × signal); nó chỉ được *sản xuất* bằng tổ hợp tất định 50/50 đã khai trước thay vì một lần chạy engine. **Nếu** trọng số tay áo từng được quét (0,3/0,7…), mỗi giá trị là một trial mới và K tăng. Khai `cadence_by_label` sẽ **không** tăng K nhưng sẽ tách schedule và tách hash — chi phí không đổi lấy lợi ích bằng không.

---

## 5. Chạy lại gì, và tuyệt đối không chạy lại gì

**Digest nhãn KHÔNG dịch. Không chạy lại `02_labels`.** `02` dựng từ `session_panel` và `load_mt5_bars`, không đọc `_PRICE_CONFIG`, không đọc `config/backtest/base.yaml`, không đọc khoá nào bị viết lại ở mục 2. `fwd_ret_8h d6487f7333a683cc`, `dir_tb_8h 1eb71e01111c53b3`, `fwd_ret_24h f2861a79ee3c6cc2`, `market_data 6d63567e7b83eea2` giữ nguyên. Nếu vì lý do nào đó `02` được chạy lại, cả bốn phải tái tạo **byte-identical**; một digest dịch = dừng lại và báo cáo.

> **GHI CHÚ 2026-09-08 (builder) — đoạn trên VẪN ĐÚNG NGUYÊN VĂN, và đây là lý do.** `FOLD_GEOMETRY_DECLARATION.md` §2 **cho phép** ba digest nhãn dịch (A′: publish mọi nhãn trên tập khoá của lưới quyết định, `null` ở nơi nhãn chưa xác định), nhưng buộc **đo điều kiện rẽ nhánh trước**: tập ngày duy nhất của parquet nhãn chính sau khi độn phải **bằng** tập hiện tại, vì `04:176-182` suy thang fold từ đúng tập đó. **Đã đo, và nó KHÁC**: 2.453 ngày so với 2.451. Hai ngày thừa là **2018-01-31** (tape tắt sau bar 14:00, và **2018-02-01 không có một bar H1 nào trên cả hai kim loại** — gián đoạn feed hai ngày) và **2018-09-03** (Labor Day Mỹ, thiếu bar 10:00 UTC nên cả hai venue off-horizon). Tách riêng: 2018-01-31 vô hại; **một mình 2018-09-03** đẩy `train_start` của fold 3 từ **2018-08-30 sang 2018-08-31**, fold 4 thừa hưởng. Nên A′ **không được thực hiện** (đúng nhánh DỪNG của chính tuyên bố đó): `02_labels.py:693` không đổi, `02`/`03`/`04`/`05` không chạy lại, **bốn digest trên đứng nguyên**, và `04` vẫn `4b1a0239330a0e7d`. Ai đọc đoạn này sau: bốn digest ở trên là giá trị **hiện hành**, không phải giá trị lịch sử. Chi tiết và điều kiện gỡ chặn nằm ở `BOT.md` (Decisions log 2026-09-08 và Open questions).

> **CẬP NHẬT 2026-09-08 (builder) — GHI CHÚ Ở TRÊN GIỜ LÀ LỊCH SỬ. Ngoại lệ của §5 được cấp LẦN THỨ HAI, có văn bản:
> `FOLD_GEOMETRY_DECLARATION.md` → "TUYÊN BỐ THỨ HAI". A′ ĐÃ ĐƯỢC THỰC HIỆN.** Điều kiện rẽ nhánh vẫn đo FALSE
> (2.453 ngày so với 2.451, đo lại read-only trước khi động vào gì), nhưng chuyển động đã được **cô lập tới một ngày,
> một biên, hai trường** bằng thí nghiệm bốn timeline, và luật "digest dịch → revert" được viết để bắt chuyển động
> *không giải thích được*. Mentor mở lại quyết định và chấp nhận cú dịch. **Bốn digest ở đoạn trên là GIÁ TRỊ LỊCH SỬ
> kể từ dòng này.** Giá trị hiện hành, đo sau khi chạy `02` một lần và `04` đúng một lần:
>
> | Artifact | Trước | Sau |
> |---|---|---|
> | `labels/fwd_ret_8h.parquet` | `d6487f7333a683cc` | **`9fbf73e65699693c`** |
> | `labels/fwd_ret_24h.parquet` | `f2861a79ee3c6cc2` | **`1a96d0952eb9d1de`** |
> | `labels/dir_tb_8h.parquet` | `1eb71e01111c53b3` | **`3b76fd3d690b7fa9`** |
> | `features/model_based.parquet` | `4b1a0239330a0e7d` | **`98f64455870d69de`** (24.748 → **24.740** hàng) |
> | `market_data` (input của cả ba nhãn) | `6d63567e7b83eea2` | **`6d63567e7b83eea2` — BẤT ĐỘNG** |
> | `features/financial.parquet` | `0557461a95579969` | **`0557461a95579969` — BẤT ĐỘNG, `03` không chạy lại** |
> | `session_panel` (input của `04`) | `526e4afb5f1679cc` | **`526e4afb5f1679cc` — BẤT ĐỘNG** |
>
> Ba parquet nhãn giờ có **9.810 hàng mỗi cái** (đúng tập khoá của lưới quyết định), số hàng **non-null không đổi**:
> 9.629 / 7.788 / 9.581. `fold_geometry` đổi đúng **2 trong 20 trường**: `train_start` của fold 3 và fold 4,
> `2018-08-30 → 2018-08-31`. Không `train_end`, `val_start`, `val_end` nào dịch, biên holdout bất động. `05_evaluation`
> đã chạy lại và tái tạo 4.056 hàng / 2.028 slot / 43 candidate / 4 STOP + 26 PROCEED + 13 REVISE / 0 BH rejection;
> IC của 33 cột financial tái tạo tới sai số máy (max |Δ| 4,9e-17). **K vẫn 2.514** — không nhãn nào bị bỏ, và ba lần
> chạy lại này **không sinh trial nào**. Chi tiết đầy đủ và chữ ký nghiệm thu (viết TRƯỚC khi chạy) nằm ở `BOT.md`,
> Decisions log 2026-09-08.


**Digest feature KHÔNG dịch.**

- `03_financial_features` (`0557461a95579969`): không đọc khoá nào bị đổi. **Không chạy lại.**
- `04_model_based_features` (`4b1a0239330a0e7d`): **sửa một dòng** — `:116` `SLOTS_PER_YEAR = int(SETUP["evaluation"]["periods_per_year"])` → `int(SETUP["decision"]["slots_per_year"])`. Giá trị **504 → 504**, nên `MIN_TRAIN_SLOTS`, dòng in ở `:136` và mọi output đều bất biến. **Chạy lại đúng một lần và KHẲNG ĐỊNH digest vẫn là `4b1a0239330a0e7d`.** Nếu nó dịch, phép sửa không phải no-op → revert.
- `05_evaluation`: không đọc khoá nào bị đổi. **Không chạy lại.**
- `01_feasibility_analysis:141` gán `PERIODS_PER_YEAR` và **không dùng ở đâu nữa**; đổi sang `decision.slots_per_year` để cái tên không nói dối. Chạy lại tuỳ chọn; không con số in ra nào đổi.

**Phải chạy lần đầu**: bộ test mục 6, và một backtest đơn lẻ `register=False` để đo (a) hold thực, (b) `observed_periods_per_year(daily_returns)` — ghi cả hai vào BOT.md.

**Không được làm**: đừng đụng `session_panel`, `decision_grid`, `build_features`; đừng thêm hàng vào panel; đừng sửa `_CADENCE_CALENDAR_DAYS_PER_PERIOD`; đừng chạy bất cứ stage nào ghi vào `run_log/` của repo.

**Runtime, cần biết trước**: grid từ 504 hàng/năm/kim loại lên ~6.000 → engine đi qua nhiều hơn ~12× số bar cho mỗi backtest, trong khi sweep có ~3.204 backtest. Chi phí giao dịch **không** tăng (phí tính trên fill, không trên bar). Đo thời gian một backtest trước khi bấm sweep; nếu không kham nổi, đòn bẩy trung thực là **cắt bớt số prediction set** (một lựa chọn được khai và được đếm vào K), không phải làm thô lưới giá.

---

## 6. Test phải viết

Đảo `bots/exness_gold_sess/tests/test_backtest_grid.py` (giữ hai test cũ ở dạng đảo: loader không còn raise; `price_grid_expresses_primary_label is True`) và thêm test dưới đây — **khẳng định trên thời gian giữ thực hiện của một backtest thật, không trên giá trị config**:

> Chạy `run_backtest(..., register=False)` cho **sổ `london`**, nhãn `fwd_ret_8h`, trên một cửa sổ validation ngắn (một fold, một quý là đủ), với một prediction frame tất định trên decision grid (điểm số hằng số dương cho cả hai kim loại → mở vị thế ở mọi slot). Lấy `result.engine_result.to_trades_dataframe()`. Với mỗi vị thế đã đóng, tính **số bar H1 của lưới giá giữa fill vào và fill ra** và **số phút đồng hồ** giữa hai fill đó. Khẳng định:
>
> 1. `hold_bars` **== 8 trên 100 %** vị thế (không có 5, không có 19, không có 1);
> 2. `hold_minutes == 480` trên mọi vị thế **trừ** tập early-close đã khai (`label_end_ts` null), và tập ngoại lệ đo được ≤ số hàng đã ghi trong BOT.md (79-93/kim loại trên toàn cửa sổ phát triển) — in ra từng ngày;
> 3. `8 == int(labels.horizons[labels.primary].rstrip("Hh"))` để test gắn vào nhãn đã khai chứ không vào hằng số 8 gõ tay;
> 4. lặp lại (1) cho sổ `ny` — cùng 8 bar, chứng minh 19 h đã chết.

Hôm nay test này **fail** vì loader raise `PriceGridCannotExpressLabel` trước khi có trade nào; sau khi sửa nó pass trên chính con số mà defect nói là sai. Đó là điều kiện "fail hôm nay, pass sau khi sửa" — và nó cũng là **phép đo hiệu chuẩn `HOLD_BARS`** ở mục 3.2.

Thêm một test rẻ chặn được lỗi âm thầm nhất trong bảng mục 2:

> `resolve_decision_schedule(pred_ts_of_the_london_book, cadence="session", step=get_rebalance_step("exness_gold_sess","fwd_ret_24h"))` trả về **mọi** decision instant của sổ đó (không thưa đi), và trên sổ gộp nó **không** trả về một tập chỉ chứa một venue. Đây là bài kiểm tra mà `nasdaq100_microstructure/config/setup.yaml:632-641` gọi là "the values did not change; what they count did".

---

**Next step**: viết lại đúng bốn artefact — `case_studies/utils/backtest_loaders.py` (`_PRICE_CONFIG` + nhánh loader H1 khoá theo bar close), `case_studies/exness_gold_sess/config/backtest/base.yaml` (`data_frequency: 1h`), `case_studies/exness_gold_sess/config/setup.yaml` (bảng mục 2), `case_studies/exness_gold_sess/04_model_based_features.py:116` — rồi chạy test mục 6 **trước** khi lật `price_grid_expresses_primary_label` thành `true`.

**Read**: `case_studies/utils/backtest_runner.py:56-63, 102-152, 1408-1435, 1471-1506, 1582-1601, 2426-2450` (lưới lợi nhuận là lưới NGÀY; lịch rebalance là tập con của prediction timestamps; position rule là ngoại lệ duy nhất) — Tập 3, stage `14_backtest`, Ch16 `09_performance_reporting` / `11_sharpe_ratio_inference`; và `case_studies/nasdaq100_microstructure/config/setup.yaml:247-268, 593-600, 628-646` cùng `.claude/skills/ml4t/references/case-studies.md:139-151`.
