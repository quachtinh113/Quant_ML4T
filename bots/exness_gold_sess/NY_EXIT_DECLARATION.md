# DECLARATION — điểm thoát của sổ New York, `exness_gold_sess`

Ngày 2026-09-08 · **Sửa §3.2 và §3.3 của `PRICE_GRID_DECLARATION.md`** · Repo root `D:/05_Quant/machine-learning-for-trading` (branch `exness-bots`)

Đây là một **declaration**. Builder thực thi từng dòng.

Roadmap: cổng **Giai đoạn 5** (`roadmap.md:168-170`). Vật thể phải sửa là `13_backtest` + `setup.yaml`; `run_log/` vẫn rỗng nên đây là *authoring*.

Guard: **Costs** (fill nào bị tính và ở spread nào), **Parity** (live phải thoát ở đúng chỗ backtest thoát), **Multiple testing** (7 hay 8 phải là *dẫn xuất*, không phải *lựa chọn*), **Point-in-time** (hằng số lấy từ lịch phiên, không từ giá trị từng hàng).

---

## 0. Chẩn đoán đúng, thay cho chẩn đoán của builder

Builder viết trong docstring test (`bots/exness_gold_sess/tests/test_backtest_grid.py:371-373`): *"the metals stop quoting between 21:00 and 22:00 UTC every day"*. **Sai, và BOT.md của chính bot bác bỏ nó** (Decisions log 2026-09-07, đo trên 997 session-day mỗi kim loại):

> *"Giờ nghỉ một tiếng của server nằm ở 21:00-22:00 UTC khi Mỹ dùng DST và 22:00-23:00 UTC khi không, nên đúng một trong hai nến 21:00 / 22:00 được in ra mỗi ngày"* — 630 của 997 ngày rơi vào mùa hè.

Dữ kiện đúng, mạnh hơn nhiều, và nó khép mọi cửa thoát bằng đếm bar:

> **Giờ đóng phiên New York CHÍNH LÀ thời điểm bắt đầu giờ nghỉ Globex hằng ngày, ở cả hai mùa, vì cả hai đều neo vào 17:00 America/New_York.** `bots/_shared/sessions.py` cho `new_york` 08:00-17:00 địa phương (`_features.py:22-24`); giờ nghỉ là 17:00-18:00 địa phương (`bots/assets/XAUUSD.md:110, 188`). Mùa hè: đóng 21:00 UTC, nghỉ 21:00-22:00. Mùa đông: đóng 22:00 UTC, nghỉ 22:00-23:00.

Hệ quả trên lưới khoá theo bar close, với `execution_price: open` + `execution_mode: next_bar`:

- hàng khoá `T` mang `open` = giá tại `T − 60`; một weight ghi tại decision instant `d` khớp ở **open của hàng khoá `d+60`**, tức **đúng giá tại `d`** (lý do §1.1 chọn khoá theo bar close; London đo được lệch 0 phút trên 100 % xác nhận nó);
- muốn khớp tại `label_end_ts = E` thì phải có **hàng khoá `E + 60`**. Ở New York, khoảng `[E, E+60]` **chính là giờ nghỉ** — hàng đó không tồn tại theo định nghĩa, ở cả hai mùa. Thứ Sáu thì `[E, E+60]` là cả cuối tuần.

Nên: **không phải "8 sai, 7 đúng"; mà là "không giá trị bar-count nào khớp ĐÚNG endpoint của New York"**. 8 bar khớp ở bản in **đầu tiên sau** giờ nghỉ (trễ 60 phút; thứ Sáu là bản in mở cửa Chủ nhật 22:00 UTC — 3.420 phút, 3 đêm swap, một gap cuối tuần). 7 bar khớp ở bản in **cuối cùng trước** giờ nghỉ (sớm 60 phút, đều đặn cả hai mùa: mùa hè 20:00 UTC, mùa đông 21:00 UTC).

Kiểm chứng số đo của builder khớp cả hai mùa: mùa hè entry row khoá 14:00 → +7 = 21:00, +8 = 23:00 → 540 phút; mùa đông entry row khoá 15:00 → +7 = 22:00, +8 = 00:00 → 540 phút. Vì thế 1.574/1.984 ở 540 mà **không** tách theo mùa — đúng như đo.

---

## 1. Sổ New York thoát thế nào

### 1.1 Phán quyết: `TimeExit` **đủ sức diễn đạt**; giá trị là **7** cho `ny`, **8** cho `london`

Không cần position rule mới trong thư viện dùng chung. Không đổi `execution_price` / `execution_mode`. Stage truyền được ngay hôm nay: `risk={"position_rules": [{"type": "time_exit", "bars": N}]}` → `_build_position_rules` (`case_studies/utils/backtest_runner.py:2527-2551`) → `broker.set_position_rules` (`:1575-1576`). Việc duy nhất phải sửa trong `case_studies/exness_gold_sess/13_backtest.py:453-477` là **chuyển khối `risk` vào trong vòng lặp `for book`** và lấy `hold` theo `(label, book)`.

**Ba phương án đã cân, và tại sao 8 bar thua tuyệt đối:**

| | thoát ở | spread thực của khoảnh khắc đó | parity với live | kill criteria của bot |
|---|---|---|---|---|
| 8 bar (hiện trạng) | bản in **đầu tiên sau** giờ nghỉ / mở cửa Chủ nhật | **chưa từng đo** — bucket `other` **rỗng** trên cả hai kim loại (`bots/assets/XAUUSD.md:216`); `XAUUSD.md:107,110` gọi đúng hai khoảnh khắc này là "spread giãn, thanh khoản mỏng, gap khi mở lại" | live sẽ khớp ở gap mở cửa, không tái lập được | **vi phạm (g) trên 100 % vị thế NY** |
| **7 bar (chọn)** | bản in **cuối cùng trước** giờ nghỉ (20:00 / 21:00 UTC) | nằm gọn trong bucket `new_york` đã đo (hằng số 260/30 points) | một giờ giao dịch bình thường, gửi lệnh thị trường được | không vi phạm điều nào |
| rule mới khớp tại **close** của bar +7 | đúng `label_end_close` | bản in cuối trước halt: spread giãn, engine tính spread trong phiên → **tính thiếu** | phá đối xứng entry/exit, buộc live có nhánh riêng cho NY (Ch25 §25.1: hai đường code sẽ phân kỳ) | — |

**Dữ kiện chốt hạ là kill criterion (g) của chính bot**: *"tạm dừng khi một mã bị halt/gap qua cửa sổ nghỉ hằng ngày trong khi bot còn giữ lệnh"*. Cách thoát 8 bar làm đúng việc đó **trên 100 % vị thế New York, theo thiết kế**. Một backtest mà mọi giao dịch của nó đều kích hoạt tiêu chí dừng của chính nó thì không phải backtest bảo thủ — nó định giá một rủi ro mà nhãn không chứa, và bot đã tự đặt tên cho rủi ro đó. Đây là lý do bác bỏ, không phải "7 đẹp hơn".

Phương án rule-mới bị bác thêm vì **Costs**: `slippage_bps` = p90 spread đo trong phiên sẽ bị tính cho một crossing ở khoảnh khắc chưa từng được đo. 7 bar khớp ở giờ đã đo, nên con số chi phí đúng nghĩa — điều kiện để cổng breakeven nói được gì.

Giá phải trả của 7 bar, khai rõ: **cửa sổ giao dịch thực của sổ NY là 7 giờ trên một nhãn 8 giờ** — cắt bớt, đều đặn, cùng chiều, trên 100 % hàng, và luôn nằm **bên trong** cửa sổ nhãn. So với 8 bar (dư 1 giờ, cộng 18,5 % dư 49 giờ và 3 đêm swap), đây là sai lệch duy nhất theo hướng bảo thủ.

### 1.2 Hằng số phải **được dẫn xuất**, không được gõ tay

Đây là điểm quan trọng nhất. Đừng viết `{"london": 8, "ny": 7}`. Viết **một luật**, áp cho cả hai sổ, để hằng số tự sinh ra và tự gãy to tiếng nếu broker đổi lịch nghỉ:

> `HOLD_BARS[book]` = số hàng lưới giá giữa **entry row** và **hàng cuối cùng có fill instant ≤ `label_end_ts` đã niêm phong**.
>
> Chính xác, trên `bar_index` của lưới đã đăng ký: `i0` = chỉ số hàng đầu tiên có khoá `> d` (hàng mà entry khớp); `i1` = chỉ số hàng cuối cùng có khoá `≤ label_end_ts + 60min`; `N = i1 − i0`. Khẳng định `N` là **hằng số trong mỗi sổ**, trên mọi symbol, mọi mùa, mọi fold; nếu không, dừng và báo cáo.

Chạy công thức này ra đúng: `london` → 8, dư 0 phút; `ny` → 7, dư 60 phút, cả hai mùa (mùa hè `E+60 = 22:00` không có hàng → hàng cuối là 21:00 = `i0+7`; mùa đông `E+60 = 23:00` không có hàng → hàng cuối là 22:00 = `i0+7`).

Ràng buộc thực thi:

- dẫn xuất chạy trên **panel đã niêm phong + lưới giá đã đăng ký**, cửa sổ `universe.history_start → evaluation.holdout_start`. **Không được đọc holdout.**
- hàng có `label_end_ts` null (early close) **loại khỏi phép dẫn xuất** (không có endpoint để đo) nhưng **vẫn giao dịch** với cùng hằng số — §3.3 giữ nguyên.
- không có point-in-time violation: `label_end_ts` ở đây là **giờ đóng phiên của venue**, biết trước theo lịch, hệt luật thứ Sáu; và cái đi vào spec là **một hằng số mỗi sổ**, không phải tra cứu theo hàng.

### 1.3 London có phải đổi không? Bất đối xứng có đúng không?

**London giữ nguyên 8. Bất đối xứng là đúng, và nó không phải bất đối xứng của LUẬT — chỉ của HẰNG SỐ.** Một luật duy nhất (mục 1.2) áp cho hai venue; hai venue có tape khác nhau vì giờ đóng của một trong hai trùng giờ nghỉ, nên hai con số khác nhau rơi ra. Đó chính là hình dạng đúng của một quy tắc đã khai: hằng số là *đầu ra* của luật, không phải *đầu vào*. Đo được rồi: London 8 bar / 480 phút / lệch 0 phút trên 100 %.

### 1.4 Tiêu chí đạt mới, thay §3.2

§3.2 viết *"tiêu chí đạt duy nhất: hold fill-to-fill = 8 bar = 480 phút"*. Phép đo chứng minh tiêu chí đó **không đạt được** trên NY dưới quy ước fill đã khai. Thay bằng phát biểu tổng quát hơn (nó **quy về** 480 phút ở mọi nơi tape liên tục, nên London không đổi gì):

> 1. `exit_fill_instant` là fill instant **muộn nhất ≤ `label_end_ts`**;
> 2. `0 ≤ label_end_ts − exit_fill_instant < 60` phút trên 100 % vị thế có endpoint niêm phong;
> 3. `hold_minutes == 60 × hold_bars` — **không khoảng đóng cửa nào nằm trong lần giữ** (mệnh đề giết weekend carry), miễn trừ đúng tập early-close đã khai.

---

## 2. Nhãn có dịch không? Digest nào chạy lại?

**Không nhãn nào, không luật endpoint nào, không digest nào dịch. Không stage 01-05 nào chạy lại.**

- Luật endpoint (`_features.decision_grid:202-222`) **đúng và giữ nguyên**. `fwd_ret_8h` vẫn là "decision close → session close", `labels.horizons` vẫn `8H`.
- `HOLD_BARS` chỉ sống trong khối `risk` của spec `13_backtest`. Không stage 01-05 nào đọc nó.
- `backtest_hash` của các spec `ny` đổi so với lần chạy characterisation — vô hại: `register=False`, `run_log/` rỗng.

**Đổi cơ chế thoát, không đổi nhãn** — nhưng phải **khai khoảng chênh còn lại**, nếu không bot đang tự nhận một parity nó không có:

- giữ `backtest.price_grid_expresses_primary_label: true` (lưới *có* diễn đạt chân trời 8 bar; thứ bị mất là hàng khớp, do tape nghỉ);
- thêm khoá mới `backtest.exit_gap_minutes_by_book: {london: 0, ny: 60}` kèm `derivation: last_grid_fill_at_or_before_label_end` và ngày đo. Không vào hash, nhưng phải có mặt để người đọc sau tìm thấy tuyên bố và phép kiểm ở cùng một chỗ;
- ghi vào `BOT.md` Decisions log: sổ NY **huấn luyện trên nhãn 8 h, giao dịch cửa sổ 7 h**, chênh lệch đã đo và đã khai. Mọi con số phase 5-7 đọc từ **fill thực**, không từ nhãn, nên chuỗi bằng chứng vẫn liền — nhưng khớp nối IC(8 h) → P&L(7 h) là có thật và phải được **đo** ở Giai đoạn 6: `corr(fwd_ret_7h_probe, fwd_ret_8h)` và IC của cùng bộ prediction trên probe 7 h, trên riêng hàng NY. Đây là **chẩn đoán để báo cáo, không phải mặt phẳng để chọn**; không được chọn model theo nó. Nếu IC suy giảm đáng kể, câu trả lời trung thực ở phase 6 là **niêm phong lại nhãn NY ở chân trời giao dịch được** — lúc đó digest *sẽ* dịch và đó là một identity mới, khai trước.

---

## 3. Một trial nữa, hay sửa lỗi thực thi của một luật đã khai?

**Sửa lỗi thực thi. K không đổi.**

`16_strategy_simulation/12_dsr_validation.py:107-114, 152-163` tính DSR từ `n_trials` **và** `variance_trials` = phương sai của các Sharpe **đã quan sát trên các cấu hình**; `:253-255` nói rõ K là "số chiến lược đã thử". **Một tham số được cố định bởi phép đo LỊCH của tape — hàng nào mang endpoint — không bao giờ sinh ra một Sharpe, nên nó không vào mẫu cực đại đó.** Nó cùng loại với `rebalance_step = 1` hay luật thứ Sáu: dẫn xuất từ luật đã khai, không phải chọn từ kết quả.

Hơn nữa nhánh này **đã được khai trước** trong chính §3.2: *"Nếu hold thực đo được là 9 bar…, sửa hằng số thành 7, ghi con số đo được và lý do vào BOT.md, chạy lại test."* Phép đo vừa chạy chính là phép hiệu chuẩn đó; nó phát hiện một ca §3.2 không lường trước (tape nghỉ) và hằng số được dẫn xuất lại **từ cùng một tiêu chí**, đã tổng quát hoá ở mục 1.4.

**Ba điều kiện, thiếu một là thành trial** — ghi vào BOT.md:

1. giá trị đến từ công thức mục 1.2, **không** từ so sánh hiệu năng của 7 và 8;
2. **đúng một** giá trị chạy cho mỗi sổ trong toàn bộ sweep; nếu ai đó chạy cả 7 và 8 rồi chọn theo Sharpe thì `K += |books| × |labels| × |predictions| × |signal_specs|` và DSR phải tính lại;
3. lần chạy characterisation `register=False` chỉ báo cáo **phân phối hold**, không báo cáo Sharpe — đúng như builder đã làm. Không được lấy bất kỳ con số hiệu năng nào từ nó ra dùng.

Arm `hold_2h` (2 bar) không đổi và vẫn kích hoạt trước cả hai hold nền.

---

## 4. Ba phán quyết còn lại

### 4.a `calendar.data_frequency: 1h` không tới engine — **giữ nguyên giá trị, sửa lời chú, thêm một phép kiểm**

Builder chẩn đoán đúng, và làm đúng khi thực thi rồi báo thay vì tự thay. Nhưng **không được đẩy giá trị vào `feed:`**:

- Bằng chứng: `backtest_presets.py:210-215` merge `{**derived_feed, **explicit_feed}`, và `derived_feed["data_frequency"] = _infer_data_frequency(case_config.cadence)` (`:103`, `:71-81`) → `"session"` không khớp token nào → `"daily"`. Khoá dưới `calendar:` **không được đọc bởi bất kỳ dòng nào trong repo** — grep `data_frequency` chỉ ra 14 file `base.yaml` và không consumer nào.
- Nó **decorative ở MỌI case study**, không riêng bot này: `nasdaq100_microstructure/config/backtest/base.yaml:22` khai `1m` trong khi cadence của nó suy ra `15m` — cùng phân kỳ, trên bot intraday nhất của repo, chưa ai để ý. Đây là **phát hiện cross-bot**, không thuộc về bot này để sửa.
- Đẩy `feed: {data_frequency: 1h}` vào bây giờ là đổi một tham số engine mà **hiệu ứng chưa từng được đo trong repo này**, để sửa một khoá mà hiệu ứng hiện tại **bằng không**. Con số duy nhất nó có thể phá — annualisation — đã đo và đã đúng: `observed_periods_per_year = 292.20` trên lưới ngày (292,20/252 = 1,16, trong band của `reconcile_periods_per_year`, `backtest_runner.py:142-143`), xác nhận dự báo của §2 rằng lưới H1 **không** nâng annualisation và bác bỏ tiền đề ~6.084. Rủi ro một chiều: chỉ có thể xấu đi.

**Phán quyết**: giữ `calendar.data_frequency: 1h`; **viết lại lời chú** trong `config/backtest/base.yaml:33-39` để nói đúng sự thật đã đo; và **biến nó thành sự kiện được kiểm**: thêm vào test lưới một assertion rằng frequency mà engine resolve đúng là cái ta nghĩ, cộng với `observed_periods_per_year(daily_returns)` của một lần chạy thật nằm trong band quanh 252. Một khoá trơ mà được khẳng định thì không còn nói dối được.

### 4.b `bars_per_day: 2` → **1**. Đây là lỗi thật, và là tàn dư của §3.4

`signals.py:377`: `W = int(lookback_days * bars_per_day)` đếm **hàng của frame prediction, theo symbol**. `apply_session_filter` chạy ở `backtest_runner.py:1162`, **trước** khi weight được dựng — nên trong **mọi** engine book, mỗi symbol có **đúng một hàng mỗi ngày**. Ý định đã khai là ngưỡng lăn 63 ngày; ở `bars_per_day: 2` cửa sổ là 126 hàng ≈ **126 ngày giao dịch ≈ 6 tháng**, và warmup `min_samples = W//2` là 63 ngày thay vì 31.

Giá trị 2 **đúng khi `both` còn là engine book** — nó là tàn dư của chính §3.4 khi sổ gộp bị rút thành tổng hai tay áo. Quan hệ nhân quả đó phải vào Decisions log. Tiền lệ: `exness_fx_d1/config/setup.yaml:174` khai `bars_per_day: 1` vì cùng lý do. Không phải trial: một spec, một giá trị, cố định trước sweep, bằng dẫn xuất.

**Và phải được khẳng định, không chỉ được viết**: `13_backtest` đã đo tách sổ rồi (`:301-324`) — mở rộng để khẳng định trong mỗi sổ, mỗi symbol, số hàng mỗi ngày **bằng đúng `bars_per_day`** trên ngày modal, và in phân phối.

### 4.c Seam `apply_session_filter` — **đúng, và là seam DUY NHẤT đúng**

1. **Tiền lệ**: nó nằm cạnh `apply_universe_filter` (`backtest_runner.py:876`) tại cùng call site (`:1150-1167`) — hàm tồn tại vì đúng bài toán này, và đúng lý do builder nêu.
2. **Thứ tự là bắt buộc, không phải tiện lợi**: nó chạy **trước** khi `rebalance_schedule` được dựng từ prediction timestamps (`:1408-1410`). Nếu lọc sau, engine sẽ rebalance tại decision instant của venue kia và **điều chỉnh vị thế giữa chừng lần giữ** — tái tạo đúng lỗi sổ gộp của §3.4 ngay bên trong một sổ đơn.
3. **Nó cũng nằm trước khi signal được dựng** — và đó chính là thứ ép ra phán quyết 4.b. Hai phán quyết khoá vào nhau.

Guard phạm vi (`:825-830`, raise trên mọi case study khác) khớp quy ước "no silent fallback". Hai tinh chỉnh không chặn: docstring nên dẫn số dòng call site để tính chất thứ tự tìm được; message ValueError khi frame rỗng nên in tên sổ và khoảng instant.

---

## 5. Test chứng minh phép sửa, trên fill thực hiện

Đảo `test_the_new_york_book_exits_after_its_label_endpoint` (đừng xoá — đảo), gộp vào một test chạy trên **cả hai sổ**, và **sửa dữ kiện sai trong docstring** (mục 0). Chạy trên `run_backtest(..., register=False)`, prediction tất định, `to_trades_dataframe()`:

1. **Hằng số là dẫn xuất, không phải hằng gõ tay.** Test tự tính `HOLD_BARS[book]` bằng công thức mục 1.2 từ panel niêm phong + lưới đã đăng ký, khẳng định nó **hằng số trong mỗi sổ trên mọi symbol và mọi mùa**, in ra, rồi mới truyền vào `risk`. Khẳng định kèm: `HOLD_BARS["london"] == 8` **và** `HOLD_BARS["ny"] < HOLD_BARS["london"]`, để một ngày broker bỏ giờ nghỉ thì test đỏ chứ không im lặng.
2. **Cơ chế**: `hold_bars == HOLD_BARS[book]` và `bars_held == HOLD_BARS[book]` trên 100 %, `exit_reason == {"time_stop"}` trên 100 %, cả hai sổ.
3. **Mệnh đề chính, trên instant fill**: `0 ≤ label_end_ts − exit_fill_instant < 60` phút trên **100 %** vị thế có endpoint niêm phong, cả hai sổ; và chênh lệch đo được **bằng đúng** con số công thức mục 1.2 dự đoán cho sổ đó (london 0, ny 60) — in phân phối, không hard-code hai số trong assertion.
4. **Mệnh đề giết weekend carry, phát biểu trên đồng hồ chứ không trên lịch**: `hold_minutes == 60 × hold_bars` trên mọi vị thế **trừ** đúng tập early-close đã khai. Kèm: `hold_minutes > 24×60` trên **0** vị thế của tập không-miễn-trừ, in số đếm để hồi quy nhìn thấy được (hôm nay: 368).
5. **Lỗ hổng chưa ai đo — giá fill, không phải chỉ instant.** Test London hiện tại (`:355-365`) tự suy `exit_price_ts = exit_time − 60` rồi so *instant*; nếu engine thực ra khớp ở **close** của hàng thì "lệch 0 phút" là hệ quả của chính giả định đó và không chứng minh gì. Phép sửa: join `exit_price` của trade vào lưới giá theo `exit_time` và **đo** nó bằng `open` của hàng đó; khẳng định bằng đúng `open` (hoặc `open × (1 ± slippage_bps)` nếu engine ghi giá đã trừ chi phí) và **ghi lại kết quả nào đúng** — dữ kiện chưa ai trong repo này đo. Với London, cùng phép join phải cho `exit_price ≈ label_end_close` của panel niêm phong.
6. **Tập early-close vẫn in ra, không lọc**: 66 hàng trong cửa sổ validation, toàn NY, 33/kim loại; khẳng định số đếm khớp con số ghi trong BOT.md và in `hold_minutes` thực của từng hàng. Chúng miễn trừ khỏi (3) và (4), **không** miễn trừ khỏi (2).

Test này **đỏ hôm nay** (NY đang 540/3.420 phút) và **xanh sau khi sửa**.

---

## 6. Ba việc bị lộ ra, phải khai trước sweep

1. **Sổ `fwd_ret_24h` có phiên bản khuyết tật của chính vấn đề này, chưa đo.** Với `drop_friday`, hàng prediction thứ Sáu bị lọc, nên vị thế mở chiều thứ Năm **không còn weight nào ghi đè** ở decision thứ Sáu — thứ đóng nó là **backstop 24 bar**, mà 24 hàng lưới ≈ 25 giờ đồng hồ (mỗi ngày nuốt một giờ nghỉ). Ý định §3.3 ("sổ giữ tiền mặt ngày thứ Sáu") đang được thoả mãn **do tình cờ, không do thiết kế**, và thời điểm thoát thực chưa ai đo. Trước khi sweep nhãn 24 h: đo bằng đúng harness mục 5, và **dẫn xuất** backstop theo cùng luật mục 1.2 thay vì gõ 24.
2. **Tàn dư early-close, 1,6 % sổ NY**: cách sửa trung thực là **lịch nghỉ lễ công bố trước** (biết trước, hợp lệ như luật thứ Sáu), **không** phải lọc theo endpoint null (chỉ biết sau — cấm bởi §3.3). Repo chưa có file lịch đó, nên: giữ nguyên (giao dịch bình thường), **đếm và đặt tên trong BOT.md**, và đưa vào Giai đoạn 6 kèm điều kiện "khi có file lịch". Khai bây giờ hoặc không bao giờ — thêm bộ lọc này *sau khi nhìn kết quả* vừa là một trial vừa là một cú peek.
3. **Kill criterion (g) cần được tinh chỉnh, không phải bị bỏ qua**: sau phép sửa, (g) không còn kích hoạt trên giờ nghỉ hằng ngày; nó vẫn kích hoạt trên early close. Ghi rõ vào BOT.md để phase 8 không phát báo động cho một trạng thái đã khai.

---

**Next step**: sửa `case_studies/exness_gold_sess/13_backtest.py:453-477` — chuyển khối `risk` vào trong vòng lặp `for book`, lấy `HOLD_BARS[(label, book)]` từ hàm dẫn xuất ở mục 1.2 (khẳng định hằng số trong mỗi sổ), rồi chạy test mục 5 **trước** khi đụng tới `bars_per_day` hay lời chú `base.yaml`.

**Read**: `bots/exness_gold_sess/BOT.md` Decisions log 2026-09-07 (giờ nghỉ theo mùa, 630 của 997 session-day) cùng Kill criteria (g) — hai dữ kiện quyết định phán quyết mục 1; và `16_strategy_simulation/12_dsr_validation.py:107-114, 152-163, 253-255` (Tập 3, Ch16 `12_dsr_validation`) cho định nghĩa trial mà mục 3 dựa vào.
