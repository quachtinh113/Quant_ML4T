# DECLARATION — hình học fold cho ba nhãn, `exness_gold_sess`

Ngày: 2026-09-08 · Mentor declaration, gỡ chặn Phase 4 · Repo root `D:/05_Quant/machine-learning-for-trading` (branch `exness-bots`)

Đây là một **declaration**, không phải gợi ý. Builder thực thi từng dòng. Nó **cho phép ba digest nhãn dịch**, điều mà `PRICE_GRID_DECLARATION.md` §5 cấm — mục 2 dưới đây là văn bản cho phép đó.

Vị trí roadmap: **Giai đoạn 4 · Mô hình** (`roadmap.md:144-146`). Guard: *Leakage across folds* (`mentor-protocol.md:25`) và *Multiple testing* (`mentor-protocol.md:27`). Guard đã bắt đúng: đây là **leak**, không phải coverage gap. Không được đi tiếp bằng cách làm guard im lặng.

**Review cross-bot, ghi trước mọi thứ khác**: `case_studies/exness_gold_sess/04_model_based_features.py` **không gọi** `assert_variant_folds_are_out_of_sample`, trong khi ba case study anh em đều gọi ngay tại stage sản xuất — `case_studies/fx_pairs/04_model_based_features.py:534`, `case_studies/exness_fx_d1/04_model_based_features.py:528`, `case_studies/xau_fx_mt5/04_model_based_features.py:507`, `case_studies/etfs/04_model_based_features.py:390`. Bản fork bỏ mất lời gọi đó, nên khuyết tật nổi lên chậm hai stage. Đây là "hai bot giải cùng một bài theo hai cách mà không ghi lý do"; phải sửa cùng lần chạy lại `04` dưới đây.

---

## 1. Chọn cách nào, và mỗi cách trả giá bằng bằng chứng gì

**Chọn: A′ — "một lưới, ba nhãn". Mọi nhãn được publish trên cùng một tập khoá (timestamp, symbol) của lưới quyết định, giá trị `null` ở nơi luật của chính nhãn đó chưa xác định.** Không phải A nguyên bản, không phải B, không phải C, không phải D.

Cơ chế, một dòng: khuyết tật nằm ở `case_studies/exness_gold_sess/02_labels.py:693`

```python
labels_df.select(columns).drop_nulls(),
```

`.drop_nulls()` khiến **tính xác định của nhãn quyết định timeline của nhãn**, và `_derive_modeling_splits` (`case_studies/utils/cv_window.py:164-178`) suy fold từ chính timeline đó. Ba nhãn ragged → ba thang fold khác nhau → artifact `04` chỉ đóng dấu **một** thang. Chính `utils/modeling.py:941-943` viết ra nguyên tắc bị vi phạm: *"Deriving boundaries from the feature-joined frame lets warm-up nulls or feature availability shift the calendar"* — hôm nay warm-up (ATR 14 của `dir_tb_8h`) và availability (thứ Sáu của `fwd_ret_24h`) đang dịch chính cái lịch đó. Sửa đúng chỗ là chỗ đó, không phải ở buffer (builder đã đo 42 tổ hợp: buffer dịch cả `train_end` lẫn `val_start`, nên nó không bao giờ là đòn bẩy — đúng).

A′ đúng về cơ học: ba timeline bằng nhau → `val_start` fold F bằng nhau → guard so `variant.val_start > primary.train_end` với khoảng cách đúng bằng purge của nhãn chính (1 phiên), dương trên cả 8 hàng (2 variant × 4 fold). Buffer riêng của mỗi nhãn vẫn làm việc của nó — kéo `train_end` của chính nhãn đó lùi thêm — chứ không còn kéo theo `val_start`.

**Giá của từng phương án, bằng bằng chứng chứ không bằng CPU:**

| | Mất gì về bằng chứng | Trả gì | Phán quyết |
|---|---|---|---|
| **A′ (chọn)** | **Không mất gì.** Không một hàng có nhãn nào bị thêm, bớt hay đổi giá trị: chỉ thêm hàng `null`. `fwd_ret_24h` giữ trọn 305 hàng early-close mà Decisions log 2026-09-08 đã giành lại | 3 digest nhãn dịch (republish), 1 lần chạy `02`, 1 lần chạy `04` phải tái tạo digest | **Thực hiện** |
| A nguyên bản (neo theo tập khoá của nhãn chính) | Mất ~305 giá trị `fwd_ret_24h` ở phiên early close — tái lập đúng thiên lệch mà bản vá phase 2 (defect A) vừa gỡ | Rẻ hơn A′ một chút | Bỏ; rẻ hơn bằng cách bỏ dữ liệu |
| B (`04` ghi geometry theo từng nhãn) | Ba nhãn sẽ **đọc giá trị feature khác nhau** trên cùng một (timestamp, symbol) → so sánh horizon không còn cùng feature. Đây là phiên bản nhẹ của chính lỗi của D | Phải sửa hạ tầng dùng chung — `temporal_artifact_fold_boundaries` trả **một** danh sách, artifact có **một** cột `fold`, `require_fold_scoped_temporal_compatibility` (`case_studies/research/cv.py:36-46`) khớp theo `fold` id — 8 case study khác phụ thuộc; ước lượng của `04` nhân ba | Bỏ |
| C (chỉ fit `fwd_ret_8h`) | Mất **so sánh horizon** (8h có phải horizon đúng không) và mất **toàn bộ nhánh phân loại** (triple barrier, giá trị của việc *không* vào lệnh). Bot không trả lời được hai câu hỏi nó đã khai trong `setup.yaml::labels`. Đổi lại K giảm còn 1.068 — đó là **lợi** về bằng chứng, không chỉ về compute | Rẻ nhất | **Fallback duy nhất**, dùng khi và chỉ khi điều kiện rẽ nhánh ở mục 2 bật |
| D (bỏ 10 cột `04` khỏi variant) | Guard chỉ chạy khi có temporal artifact (`utils/modeling.py:953`); bỏ artifact = làm guard **im lặng**, không phải an toàn. Và ba nhãn khi đó đọc hai tập feature khác nhau, nên mọi kết luận về horizon lẫn lộn horizon với feature set | — | **Bác bỏ, xác nhận builder đúng khi bác bỏ ngay** |

**Trả lời thẳng**: **không**, không phải "hai nhãn variant không thể mang feature chương 9 một cách trung thực trên lưới này". Vật cản là **một quy ước xuất bản** (`.drop_nulls()` ở `02_labels.py:693`, quy ước chung của cả 8 case study khác), không phải bản chất của lưới hai slot/ngày. Lưới hai slot/ngày chỉ làm quy ước đó **lộ ra**; trên lưới daily của `exness_fx_d1` cùng quy ước ấy sinh ra lệch 20 instant và may mắn không đủ để vượt purge.

---

## 2. Digest nào dịch, stage nào chạy lại, và kiểm chứng bằng gì

**Digest nhãn: cả ba dịch.** `fwd_ret_8h d6487f7333a683cc`, `fwd_ret_24h f2861a79ee3c6cc2`, `dir_tb_8h 1eb71e01111c53b3` — cả ba parquet lớn lên vì thêm hàng `null`, nên `value_digest` dịch. `PRICE_GRID_DECLARATION.md` §5 nói: *"một digest dịch = dừng lại và báo cáo"*. **Đây chính là báo cáo đó, và đây là quyết định của mentor cho phép nó dịch.** Điều kiện cho phép giống hệt tiền lệ đã ghi trong Decisions log 2026-09-08 (`bots/exness_gold_sess/BOT.md:120`, `e9f92066b0718d64 → f2861a79ee3c6cc2`): run log **rỗng**, `registry.db` không tồn tại, manifest md5 `1aeeb63842c6e6d69a41ff140332d665` chứng minh chưa một hàng nào được đăng ký. Đây là thời điểm rẻ nhất trong cả dự án để trả giá này; sau phase 5 nó là không thể.

**`market_data 6d63567e7b83eea2` KHÔNG dịch** và phải được khẳng định: nó được lấy trên `panel.drop_nulls("label_end_ts")` (`02_labels.py:179-180`), tức đúng tập hàng mà A′ đang thêm vào dưới dạng `null`.

**Chạy lại:**

- **`02_labels`: có.** Thay đổi đúng một biểu thức, `.drop_nulls()` → `.drop_nulls(["timestamp", "symbol"])`. Không đụng luật nhãn nào.
- **`03_financial_features`: KHÔNG.** `inputs` của nó là `session_panel` + `load_mt5_bars:1h` + `load_mt5_bars:daily` (`03_financial_features.py:678-684`), không có nhãn. `0557461a95579969` đứng yên.
- **`04_model_based_features`: có, đúng MỘT lần** — gộp chung với lần chạy lại mà `PRICE_GRID_DECLARATION.md` §5 đã lên lịch cho sửa đổi một dòng `SLOTS_PER_YEAR` (504 → 504). Trong cùng lần đó thêm lời gọi `assert_variant_folds_are_out_of_sample(CASE_STUDY_ID, PRIMARY_LABEL)` theo đúng chỗ ba case study anh em đặt nó.
- **`05_evaluation`: không bắt buộc.** Nó không ghi artifact có digest (không gọi `write_artifact`), chỉ đọc nhãn rồi `drop_nulls`. Nếu chạy, phải tái tạo **43 candidate / 4.056 hàng validation / 4 STOP, 26 PROCEED, 13 REVISE**.

**Điều kiện rẽ nhánh, đo TRƯỚC khi chạy lại `04`, không phải phán đoán:**

```
set(dates(fwd_ret_8h.parquet mới))  ==  set(dates(fwd_ret_8h.parquet cũ))   ?
```

`04:176-182` suy thang fold từ **tập ngày duy nhất** của parquet nhãn chính. Các hàng độn nằm ở phiên early close (venue kia trong ngày đó thường vẫn resolve, nên ngày đã có sẵn) và ở đuôi chuỗi (nơi có thể sinh ngày mới, kể cả sau `holdout_start`, nơi `build_holdout_cv` đọc).

- **Bằng nhau** → chạy lại `04`, digest **phải** vẫn là `4b1a0239330a0e7d`. Đi tiếp.
- **Khác nhau** → **DỪNG, báo cáo, không chạy `04`.** Khi đó thang fold dịch và đó là một tuyên bố thứ hai (mở lại bằng chứng phase 3). Fallback tạm thời cho vòng này là **C** với K = 1.068, ghi vào Trials table kèm ngày và lý do.

**Chữ ký chính xác để kiểm chứng thay đổi của `04` đúng là thay đổi đã định:**

| Trường trong sidecar | Sau khi sửa |
|---|---|
| `digest` | **`4b1a0239330a0e7d`, không dịch** (`digest = value_digest(df)`, chỉ trên giá trị của bảng — `case_studies/utils/artifact_digest.py:147`) |
| `inputs["labels:fwd_ret_8h"]` | **dịch** (nó là `value_digest(label_frame)` trên cả parquet nhãn chính — `04_model_based_features.py:697`) |
| `inputs["session_panel"]` | không dịch |
| `fold_geometry` | 5 fold, biên giống hệt, fold 4 vẫn là holdout vintage |

Chữ ký đó đọc thành đúng một câu: **vintage của timeline nhãn dịch, giá trị được fit thì không.** Nếu `digest` dịch → phép sửa không phải no-op → revert, đúng luật §5. Ngoài ra ba seal của `04` vẫn phải in **0.00e+00**, purge đo trên lưới vẫn **3-5 slot**, vẫn **10 cột / 24.748 hàng / 5 fold**.

**Bất biến bắt buộc của `02`** (điều biến A′ thành thay đổi *timeline* chứ không phải thay đổi *nhãn*) — khẳng định trong notebook, không chỉ in ra:

1. `new.drop_nulls(<label>)` **trùng khít** parquet cũ trên cả khoá lẫn giá trị: số hàng non-null đúng **9.629 / 7.788 / 9.581**.
2. Ba parquet có **cùng `n_rows`** và **cùng tập khoá**.
3. `MARKET_DATA_DIGEST == 6d63567e7b83eea2`.
4. `dir_tb_8h.parquet` vẫn mang cột `fwd_ret_8h` và vẫn qua được `LabelCatalog.publish` (`case_studies/research/labels.py:143-159`) khi cột eval có `null`.

Đường fit không cần thay đổi gì để chịu được `null`: `utils/modeling.py:1896-1898` đã lọc `is_not_null() & is_not_nan()` trên nhãn **trong từng fold** trước khi fit. Hàng độn không bao giờ vào một lần fit nào.

**Cập nhật hồ sơ**: `BOT.md` phase-1 row + một hàng Decisions log theo đúng khuôn của hàng 2026-09-08; `PRICE_GRID_DECLARATION.md` §5 (bốn digest mới) và §4 (đang còn ghi "≈ 3 × 3 × 178 × 2 = 3.204", sai theo phép đo ở mục 3).

---

## 3. K, viết bằng số học — và nhãn bị bỏ có được trừ khỏi K không

**Xác nhận phép đo của builder là đúng, 2.514 chứ không phải ~3.204.**

```
prediction sets  fwd_ret_8h  = 28 linear + 15 GBM × 10 checkpoint = 28 + 150 = 178
                 fwd_ret_24h = 28        + 15      × 10           = 178
                 dir_tb_8h   = 13 logistic + 5 multiclass × 10    = 13 + 50  =  63
                 tổng = 178 + 178 + 63 = 419

K = |books| × tổng_nhãn(|prediction sets|) × |signal specs|
  = 3 × 419 × 2 = 2.514

tách theo nhãn:  8h 3×178×2 = 1.068 | 24h 1.068 | dir_tb 3×63×2 = 378   (tổng = 2.514)
```

Chênh 690 so với spec cũ đúng là do menu phân loại là 13 + 5 chứ không phải 28 + 15. Đây **không** phải menu bị cắt: mọi preset đã khai vẫn được fit. Với A′, **K giữ nguyên 2.514** vì không nhãn nào bị bỏ.

**Bỏ một nhãn khỏi phase 4 có trừ được trial của nó khỏi K không?** Quy ước của repo, đọc từ chỗ DSR thực sự được tính: `case_studies/utils/uncertainty.py:1085-1201` tính `deflated_sharpe_ratio` trên **ma trận cohort đã align** của các chuỗi `daily_returns.parquet` **đã đăng ký**; `k_variants` là kích thước cohort, `member_digest` (`:1109-1113`) đặt tên đúng cohort đó, và `n_trials_effective_mp` / `_er` là hiệu chỉnh Marchenko-Pastur / effective-rank trên ma trận tương quan (`:1168-1179`). Cổng phase 5 nói *"Deflated Sharpe với **số trial thực**"* (`roadmap.md:167`).

Chính xác:

- Một nhãn **chưa bao giờ được fit** không sinh cột nào trong ma trận, không nằm trong `member_digest`, và **không đóng góp vào kỳ vọng cực đại** — DSR hiệu chỉnh cho `max` trên K lần thử *đã được chấm*; cái chưa chấm không thể nằm trong `max`. **Được trừ.**
- Điều kiện để phép trừ đó trung thực: quyết định bỏ phải **không được điều kiện hóa trên kết quả nào**. Ở đây thoả: registry rỗng, manifest md5 bất biến, nguyên nhân là một khuyết tật hình học phát hiện trước lần fit đầu tiên. Đây là **sửa đổi tiền đăng ký**, không phải cắt tỉa hậu nghiệm.
- Bookkeeping bắt buộc: ghi một hàng Trials có ngày, ghi K rơi từ 2.514 xuống 1.068 **kèm lý do cơ học**, để người đọc sau không hiểu nhầm rằng 1.446 trial đã được thử và giấu đi.
- **Hoãn không phải là xoá**: nếu variant được phục hồi *sau khi* đã nhìn kết quả của `fwd_ret_8h`, thì lựa chọn đã điều kiện hóa trên kết quả; khi đó cohort là **hợp** của hai lượt, K trở lại 2.514, và DSR phải tính lại trên cohort hợp đó. Phải viết câu này vào Trials table ngay lúc hoãn, không phải lúc phục hồi.

---

## 4. `dir_tb_8h` có đáng fit trên bot này không

**Sửa một lập luận của builder — theo hướng có lợi cho nhãn này.** Việc không có `sample_weight` ở bất kỳ đường fit nào **không** đủ để loại `dir_tb_8h` trên lưới này. Trọng số uniqueness của López de Prado sửa **concurrency biến thiên**; ở đây `02_labels` đo `N_eff/N = 0.5001` vì *"exactly one later decision starts inside every holding window"* — concurrency là **hằng số 2**, nên trọng số uniqueness sẽ **đồng nhất** và đổi đúng bằng không lên ước lượng. Cái mà chồng lấn thực sự làm hỏng là **khoảng tin cậy**, và nó đã được sửa bằng HAC lag.

**Rủi ro thật nằm ở thống kê xếp hạng.**

- `ic_mean` của registry là cross-sectional IC với `min_obs=5` cứng (`case_studies/utils/registry/metrics.py:113`); panel này rộng **2**. `auc_mean_daily` / `auc_t_hac` null vì nhãn ba trạng thái và cross-section hai tên. Builder đã thay bằng **pooled panel IC** của chính bot (`_model_reading.py:123-165`) — đúng, và áp dụng cho *cả ba* nhãn.
- Còn lại `auc_roc`: `metrics.py:143-154` gộp ba lớp thành "up vs not-up", tức **ném lớp timeout chung rổ với lớp down** — đúng cái phân biệt mà triple barrier được dựng ra để tạo; và `analytics.py:376-377` ghi rõ chỉ số pooled đó *"also rewards a model for the base rate moving through time"*, trong khi Hypothesis của bot nói mẫu 2017-2026 có xu hướng tăng rõ rệt. Nên `auc_roc` **không được dùng để chọn**, chỉ được đọc như mô tả.
- Lớp timeout 4.039/9.581 (42 %) **không phải khuyết tật**: đó là cái mà barrier ±0.5 × ATR(14) với hạn 8 bar sinh ra. Nhưng nó có nghĩa 42 % khối lượng của nhãn nói "đừng làm gì", và đó chính là nội dung kinh tế duy nhất mà nhãn này thêm vào so với `fwd_ret_8h`.

**Phán quyết: giữ nó trong menu**, **nhưng ràng vào một cổng khai trước khi fit**. `dir_tb_8h` kiếm được 378 trial của nó khi và chỉ khi, khai vào `setup.yaml` **trước** lần fit đầu tiên:

1. **Khai điểm số vô hướng** mà class probability được quy về để vào pooled panel IC — mặc định `P(up) − P(down)`, khai bằng chữ, không để runner chọn hộ; và `actual` là `fwd_ret_8h` (đúng `labels.classification_eval_label`), y hệt nhãn chính.
2. **Thắng nhãn chính trên chính sân của nhãn chính**: cấu hình `dir_tb_8h` tốt nhất phải có pooled panel IC (HAC) **cao hơn** cấu hình `fwd_ret_8h` tốt nhất trên **cùng bộ fold 0-3**. Thua ở đây nghĩa là 63 prediction set chỉ là 63 lần rút thêm trên cùng một target — chỉ làm phồng kỳ vọng cực đại của DSR mà không thêm chiều thông tin nào.
3. **Hoặc** chứng minh cái mà nhãn hồi quy không diễn đạt được: **giá trị của việc không vào lệnh**. Đo trên tập slot mà mô hình dự đoán lớp timeout — hit rate và lợi nhuận sau chi phí trên tập *được giao dịch* so với tập đầy đủ, và breakeven cost của `16_costs` **cao hơn** so với sổ `fwd_ret_8h` cùng book.

Không đạt (2) cũng không đạt (3) → nhãn bị đóng, 378 trial được ghi là **đã chi** (chúng đã được chấm), K vẫn 2.514, và Trials table nói rõ nó đã được thử và bị bác.

---

## 5. Một test hôm nay fail, sau khi sửa thì pass

`bots/exness_gold_sess/tests/test_fold_geometry.py` — theo khuôn characterisation test của `test_backtest_grid.py:1-33` (skip khi thiếu `ML4T_OUTPUT_DIR`), nhưng **file này không bị đảo ngược khi fix land — nó chuyển từ đỏ sang xanh và ở lại xanh**.

- `test_three_labels_share_one_fold_ladder` — gọi `assert_variant_folds_are_out_of_sample("exness_gold_sess", "fwd_ret_8h")`; kỳ vọng trả về **8 hàng** (2 variant × 4 fold), mỗi hàng `gap > timedelta(0)` và mọi `gap` **bằng nhau** (= purge của nhãn chính).
  *Hôm nay*: raise `AssertionError` tại `cv_window.py:382` với 14 instant leak → **FAIL**. *Sau fix*: PASS.
- `test_label_timelines_are_identical` — tập `timestamp` duy nhất của ba parquet bằng nhau.
  *Hôm nay*: 4.817 / 3.894 / 4.793 → **FAIL**. *Sau fix*: PASS.
- `test_padding_added_no_labelled_row` — chống hồi quy, **pass cả trước lẫn sau**: số hàng non-null đúng 9.629 / 7.788 / 9.581, và `market_data` digest `6d63567e7b83eea2`. Nó là thứ chứng minh A′ đổi timeline chứ không đổi nhãn.

Lưu ý cho người viết test: **đừng** dùng `modeling_fold_boundaries` trên bot này — nó đi qua `fold_boundary_date` (`cv_window.py:72-76`), hàm này **raise** khi biên mang giờ trong ngày, mà biên của bot này là 09:00/13:00. Dùng thẳng `assert_variant_folds_are_out_of_sample` (trả cột `gap`) hoặc `fold_boundaries`.

Chạy: `uv run --with pytest==9.0.3 python -m pytest bots/exness_gold_sess/tests/test_fold_geometry.py -q -s` với `ML4T_OUTPUT_DIR=~/ml4t/experiments/exness_gold_sess`.

---

Không có con số lợi nhuận nào trong tuyên bố này và không được thêm vào. Thứ duy nhất phase 4/5 được phép sinh ra là: pooled panel IC kèm khoảng HAC, DSR kèm cohort `member_digest` và số trial thực, một holdout chấm đúng một lần, và một breakeven cost so với p90 spread đã đo.

**Next step**: đo điều kiện rẽ nhánh trước mọi thứ khác — so tập ngày duy nhất của `labels/fwd_ret_8h.parquet` khi bỏ `.drop_nulls()` với tập ngày hiện tại; bằng nhau thì thực hiện A′ đầy đủ, khác nhau thì DỪNG và báo cáo (không chạy `04`).

**Read**: `case_studies/utils/cv_window.py:309-387` (`assert_variant_folds_are_out_of_sample`, docstring giải thích vì sao artifact chỉ mang một thang fold) cùng `utils/modeling.py:941-943` và `:952-986`; Tập 1 outline "Chapter 6: The Machine Learning Process" → notebook `02` (purge/embargo suy từ horizon nhãn) và Tập 1 "Chapter 7: Defining the Learning Task" §7.2 Label Engineering + notebook `03_label_methods`.

---
---

# TUYÊN BỐ THỨ HAI — chấp nhận dịch một ngày trên fold 3 và 4

Ngày 2026-09-08 · Mentor declaration · **Cấp ngoại lệ lần thứ hai** cho luật "digest dịch thì revert" của `PRICE_GRID_DECLARATION.md` §5.

Bối cảnh: điều kiện rẽ nhánh ở §2 trên đo ra **FALSE**. Builder DỪNG đúng. Hai ngày mới xuất hiện trong timeline; thí nghiệm cô lập bốn timeline cho thấy **chỉ 2018-09-03** đẩy `train_start` của fold 3 và 4 từ `2018-08-30` sang `2018-08-31`. Không `train_end`, `val_start`, `val_end` nào dịch.

## 1. Phán quyết: **A′, chấp nhận cú dịch**. Không lấy C.

Một điều §1 ở trên không nói và phải nói ra vì nó đổi hình dạng lựa chọn: với phép đo mới (**116** hàng chứ không phải 305) và `|keys(dir_tb_8h) − keys(fwd_ret_8h)| = 0`, **option A nguyên bản chính là phương án GIỮ ĐƯỢC digest**. A neo cả ba nhãn vào key set của `fwd_ret_8h` → timeline 2.451 ngày không đổi → thang fold bất động → `4b1a0239330a0e7d` tái tạo được. Vậy lựa chọn thật hôm nay là **A (giữ digest, mất 116 hàng, giữ nguyên phân kỳ 01/04)** so với **A′ (mất 0 hàng, sửa phân kỳ, dịch một ngày)**.

**Đưa 2018-09-03 vào timeline là ĐÚNG HƠN, không phải kém đúng hơn.** Lịch mà thang fold đếm trên đó phải là **lịch quyết định**, không phải lịch "nhãn có giải được không". Trên Labor Day bar quyết định tồn tại, bot ra quyết định và thực tế giao dịch xuyên ngày đó; chỉ endpoint không giải được vì thiếu bar 10:00. Loại nó ra nghĩa là để **một tai nạn của tape quyết định biên train của fold 3** — đúng thứ `utils/modeling.py:941-943` cấm.

Hệ quả kiểm chứng được: hôm nay thang fold là hàm của horizon 8h **và** của tính đầy đủ của tape; nếu ngày mai bar 10:00 được backfill, thang fold hôm nay **tự dịch mà không ai chạm vào gì** — đó là định nghĩa của một biên không tái tạo được. Sau A′ thang fold là hàm của tape giá cộng luật phiên, bất biến trước mọi thay đổi luật nhãn. Cùng lập luận cho 2018-01-31 (inert, nhưng cùng luật; tính inert của nó không được dùng làm lý do).

**Cú dịch trả giá bằng gì, đo chính xác:**

- Không một cửa sổ được chấm nào dịch: `train_end`, `val_start`, `val_end` bất động trên cả 5 fold; biên holdout và fold 4 vintage bất động. Không hàng nào đổi fold, không hàng nào đổi vai trò train/val.
- Cái dịch: `04:349-351` dựng mask trên panel theo `train_start..val_end`, nên fold 3 và fold 4 mất **đúng hai instant đầu** (2018-08-30 london + ny × 2 kim loại = 4 hàng mỗi fold). Artifact rơi **24.748 → 24.740**. Panel không đổi hàng nào — 2018-09-03 đã nằm trong panel từ bản vá phase 2, nên A′ **không thêm dữ liệu vào fit**, chỉ tịnh tiến đầu cửa sổ một hàng.
- Giá trị 10 cột của fold 3/4 dịch bởi transient khởi tạo của Kalman/HMM/ARIMA. Đây là rủi ro thật duy nhất: EM của HMM trên mẫu lệch hai instant có thể rơi vào cực trị khác. Có `random_state=seed` (`04:475`) và quy ước thứ tự trạng thái (`04:513`) nên vẫn định danh được — nhưng phải **đo**, không được giả định.
- Kéo theo `05` phải chạy lại. Phase 5 (grid và hold) không đọc `04` nên mọi đo đạc của `PRICE_GRID_DECLARATION` và `NY_EXIT_DECLARATION` đứng nguyên.

**Không lấy C**: C đổi 116 hàng và một dòng bookkeeping lấy bằng việc **xoá vĩnh viễn** so sánh horizon và toàn bộ nhánh phân loại — mất bằng chứng để tiết kiệm một lần chạy 04/05 mà hôm nay giá của nó là vài phút CPU trên một registry rỗng. Sau phase 5 tỷ giá đảo ngược.

**Về revert condition §2**: luật "digest dịch → revert" được viết để bắt **chuyển động không giải thích được**. Ở đây chuyển động đã bị cô lập tới một ngày, một biên, hai trường, bằng thí nghiệm bốn timeline. Chuyển động đã giải thích thì **mở lại quyết định**, không tự động revert. Văn bản này là tuyên bố thứ hai đó.

## 2. Chữ ký nghiệm thu — viết thành văn TRƯỚC khi chạy

Đây là thứ biến "digest dịch" từ tai nạn thành tiên đoán. Ghi vào BOT.md **trước**, rồi mới chạy.

**a.** `fold_geometry` sidecar: diff **từng trường** trên 5 fold × 4 biên = 20 trường. Đúng **2** trường được phép đổi: `train_start` của fold 3 và fold 4, cả hai `2018-08-30 → 2018-08-31`. Trường thứ ba đổi → revert.

**b.** Số hàng **24.740** (= 24.748 − 8), **tiên đoán từ panel trước khi chạy** rồi đối chiếu; vẫn 10 cột, 5 fold.

**c.** **"No-op" mới là no-op THEO FOLD**: hàng của fold 0/1/2 phải trùng khít parquet cũ trên cả 10 cột (join theo `(fold, timestamp, symbol)`, `max|diff| = 0.0`). Đây là mệnh đề mạnh nhất còn nói được, thay đúng vai trò mà md5 toàn file từng giữ.

**d.** Fold 3/4: **khai dung sai trước khi nhìn số**. Kỳ vọng cơ học là transient rất nhỏ trên hàng validation. Vượt dung sai → không phải transient, phải điều tra, nghi can số 1 là HMM EM. Riêng HMM: khẳng định thứ tự trạng thái không hoán vị giữa hai lần chạy.

**e.** Ba seal vẫn in **0.00e+00**; purge đo trên lưới vẫn **3-5 slot** mỗi fold × kim loại.

**f.** `inputs`: `labels:fwd_ret_8h` **dịch**; `session_panel` không dịch; `financial 0557461a95579969` và `market_data 6d63567e7b83eea2` **không dịch**.

**g.** `assert_variant_folds_are_out_of_sample` trả **8 hàng**, mọi `gap` dương và **bằng nhau**; `bots/exness_gold_sess/tests/test_fold_geometry.py` chuyển **2 đỏ 1 xanh → 3 xanh** và ở lại xanh.

**h.** Bất biến `02` của §2 nguyên văn (9.629 / 7.788 / 9.581 hàng non-null, ba parquet cùng key set, publish qua được với cột eval null), cộng một dòng đếm mới: **65 khoá panel không có nhãn nào** dưới cả ba luật (9.810 − 9.745) — in ra, không để im lặng.

**i.** `05`: **4.056 hàng validation / 2.028 slot** tái tạo (hàng độn null bị `drop_nulls` loại); **33 cột financial tái tạo IC chính xác** — đây là falsifier cho "thay đổi bị giam trong fold 3/4"; chỉ 10 cột model-based được phép dịch. Nếu một cột lật STOP/PROCEED/REVISE: **ghi lại, không chỉnh ngưỡng**. Ngưỡng đã khai; nhìn số mới rồi sửa ngưỡng là điều kiện hoá trên kết quả.

## 3. Guard, và hai phát hiện

**(i) Vị trí guard.** Builder đúng khi không thêm mù, nhưng lý do đúng không phải "nó chặn C". Lý do đúng là: sửa một stage đang có artifact sống mà không chạy lại tạo ra sidecar không khớp mã sinh ra nó. Điều kiện builder thiếu là *một lần chạy* — lần chạy đó giờ đã được cấp. **Patch một dòng vào cùng lần chạy A′**, đúng chỗ ba anh em đặt (`case_studies/fx_pairs/04_model_based_features.py:534`, sau bảng fold, trước Fold Contract).

Thêm **một assert thứ hai** trong cùng lần chạy: sau khi dựng `timeline` từ label parquet (`04:182-185`), khẳng định key set của nó **bằng** key set development của session panel. Đó là bất biến A′ đặt đúng chỗ nó có thể vỡ, và nó khoá vĩnh viễn phân kỳ 01/04. Hai dòng, đều là assert, không đổi logic, một lần chạy.

**(ii) Phân kỳ 01 vs 04 — khuyết tật ở ĐÂU.** Không phải ở fork. `04:182-188` làm **đúng** quy ước thư viện: `_derive_modeling_splits` (`case_studies/utils/cv_window.py:95-179`, cụ thể `:164-167`) *định nghĩa* fold chuẩn tắc là fold suy từ timeline của label parquet. `01_feasibility_analysis.py:578-580` mới là chỗ lệch quy ước. Khuyết tật thật nằm ở **hợp đồng của thư viện dùng chung**: nó dùng label parquet làm **proxy cho lịch giao dịch**, và proxy đó chỉ đúng khi mọi ngày quyết định đều giải được nhãn. Trên 8 case study kia điều đó đúng nên lỗi tàng hình; lưới hai slot/ngày với endpoint không giải được là thứ làm nó lộ ra.

Sửa ở đâu: **không sửa thư viện** (8 case study phụ thuộc, `require_fold_scoped_temporal_compatibility` khớp theo `fold` id). **Không sửa `01` để đọc label parquet** — phase 1 phải chạy được khi chưa có nhãn nào tồn tại. A′ là cách sửa đúng vì nó biến proxy thành **đẳng thức**: key set label parquet bằng key set lưới quyết định, nên quy ước thư viện và khái niệm đúng trùng nhau, và 01 với 04 khớp **theo cấu tạo** chứ không theo may mắn. Đây là lập luận ủng hộ A′ mạnh nhất trong cả hồ sơ, mạnh hơn cả lập luận 116 hàng.

**Anh em**: không sửa code, nhưng bắt buộc **một phép đo và một dòng ghi chép**. Với mỗi case study có stage 04 (`fx_pairs`, `exness_fx_d1`, `xau_fx_mt5`, `etfs`): tập ngày của label parquet chính có bằng tập ngày của panel hay price grid không. Bằng → proxy vô hại, không làm gì. Khác → cùng lớp khuyết tật tiềm ẩn, **ghi vào BOT.md của bot đó và KHÔNG sửa hồi tố** nếu bot đó đã có hàng đăng ký (biên bằng chứng). `exness_fx_d1` đã biết một phần: ba nhãn lệch 20 instant, qua guard với gap 4 ngày đồng nhất — vô hại, nhưng cùng nguyên nhân, và phải ghi rằng hai bot giải cùng một chỗ theo hai cách với lý do đã ghi.

**(iii) 305 thành 116.** Chấp nhận phép đo; hàng "A nguyên bản" trong §1 sai và phải sửa thành **116 (58 instant × 2 kim loại)**. Phán quyết về A không đổi nhưng **lý do đổi**: trước đây A bị bỏ vì đắt; bây giờ A bị bỏ vì (a) 116 hàng đó là early close — một tập chọn theo lịch biết trước, đúng lớp thiên lệch mà defect A vừa gỡ, và (b) A **không** sửa phân kỳ 01/04, vẫn để thang fold treo vào tai nạn giải nhãn của chính nhãn chính.

## 4. Phải ghi vào BOT.md, để người đọc sau không đọc nhầm

Một hàng Decisions log đề ngày, gọi tên là **tuyên bố thứ hai**, chứa:

1. Điều kiện rẽ nhánh đo FALSE; hai ngày mới và **nguyên nhân trên tape** của từng ngày; bảng bốn timeline nguyên văn (đó là thí nghiệm cô lập, không phải chú thích).
2. **Câu quan trọng nhất**: *A (phương án giữ được digest `4b1a0239330a0e7d`) đã sẵn có và bị bác*, kèm hai lý do ở mục 3(iii). Thiếu câu này, người đọc sau sẽ tưởng cú dịch là bất khả kháng — và đó chính là cách một lựa chọn có chủ ý bị đọc thành tai nạn.
3. Bằng chứng không điều kiện hoá trên kết quả: `run_log/` rỗng, không `registry.db`, manifest md5 `1aeeb63842c6e6d69a41ff140332d665`, **0 mô hình đã fit**, K = 2.514 và cổng `dir_tb_8h` đều khai **trước** lần fit đầu tiên. Quyết định ra trên **bảng fold**, chưa từng nhìn một output mô hình nào vì chưa có cái nào tồn tại. Và chữ ký nghiệm thu ở mục 2 được viết **trước** khi chạy lại — ghi rõ thứ tự đó.
4. Digest trước và sau: `04 4b1a0239330a0e7d` thành mới; ba nhãn `d6487f7333a683cc / f2861a79ee3c6cc2 / 1eb71e01111c53b3` thành mới; bất động: `market_data 6d63567e7b83eea2`, `financial 0557461a95579969`. Cộng `24.748 → 24.740`.
5. Sửa hàng "Trading problem / Walk-forward": fold 3 `train_start` bằng **2018-08-31 trên cả `01` và `04`** kể từ A′, và nói rõ **trước** A′ hai stage bất đồng (01 ghi 2018-08-31, 04 đóng dấu 2018-08-30) — hàng hiện tại đang ghi số của `01` như thể là số của cả bot.
6. Sửa hàng phase 4: thay "A′ NOT executed" bằng kết quả; ghi `05` đã chạy lại và cột nào (nếu có) lật hạng.
7. `PRICE_GRID_DECLARATION.md` §5 (digest mới, và ghi rằng ngoại lệ §5 được cấp **lần thứ hai**, có văn bản) và §4 (K = 2.514, sửa 3.204).
8. Trials table: **K vẫn 2.514, không nhãn nào bị bỏ**, và một dòng nói rõ việc chạy lại 02/04/05 **không** sinh trial nào — để không ai cộng nhầm.
9. Ghi chép cross-bot ở mục 3(ii): quy ước thư viện dùng label parquet làm proxy lịch giao dịch, kết quả phép đo trên các anh em, và lý do bot này khác.

---

**Next step**: viết chữ ký nghiệm thu 2a-2i thành văn vào BOT.md **trước**, rồi mới sửa `case_studies/exness_gold_sess/02_labels.py:693`, chạy `02`, thêm hai assert vào `04_model_based_features.py`, chạy `04` **đúng một lần**, rồi `05`.

**Read**: `case_studies/utils/cv_window.py:95-179` (nơi thư viện *định nghĩa* fold chuẩn tắc theo label parquet — gốc của phân kỳ 01/04) và `utils/cv_splits.py:414-441` (`train_start = _ts(train_idx[0])`, đếm ngược theo chỉ số, nên một hàng thêm vào **trong** cửa sổ dịch đúng biên đầu và không dịch gì khác — cơ chế của toàn bộ vụ này); Tập 1, Ch06 notebook `02` (purged walk-forward: purge và embargo suy từ horizon nhãn, không từ tính khả giải của nhãn).
