# PHASE 1 SPEC — `exness_gold_sess` (bản mentor, để builder thực thi)

Ngày: 2026-09-07 · Mentor review + đặc tả phase 1 · Repo root `D:/05_Quant/machine-learning-for-trading` (branch `exness-bots`)

Tài liệu này **thay thế mọi con số ước lượng** trong `bots/exness_gold_sess/BOT.md` và `bots/assets/XAUUSD.md` §1–§11 ở những chỗ nói rõ là "ghi đè". Mọi khẳng định đều có `path:line` hoặc PDF volume + outline entry.

---

## 0. Điều kiện tiên quyết — đọc trước khi gõ dòng lệnh đầu tiên

**Phát hiện quan trọng nhất của review này: con số "H1 chỉ có từ 2022-10-25" là artefact của loader, không phải giới hạn của tài khoản.**

| Nguồn | XAUUSD H1 | Cách tải |
|---|---|---|
| `data/mt5/history_depth.json` (dùng bởi `exness_fx_d1`) | 22,838 nến, **2022-10-25** -> 2026-09-04 (`bots/assets/XAUUSD.md:183`) | `bots/_shared/mt5_loader.py:258-281` `_copy_rates_chunked` -> `copy_rates_range` theo chunk lịch |
| `experiments/xau_fx_mt5/data/coverage.json` (bot `xau_fx_mt5`, **cùng demo login 206539306, cùng server Exness-MT5Trial7, cùng tuần**) | **55,780 nến, 2017-03-06** -> 2026-08-21 | `experiments/xau_fx_mt5/data/build_panel.py:333-352` `_copy_rates_deep` -> `copy_rates_from` **đếm số nến, epoch seconds** |

Docstring của chính hàm đó viết rõ nguyên nhân (`build_panel.py:335-338`): *"`copy_rates_range` in calendar chunks (the shared loader) returned only the bars the terminal had already synchronised (about 15 months of H1 on 2026-09-06), while a count-based request makes the terminal pull the older history from the server."* Lý do được ghi trong `bots/xau_fx_mt5/BOT.md:672`; **không** được ghi trong `bots/exness_gold_sess/BOT.md` hay `bots/assets/XAUUSD.md` §12.

**Hành động bắt buộc, trước mọi thứ khác (task B0):** tải lại H1 cho `XAUUSD` và `XAGUSD` bằng đường count-based (`copy_rates_from` + epoch seconds, `_epoch_utc`), ghi lại độ sâu thực đo được cho **cả hai** mã, cập nhật `bots/assets/XAUUSD.md` §12 và `XAGUSD.md` §12, và chỉ sau đó mới chốt `universe.history_start` và khối `evaluation`. XAGUSD **chưa từng được đo bằng đường này ở bất kỳ bot nào** — độ sâu H1 của bạc là ẩn số duy nhất còn lại của phase 1.

Toàn bộ hình học fold ở §2.c' dưới đây có hai nhánh vì lý do này.

---

## 1. Design review theo 9 phase

### 1.1 Trạng thái của `exness_gold_sess` (đọc từ `bots/exness_gold_sess/BOT.md`, 2026-09-05)

BOT.md **không stale** (ngày ghi khớp trạng thái: chưa có case study, phases 1–7 để trống) nên bot đủ điều kiện được review.

| Phase | Kết luận | Bằng chứng / nguồn đóng gap |
|---|---|---|
| 0 Environment | **gap** | `BOT.md:47`: `uv sync` chưa xong, thiếu `ml4t.*`. Đường đã chứng minh: WSL2 `~/ml4t` của `exness_fx_d1` (`bots/exness_fx_d1/BOT.md:78`) — dùng lại nguyên môi trường đó, không dựng mới. `.claude/skills/ml4t-quant-bot-mentor/SKILL.md:78` (rule 8). |
| 1 Strategy & learning task | **gap (toàn bộ)** | Bảng "Trading problem" trống ở `Walk-forward`, `Holdout`, `Assumed costs`, `Position limits` (`BOT.md:29-32`); Hypothesis và Kill criteria vẫn là văn bản mẫu (`BOT.md:36-41`); chưa có `case_studies/exness_gold_sess/`. Đóng bằng: §2 của tài liệu này + `bot-template.md:26-58` (Route B). |
| 1b **Mâu thuẫn nội bộ** | **gap** | `BOT.md:24` ghi mốc quyết định là **08:00 / 13:00 UTC (giờ mở phiên)**, còn `bots/assets/XAUUSD.md:22-24` ghi "**sau block mép phiên 30 phút**". Hai file mô tả hai thời điểm khác nhau cho cùng một quyết định. §2.a/§2.b dưới đây chốt một quy tắc duy nhất. |
| 1c **`SESSION_FILTER: overlap` thừa** | **gap** | `BOT.md:26` và `XAUUSD.md:122` khai `london, ny, overlap`. Với hai snapshot đã khai, `overlap` (13:00–17:00 UTC mùa đông, `bots/assets/XAUUSD.md:79`) **chứa đúng mốc NY và không chứa mốc London** -> tập hàng của filter `overlap` **trùng khít** filter `ny`. Một spec trùng = một trial vô nghĩa trong DSR (`bot-portfolio-exness.md:96`). Bỏ `overlap`. |
| 2 Point-in-time data | **gap** | Chưa có `tests/test_lookahead.py`, chưa có data-quality test; độ sâu H1 chưa chốt (§0); spread chưa đo (`XAUUSD.md:190`, `XAGUSD.md:188`). Mẫu: `bots/exness_fx_d1/tests/test_lookahead.py` + `test_data_quality.py` (`bots/exness_fx_d1/BOT.md:80`). |
| 3 Features | **n.a.** (chưa mở) — nhưng có tiền điều kiện: register phải khai **trước** khi đọc IC (`roadmap.md:118-120`, Ch20). |
| 4 Models | **n.a.** |
| 5 Backtest & inference | **n.a.** |
| 6 Portfolio/costs/risk | **n.a.** |
| 7 Holdout | **n.a.** — nhưng `evaluation.holdout_*` phải khai **trong phase 1** (`roadmap.md:72-73`). |
| 8 Deployment | **gap** (đúng như BOT.md ghi) | `BOT.md:55`: adapter chung đã có và đã test (18 tests), `deploy/` của bot chưa tạo. Magic đề xuất `260903` (`BOT.md:65`). |
| 9 Monitoring | **gap, nặng hơn BOT.md ghi** | `ls bots/_shared/` chỉ có `costs_mt5.py, mt5_broker.py, mt5_loader.py, sessions.py, testing/` — **`bots/_shared/monitor/` chưa tồn tại**, dù `bot-portfolio-exness.md:40` và `:123-128` khai nó là nơi chứa breaker cấp tài khoản cắt mọi bot cùng lúc. Chưa bot nào trong repo có `monitor/`. |

### 1.2 Nhất quán liên bot (`exness_fx_d1`, `xau_fx_mt5`) — nơi hai bot giải cùng một bài toán theo hai cách mà **không** ghi lý do

| # | Vấn đề | Bot A | Bot B | Phải làm gì |
|---|---|---|---|---|
| **X1** | **Độ sâu H1 của cùng một mã trên cùng một tài khoản** | `exness_fx_d1`: 2022-10-25 (`bots/assets/XAUUSD.md:183`), và chính vì thế đã **loại H1 khỏi thiết kế** (`bots/exness_fx_d1/BOT.md:98`, "H1 bars (3.9 years of history)" nằm trong cột *alternatives rejected*) | `xau_fx_mt5`: 2017-03-06, 55,780 nến (`experiments/xau_fx_mt5/data/coverage.json`) | §0. Đây là mâu thuẫn nghiêm trọng nhất: một quyết định thiết kế của `exness_fx_d1` dựa trên một con số mà bot anh em đã chứng minh là sai. Ghi vào Decisions log của **cả hai** bot. |
| **X2** | **Phiên là lưới quyết định hay là feature** | `exness_gold_sess`: phiên = lưới quyết định (2 snapshot) + `SESSION_FILTER` | `xau_fx_mt5` (cùng XAUUSD, cùng broker): `decision.cadence: hourly_bar_close`, `case_studies/xau_fx_mt5/config/setup.yaml:44-45` — *"sessions enter as FEATURES and as cost buckets, not as a decision grid"*, `no_trade_buckets: [other]` | Không sai — nhưng hai bot **cùng giao dịch XAUUSD trên cùng một tài khoản** với hai thiết kế đối lập và không file nào tham chiếu file kia. Phải ghi chéo (như `exness_fx_d1/BOT.md:32` đã làm với holdout), và phải ghi ràng buộc vận hành: hai bot cùng ôm XAUUSD chia chung margin, nên breaker cấp tài khoản (`bot-portfolio-exness.md:123-128`) là bắt buộc **trước** khi bot thứ hai lên demo. |
| **X3** | **Cách đo spread** | `exness_fx_d1`: tick `COPY_TICKS_INFO` 30 ngày, `bots/_shared/costs_mt5.py:202` `measure_spreads` | `xau_fx_mt5`: trường `spread` **theo nến** 2017-03 -> 2025-08, cộng quy tắc điền nến `spread == 0` theo point-in-time (`case_studies/xau_fx_mt5/config/setup.yaml:92-95`; 1,015 nến spread 0 mỗi mã, tới 2023-02-03) | Hai phương pháp trả lời hai câu hỏi khác nhau và **cả hai đều cần**: tick = chi phí *hiện tại* (cho live), per-bar = chi phí *trong mẫu* (cho backtest 2017->2025). Bot vàng phải làm cả hai và đối chiếu trên phần chồng lấn. Cảnh báo cụ thể: nếu chỉ dùng trường per-bar mà bỏ quy tắc điền, giai đoạn 2017-03 -> 2023-02 sẽ được định giá **miễn phí**. |
| **X4** | **Quy ước holdout** | `exness_fx_d1`: 12 tháng cuối của lịch sử đã có, `2025-09-01 -> 2026-08-31` (`setup.yaml:347-348`) | `xau_fx_mt5`: **một** holdout là *tape tương lai* từ `2026-08-22`, `holdout_end: null` điền một lần khi đủ `holdout_min_weeks: 12` (`setup.yaml:362-365`), và ghi rõ đánh giá khác về cửa sổ của bot kia (`bots/exness_fx_d1/BOT.md:32`) | Bot vàng phải **chọn một và ghi lý do**. Xem §2.c'. Kèm theo: `xau_fx_mt5` khai `v9_contaminated_window: ['2025-08-21','2026-08-21']` (`setup.yaml:366`) — bot vàng phải kiểm tra (read-only) `bots/exness_gold_sess/legacy_v9_continuum/config/symbols.py` xem v9 có giao dịch XAUUSD/XAGUSD không, rồi khai cờ tương ứng. |
| **X5** | **Magic number** | `exness_fx_d1` đề xuất `260901` (`BOT.md:106`); `exness_gold_sess` đề xuất `260903` (`BOT.md:65`); legacy `202500` | `xau_fx_mt5/BOT.md`: **không có magic nào được ghi** (`grep magic` = 0 hit) | Cần **một bảng magic duy nhất** trong `bots/README.md` + một test khẳng định tính duy nhất. Xem §3.2. |
| **X6** | **`mapping.class` / long-short** | `exness_fx_d1` giữ `long_short_rank_rebalance` (`setup.yaml:78`) trên 5 mã, và phát hiện stage fork từ `fx_pairs` **không truyền** `long_short` vào signal (`BOT.md:135`, open question 19) -> 1,068 trial chạy long-only trong khi prose mô tả long-short | `xau_fx_mt5` cũng khai `long_short_rank_rebalance` (`setup.yaml:76`) | Bot vàng có **2 tài sản**: `mapping.class` cross-sectional là sai từ gốc (§2.d). Và bất kể chọn gì, stage fork phải **truyền hướng vào dict `signal`** một cách tường minh, kèm test đọc `weights.parquet` xác nhận có trọng số âm — đúng bài học `exness_fx_d1/BOT.md:135`. |
| **X7** | **`periods_per_year`** | `exness_fx_d1`: 252 (`setup.yaml:350`) | `xau_fx_mt5`: 6084 = 117.3 nến H1/tuần x 52 (`setup.yaml:368-370`) | Không mâu thuẫn (lưới khác nhau) nhưng mỗi bot phải ghi phép tính. Bot vàng: xem §2.c' khối `evaluation`. |

**Gap quan trọng nhất của bot này ngay lúc này**: `case_studies/exness_gold_sess/config/setup.yaml` chưa tồn tại, và độ sâu H1 chưa được đo lại (§0). **Bước nhỏ nhất đóng nó**: chạy tải H1 count-based cho XAUUSD + XAGUSD, ghi độ sâu vào hai file `bots/assets/*.md` §12, rồi viết `setup.yaml` theo §2.

---

## 2. Đặc tả `case_studies/exness_gold_sess/config/setup.yaml` — từng khối

Route: **Route B** (`bot-template.md:26-58`), như `exness_fx_d1` (`bots/exness_fx_d1/BOT.md:96`): loader và đồng hồ quyết định khác `fx_pairs`, và id riêng giữ registry tách khỏi baseline đã công bố. Kiểm tra trước khi copy bất cứ thứ gì:

```bash
uv run python -c "from utils.paths import get_case_study_dir; print(get_case_study_dir('exness_gold_sess'))"
```

### 2.0 `universe`

```yaml
strategy_id: exness_gold_sess
setup_version: v1

universe:
  symbols: [XAGUSD, XAUUSD]      # tên trần; suffix 'm' chỉ nằm trong mt5_loader.SYMBOL_MAP
  n_assets: 2
  history_start: '<ĐO ĐƯỢC ở B0, là max của hai mã>'
```

Lý do: quy ước tên trần đã chốt ở `bots/exness_fx_d1/BOT.md:95` (đổi loại tài khoản không được re-hash mọi label). `history_start` là ngày đầu tiên **cả hai mã** có lưới H1 dày — đo bằng `build_panel.dense_history_start` (`experiments/xau_fx_mt5/data/build_panel.py:789`, ngưỡng `min_bars_per_week=100`), không đoán. Tiền lệ: `exness_fx_d1` phải dời `history_start` lên 2017-03-01 vì 2014–2016 chỉ có một nến H4/ngày (`setup.yaml:16-19`).

### 2.a Hai quyết định mỗi ngày: **mức 2** (một panel + `SESSION_FILTER`), không phải mức 1

**Khuyến nghị: mức 2.** Lý do, theo thứ tự sức nặng:

1. **Đây là thiết kế đã ghi.** `bot-portfolio-exness.md:85-89` gán mức 2 cho *bot theo phiên* ("Mức 1 · Bot daily"); `bots/assets/XAUUSD.md:122` và `BOT.md:26` đã khai `SESSION_FILTER`. Đổi sang mức 1 là đổi thiết kế đã ghi và phải ghi lý do.
2. **Số trial huấn luyện không nhân lên.** Mức 1 = 2 experiment => mọi model được fit hai lần => **gấp đôi** số training run và số prediction set. Mức 2 = một population huấn luyện phục vụ cả các filter; chỉ tầng **backtest** nhân lên. Với tiền lệ `exness_fx_d1` (534 prediction set -> 1,068 backtest, `BOT.md:152-153`), khác biệt này là hàng trăm giờ CPU.
3. **K của DSR đếm được chính xác và tự động.** Một registry => `K` = số member của population baseline, đúng cách `exness_fx_d1` đếm (`BOT.md:132`: *"DSR at the raw trial count K = 1,068 = the members of `exness_fx_d1:equal-weight-baselines`"*). Mức 1 buộc phải cộng tay K qua hai registry, và lịch sử cho thấy cộng tay là chỗ K bị "quên".
4. **Nhãn vẫn point-in-time.** Cảnh báo của `bot-portfolio-exness.md:82-83` ("nhãn tính từ đóng nến không dùng được cho quyết định lúc London open") **được thoả mãn tự động** ở mức 2: mỗi hàng quyết định có `timestamp` riêng và nhãn forward riêng tính từ chính `timestamp` đó. Khoá `(symbol, timestamp)` vẫn duy nhất — điều kiện `LabelCatalog.publish` bắt buộc (`case_studies/research/labels.py:147-150`).

**`decision.snapshot` xử lý ra sao khi nó là một giá trị đơn?** Đã kiểm tra: `decision.snapshot` **không được đọc bởi bất kỳ module dùng chung nào** (`grep -rn "snapshot" case_studies/utils/ utils/` = 0 hit) — nó là khoá tài liệu mà stage của chính bot đọc. Ngược lại `decision.cadence` / `cadence_by_label` **có** được đọc (`case_studies/utils/backtest_loaders.py:2172-2177, 2193-2205`). Vì vậy:

```yaml
decision:
  cadence: session                       # được backtest_loaders đọc
  snapshot: session_open_plus_1h         # quy tắc, không phải một giờ cụ thể
  snapshots:                             # do stage của bot đọc
    london: {venue_session: london,   offset_minutes: 60}
    ny:     {venue_session: new_york, offset_minutes: 60}
  edge_block_minutes: 30                 # = bots/_shared/sessions.py:64 EDGE_MINUTES
  execution_delay: next_bar_open
  session_close_tolerance_minutes: 60    # một nến H1
  session_calendar: metals_globex_utc    # KHÔNG dùng CME_FX
  no_trade_buckets: [rollover, other]
  session_filter_values: [london, ny]    # 'overlap' bị bỏ, xem §1.1c
  server_clock:
    utc_offset_minutes: 0
    measured_at: "<ĐO LẠI>"
    follows_dst_of: null
```

**Vào spec hash bằng cách nào.** Đã truy vết cơ chế: `plan_backtests(signal=...)` -> `Study.strategy()` -> `Strategy._build_spec` -> `build_backtest_spec(..., signal=self.signal)` (`case_studies/research/strategy.py:353-359`) -> `resolved_signal = deepcopy(signal)` đưa **nguyên vẹn** vào `strategy_spec["signal"]` (`case_studies/utils/backtest_presets.py:457-462`) -> `serializable_backtest_spec` -> `backtest_hash_from_parts` (`case_studies/research/execution.py:285-291`). Nên: **đặt `"session_filter": "london"` vào chính dict `signal`** => mỗi filter một hash riêng, không đụng nhau.

**Guard bắt buộc (đây là chỗ dễ sai nhất).** `build_target_weights_from_config` dispatch theo `config["method"]` và **bỏ qua khoá lạ** (`case_studies/utils/signals.py:537-560`). Nghĩa là nếu builder chỉ thêm khoá vào dict mà không lọc hàng, các filter sẽ có **hash khác nhau và kết quả giống hệt nhau** — đúng cái bẫy `.claude/skills/ml4t/references/case-studies.md:130-131` gọi tên: *"A new hash is a cache key, not proof an override reached the model."* Bắt buộc: (a) lọc predictions theo cột `session` **trong stage** trước khi dựng weight, và (b) một test khẳng định `weights.parquet` của filter `london` và `ny` khác nhau ở > 0 hàng.

**Đếm trial:** `K = |SESSION_FILTER| x |labels| x |prediction sets| x |signal specs|`. Với 2 filter + 1 book gộp = 3. Xem §2.d để giữ K không nổ.

### 2.b Mốc quyết định, block mép phiên 30 phút, và DST trên server UTC+0

**Quy tắc duy nhất, dùng cho cả hai phiên và cả hai mùa:**

> Mốc quyết định = **đóng nến H1 đầu tiên có giờ đóng >= giờ mở phiên + 30 phút**. Trên lưới H1 UTC, điều đó luôn là nến `[open, open+1h)` => **quyết định tại open + 60 phút**. Khớp lệnh tại open của nến kế tiếp (`next_bar_open`), tức đúng chính khoảnh khắc đó.

Vì sao quy tắc này thắng ba biến thể đang ghi rải rác:

- Nó **thoả block mép phiên 30 phút** mà `bots/assets/XAUUSD.md:22` yêu cầu và `bots/_shared/sessions.py:64` (`EDGE_MINUTES = 30`) hiện thực hoá qua cờ `edge_open` (`sessions.py:332-334`) — quyết định rơi ra ngoài cờ đó.
- Nó **nuốt luôn cửa sổ tin 13:30** mà `XAUUSD.md:24` đề xuất làm biến thể thứ hai ("hoặc 14:00 sau cửa sổ tin 13:30"): NY mở 13:00 UTC mùa đông => quyết định 14:00 UTC, sau tin; NY mở 12:00 UTC mùa hè => quyết định 13:00 UTC, tin ra lúc 12:30 cũng đã nằm trong nến quyết định. **Một quy tắc thay hai biến thể => tiết kiệm một trial.**
- **DST**: server UTC+0 quanh năm và **không** theo DST New York (đo `ServerClock.measure`, `bots/assets/XAUUSD.md:180`; xác nhận độc lập từng năm 2017–2026 ở `bots/xau_fx_mt5/BOT.md:383-386`). Nhưng **phiên** thì có DST. Cách xử lý đúng: **không hard-code giờ UTC**. `bots/_shared/sessions.py:276-282` định nghĩa `SESSIONS` bằng `ZoneInfo` theo giờ *địa phương của venue* (`london` 08:00–17:00 Europe/London, `new_york` 08:00–17:00 America/New_York), nên `session_flags_frame(timestamps, clock)` (`sessions.py:344`) tự trả về 08:00 UTC mùa đông / 07:00 UTC mùa hè cho London mà không cần bảng. `bot-portfolio-exness.md:59` nói đúng điều này: *"Định nghĩa theo UTC, DST xử lý bằng thư viện múi giờ... Không hard-code giờ server."* Bảng giờ trong `bots/assets/*.md` §2 chỉ là tài liệu người đọc, **không** là nguồn cho code.

**Điểm cuối nhãn (giờ đóng phiên) trên broker này** — phải khẳng định bằng dữ liệu, không giả định:

- London đóng 17:00 London = 17:00 UTC (đông) / 16:00 UTC (hè) -> nến đóng đúng giờ đó tồn tại.
- New York đóng 17:00 New York = 22:00 UTC (đông) / 21:00 UTC (hè). Trên tài khoản này vàng **nghỉ 22:00–23:00 UTC dưới EST và 21:00–22:00 UTC dưới EDT** (`bots/xau_fx_mt5/BOT.md:54-56`), khớp với giờ đo bằng H1 8 tuần gần nhất (`XAUUSD.md:188`: mon–thu `00:00–21:00, 22:00–24:00`, đo trong mùa hè). Nghĩa là nến cuối trước giờ nghỉ đóng **đúng** giờ đóng phiên NY ở cả hai mùa. Quy tắc an toàn: **điểm cuối nhãn = đóng nến H1 cuối cùng có giờ đóng <= giờ đóng phiên**, `session_close_tolerance_minutes: 60` (một nến), và stage `01` phải khẳng định: mỗi hàng quyết định có điểm cuối, không điểm cuối nào đóng *sau* giờ đóng phiên, và số phiên có khoảng cách > tolerance (ngày lễ, nghỉ sớm) < 1 % và bị **loại, không dùng nến cũ** — nguyên văn quy tắc `exness_fx_d1` (`setup.yaml:46-51`, sự cố 2018-01-31).
- `session_calendar`: **không** dùng `CME_FX`. `exness_fx_d1` giữ `CME_FX` vì nó cần đúng cú rollover 17:00 New York cho lưới daily (`setup.yaml:54-60`). Bot vàng ra quyết định *trong* phiên, không tại rollover; dùng lịch riêng suy từ giờ giao dịch đo được (mẫu: `xau_fx_mt5` đặt `session_calendar: mt5_24x5_utc`, `setup.yaml:58`). `evaluation.calendar` vẫn để `FX` (lịch mà `utils/cv_splits` đếm cửa sổ train/val trên đó).

### 2.c Nhãn

**Nhãn chính — `fwd_ret_8h`** (lợi nhuận từ đóng nến quyết định đến đóng phiên):

Với quy tắc "open + 1h", horizon đúng bằng **8 nến H1 ở cả hai phiên và cả hai mùa** (phiên 9 giờ, vào ở giờ thứ nhất, ra ở giờ thứ chín). Vì vậy hãy đặt tên số học `fwd_ret_8h` chứ đừng đặt `fwd_ret_sess_close`: `resolve_label_horizon` rơi về `buffer` khi không có `horizons`, và nasdaq đã ghi lại đúng cái bẫy này (`case_studies/nasdaq100_microstructure/config/setup.yaml:604-607`). Vẫn khai `horizons` tường minh.

- **Ai dựng**: `ml4t.engineer.labeling.fixed_time_horizon_labels` (import ở `07_defining_the_learning_task/03_label_methods.py:69`), hoặc đơn giản là `log C[t+8] - log C[t]` dịch theo `symbol` như `xau_fx_mt5_d1/02_labels` làm (`bots/xau_fx_mt5/BOT.md:398`). Niêm phong tại endpoint; hàng nào endpoint không đúng đóng phiên (nghỉ sớm, sự cố) thì **loại và báo cáo**.
- Ghi nhớ: `case_studies/research/labels.py` là **catalog** (`LabelDefinition`, `LabelCatalog.publish`), không phải thư viện dựng nhãn. Nó áp các ràng buộc bạn phải thoả: cột `symbol/timestamp/<name>`, khoá không null, khoá duy nhất, giá trị numeric, và với nhãn phân loại thì cột `continuous_eval_label` phải nằm **trong cùng frame** (`labels.py:143-159`).

**Biến thể 1 — `fwd_ret_24h`** (giữ qua đêm, cộng swap):

- Nhãn = **lợi nhuận giá** đến mốc quyết định cùng phiên ngày hôm sau (24 giờ = 2 slot của lịch quyết định). **Swap không nằm trong nhãn.** Lý do (khác với chữ trong `bot-portfolio-exness.md:105`, và đây là một lệch có ghi lý do): swap là chi phí giữ lệnh theo đêm mà mô hình chi phí phần trăm của engine không diễn đạt được — đúng nhận định trong `case_studies/utils/backtest_loaders.py:2283-2285` ("The swap is a holding cost per night, not a per-crossing cost the engine can express, so it is left to the cost stage"). Nhét swap vào nhãn khiến nhãn phụ thuộc loại tài khoản: đọc lại swap trên tài khoản thật (user decision 5 của `exness_fx_d1`, `BOT.md:141`) sẽ **re-hash toàn bộ label, feature, prediction**. Khai swap ở `costs.swap` và định giá ở `16_costs`, đúng như hai bot kia đã làm.
- Tên: dùng `fwd_ret_24h` chứ không `fwd_ret_1d`, để không lẫn với `fwd_ret_1d` của `exness_fx_d1` (= 1 phiên daily) và của `xau_fx_mt5` (= 24 nến H1).

**Biến thể 2 — `dir_tb_8h`** (triple-barrier trong phiên):

- **Ai dựng**: `ml4t.engineer.labeling.triple_barrier_labels` với `ml4t.engineer.config.labeling.LabelingConfig.triple_barrier(upper_barrier=..., lower_barrier=..., max_holding_period=8, side=1)` — mẫu gọi đầy đủ ở `07_defining_the_learning_task/03_label_methods.py:508-521`, `calculate_uniqueness=True` để có sample weight. **Khuyến nghị dùng bản ATR**: `atr_triple_barrier_labels` (import `03_label_methods.py:66`, ví dụ `:690-698`) vì rào cản phải co giãn theo biến động — vàng và bạc có vol rất khác nhau (`bots/assets/XAGUSD.md:66`, "beta cao hơn"), rào cản % cố định sẽ khiến bạc chạm rào liên tục còn vàng không bao giờ chạm, và cùng một % năm 2022 khác hẳn năm 2026.
- Đây là nhãn **phân loại** => phải khai `labels.classification_eval_label: {dir_tb_8h: fwd_ret_8h}`. Đây là **một trong bốn khoá pinned** (`.claude/skills/ml4t/SKILL.md:36-39`): khai một lần lúc tạo case study khi `run_log/` còn rỗng, sau đó không sửa.
- `max_holding_period=8` phải được **cắt tại biên phiên** chứ không chỉ đếm 8 nến: ngày nghỉ sớm, 8 nến sẽ vượt qua giờ đóng. Đặt thời gian hết hạn = min(8 nến, đóng phiên), rồi khẳng định trong test.

**`labels.rebalance_step` trong một case study Route B mới:**

- Khoá này **pinned** và **được đọc từ bản repo ngay cả khi có `ML4T_OUTPUT_DIR`** (`.claude/skills/ml4t/references/case-studies.md:139-151`). Vì `case_studies/exness_gold_sess/` là mới tinh và `run_log/` rỗng, builder được quyền **tác giả** nó ngay lúc tạo — và sau đó không được sửa nếu registry đã có hàng (`SKILL.md:36-39`: sửa cần một run log rỗng, nếu không hai phương pháp trộn vào một registry).
- Giá trị được **suy từ cadence và horizon**, không suy từ dữ liệu ("authored from cadence and horizon, never inferred", `case-studies.md:141`). Lịch quyết định có **2 slot/ngày**, nên:

```yaml
  rebalance_step:
    fwd_ret_8h:  1     # 8h = một slot
    fwd_ret_24h: 2     # 24h = đúng 2 slot
    dir_tb_8h:   1
```

- **Cảnh báo phải xử lý và ghi lại**: quyết định London 09:00 UTC đóng lúc 17:00 UTC, trong khi slot NY 14:00 UTC nằm *bên trong* khoảng giữ đó. Trên panel gộp (`SESSION_FILTER` = cả hai), các kỳ giữ **chồng lấn 3 giờ**, đúng thứ mà `rebalance_step` sinh ra để tránh (`exness_fx_d1/setup.yaml:361-362`). Cách xử lý trung thực, không phải bịt: (i) giữ `step: 1` và coi hai phiên là **hai sleeve của một sổ** (engine cộng trọng số); (ii) `02_labels` phải báo cáo `N_eff` bằng `measure_n_eff` (`07_defining_the_learning_task/03_label_methods.py:876`) như `exness_fx_d1` đã làm (`BOT.md:79`: "N_eff ratios 1.000 / 0.200 / 0.048"); (iii) mọi suy luận IC dùng HAC lag >= độ chồng lấn (`compute_ic_hac_stats`, mẫu ở `case_studies/xau_fx_mt5_d1/config/setup.yaml:313-319`).
- **Kiểm tra bắt buộc ở phase 5**: khi `SESSION_FILTER` chỉ giữ một phiên, lịch hiệu dụng còn 1 slot/ngày, nên `step: 2` của `fwd_ret_24h` sẽ nhảy **hai ngày** thay vì một. Đây đúng lớp lỗi nasdaq đã dính và ghi lại: *"The values did not change; what they count did"* (`nasdaq100_microstructure/config/setup.yaml:632-641`). Builder phải chạy `resolve_rebalance_timestamps` trên lưới đã lọc và in ra số ngày giữa hai lần rebalance trước khi chạy sweep.

### 2.c' Hình học fold và holdout

**Nhánh A — nếu B0 xác nhận H1 từ ~2017 cho cả hai mã (kỳ vọng cao cho XAU, chưa biết cho XAG):**

```yaml
evaluation:
  n_splits: 4
  train_size: P3Y
  val_size: P1Y
  holdout_start: '2025-09-01'
  holdout_end: '2026-08-31'
  calendar: FX
  periods_per_year: 504        # 2 slot quyết định/ngày x 252
```

Lý do từng số:

- **4 x (P3Y, P1Y)**: lặp đúng hình học của `exness_fx_d1` (`setup.yaml:344-348`, lý do ở `BOT.md:100`) để hai bot **so sánh được** và vì 8.5 năm development chứa vừa khít 4 fold lùi từng năm. `xau_fx_mt5` trên cùng mã, cùng server, cũng chốt 4 x P3Y/P1Y (`setup.yaml:359-361`) — ba bot cùng hình học là một nhất quán đáng giữ.
- **Holdout = 12 tháng cuối lịch sử đã có (quy ước `exness_fx_d1`)**, không phải tape tương lai kiểu `xau_fx_mt5`: bot này chưa có tape shadow đang chạy, và chờ 12 tuần (`holdout_min_weeks`) sẽ chặn phase 7 vô thời hạn. **Nhưng** phải khai kèm cờ nhiễm bẩn: kiểm tra read-only `legacy_v9_continuum/config/symbols.py` xem v9 có giao dịch hai mã kim loại này không; nếu có, khai `legacy_contaminated_window: ['2025-09-01','2026-08-31']` theo mẫu `xau_fx_mt5` (`setup.yaml:366`) và ghi rằng fold gần nhất chỉ dùng để **xác nhận**, không dùng để chọn.
- **`periods_per_year: 504`**: sổ chỉ giao dịch một phiên vẫn nằm trên lưới 504 slot (nằm tiền mặt ở slot kia), nên annualise trên 504 là đúng cho mọi filter. Đây là phép tính, phải ghi vào comment như `xau_fx_mt5` ghi 6084 (`setup.yaml:368-370`).
- **`allocator_lookback: 126`** (= 63 ngày giao dịch x 2 slot), tương ứng 63 daily của `exness_fx_d1` (`setup.yaml:75`) và 1512 H1 của `xau_fx_mt5` (`setup.yaml:73`).

**Nhánh B — nếu H1 thật sự chỉ có từ 2022-10-25 cho một trong hai mã:**

Development = 2022-10-25 -> 2025-08-31 ~ **2.85 năm**. Không thể lắp 4 x (3Y+1Y). Lựa chọn, theo thứ tự khuyến nghị:

1. **Chuyển mã thiếu lịch sử ra khỏi `universe` và giữ nó làm feature.** Nếu XAGUSD là mã thiếu: universe = `[XAUUSD]`, và bạc vào bằng **tỷ số vàng/bạc trên lưới D1** — đúng thứ `bots/assets/XAGUSD.md:128` và `XAUUSD.md:129` đã đặt tên là feature hàng đầu. Đây là mất mát nhỏ nhất và trung thực nhất.
2. Xây bot trên **lưới H4** (16,148 nến từ 2014-01, `XAUUSD.md:182`) với mốc quyết định xấp xỉ phiên: nến H4 mở 08:00 UTC và 12:00 UTC tồn tại (`exness_fx_d1/BOT.md:165`, open question 8), 13:00 thì không. Đổi lại: mốc quyết định lệch khỏi "open + 30 phút" và block mép phiên biến mất -> phải ghi là một thiết kế khác, không phải bot theo phiên nữa.
3. Giữ H1 ngắn với `n_splits: 2`, `train_size: P18M`, `val_size: P6M`. **Chỉ chấp nhận nếu sweep bị cắt xuống vài chục spec.**

**Lịch sử này cho được gì và không cho được gì — phải viết vào README của case study:**

- **Cho được**: kiểm định point-in-time, quan hệ chi phí/biên độ theo phiên, một ước lượng IC với sai số HAC, và một breakeven cost.
- **Không cho được**: bằng chứng qua nhiều chế độ tiền tệ. Vàng 2022-10 -> 2026 là **một xu hướng tăng liên tục**; một sổ thiên về mua sẽ đẹp trên raw return vì lý do đó chứ không vì mô hình. Đó là lý do **mọi thống kê phải báo cáo trên active return so với sổ 1/N long hai kim loại**, đúng như user decision 3 của `exness_fx_d1` (`BOT.md:139`). Và lịch sử ngắn làm minimum track record trở nên tàn nhẫn: `exness_fx_d1` cần 6,081 phiên trong khi quan sát được 1,011 (`BOT.md:83`).

**Feature cửa sổ dài lấy từ D1/H4 (có từ 2014) trong khi quyết định và nhãn ở trên H1 — CÓ, và đây là thiết kế bắt buộc**, vì `bot-portfolio-exness.md:72-74` đã viết: *"mọi cửa sổ feature của bot theo phiên bị chặn tại biên phiên, không kéo qua đêm... Cửa sổ dài hơn một phiên dùng nến D1."*

Cách join point-in-time, viết chính xác:

> Với quyết định tại `t` (ví dụ 09:00 UTC ngày D), nến D1 mới nhất được phép dùng là nến **đã đóng tại hoặc trước `t`**, tức nến của ngày D-1 (đóng 00:00 UTC ngày D). Join là **asof backward trên thời điểm ĐÓNG của nến D1**, không bao giờ join theo ngày lịch của nến.

Tiền lệ và test có sẵn: `exness_fx_d1` làm đúng vậy cho gold factor trên lưới FX — `setup.yaml:313-318` ("XAUUSD session close on the same decision bar, last print at or before it"), lý do đầy đủ ở `BOT.md:117`, và `_features.features_as_of` (`case_studies/exness_fx_d1/_features.py:418`) là harness dựng lại panel từ đúng những nến đã đóng tại một thời điểm — test (b) của bot đó so 6 thời điểm mẫu, 57 cột, max diff 0.0 (`BOT.md:80`).

**Lợi ích lớn của thiết kế tách lưới**: warmup của feature dài đếm bằng **nến D1** (252 D1 ~ 1 năm, có từ 2014-01) chứ không bằng nến H1, nên feature 252 ngày **không** ép `history_start` lùi lại. So sánh: `exness_fx_d1` mất 1,885 hàng warmup và panel chỉ dày từ 2018-08 (`BOT.md:81`).

### 2.d Đường mô hình + backtest: chuỗi thời gian từng tài sản, không xếp hạng cross-section

**Điều gì đổi khi chỉ có 2 tài sản:**

1. **`features.ranked` phải rỗng và bỏ hẳn family `cross-sectional position`.** Với 5 mã, `exness_fx_d1` đã có 8 cột `rank_*` bị STOP vì staleness 0.68–0.98 (`BOT.md:81`, open question 12). Với 2 mã, percentile chỉ nhận hai giá trị mỗi ngày — thoái hoá hoàn toàn.
2. **`mapping`**: không dùng `long_short_rank_rebalance`. Đề xuất:
   ```yaml
   mapping:
     class: per_asset_signal_timing
     position_state_space: long_short
     entry_logic: per_symbol_threshold_on_predicted_return
     sizing: fixed_fraction
   ```
   Lưu ý cơ chế: `backtest_loaders.py:2167-2169` suy `long_short` từ token của `position_state_space`, nên giữ `long_short` trong chuỗi đó nếu muốn được phép bán khống.
3. **Phương pháp signal**: `equal_weight_top_k` vô nghĩa (top-1 trong 2 mã = "mã nào dự báo cao hơn", top-2 = mua cả hai). Dùng:
   - `fixed_threshold` (`case_studies/utils/signals.py:522`) — vào lệnh khi dự báo vượt ngưỡng tuyệt đối; đơn giản nhất, ít tham số nhất;
   - hoặc `per_symbol_rolling_percentile` (`signals.py:520-521`, `324`) — ngưỡng là phân vị lăn theo *từng mã*, đúng lý do module ghi: *"per-symbol score distributions are heterogeneous"* (`case_studies/utils/slot_strategy.py:8-10`). Đây là câu trả lời trực tiếp cho vàng vs bạc.

**Stage nào fork:**

| Stage | Fork từ | Vì sao |
|---|---|---|
| `01_feasibility_analysis`, `02_labels`, `03_financial_features`, `04_model_based_features`, `05_evaluation` | `case_studies/exness_fx_d1/` (đã được sửa cho MT5 + server UTC+0, có `_features.py` một đường code) | Tiết kiệm toàn bộ phần đã debug: chunked loader, quy tắc nến quyết định, drop ngày sự cố, `03` chỉ vẽ trên development (`exness_fx_d1/BOT.md:121`) |
| `06_linear`, `07_gbm`, `12_model_analysis` | `case_studies/exness_fx_d1/` | Menu `fx_pairs` **giữ nguyên, không cắt** (`exness_fx_d1/BOT.md:122`): menu bị cắt "theo cảm giác" là một lựa chọn thực hiện trước backtest |
| `13_backtest` | **Khung** từ `case_studies/exness_fx_d1/13_backtest.py` (catalog freeze -> `plan_backtests` -> `run_backtests` -> `require_complete`, cộng guard chi phí ở `BOT.md:130`); **cơ chế signal** từ `case_studies/nasdaq100_microstructure/14_backtest.py` + `case_studies/utils/slot_strategy.py` | Khung `fx_pairs` là thứ làm số trial chính xác; cơ chế nasdaq là thứ duy nhất trong repo diễn đạt "vào theo ngưỡng từng mã, giữ đến hết phiên, thoát theo tín hiệu / TP / SL" (`slot_strategy.py:20-38`) |
| Ch16 `04_single_asset_ml4t_backtest`, `05_stateful_strategies` | **đọc, không fork** | Đây là chương giải thích cơ chế sổ một-tài-sản có trạng thái (`roadmap.md:154-155`); code sản xuất là `slot_strategy.py` + `backtest_runner`. Vol 2 > Ch16 > `04_single_asset_ml4t_backtest`, `05_stateful_strategies` |

**Đăng ký bắt buộc (nếu thiếu, stage sẽ chết ở phase 5, không phải phase 1):** thêm `exness_gold_sess` vào `case_studies/utils/backtest_loaders.py` (`ALL_CASE_STUDIES`, `_PRICE_CONFIG`, nhánh loader -> `_features.load_session_panel`) và `case_studies/utils/analytics.py` (`CASE_STUDY_META`, `PRIMARY_LABELS`, `SHORT_NAMES`, `DATASET_META`, `CADENCE_MAP` — **cả năm**, vì `tests/test_case_study_analytics.py` đòi tập khoá giống hệt nhau); thêm `HOLDOUT_CONFORMAL_EMBARGO_STEPS` cho từng nhãn vào `case_studies/utils/conformal.py`; thêm mọi stage vào `tests/overrides.yaml` với `skip: true`. Tất cả đều là bài học đã trả giá của `exness_fx_d1` (`BOT.md:119`, `:126`, `:111`).

**Quy tắc chọn trên validation và kế toán DSR:**

- Chọn **chỉ** trên split validation, trong stage backtest, **không** theo IC và **không** đụng holdout (`.claude/skills/ml4t/SKILL.md:40-42`).
- `K` = số member của population baseline đã đăng ký, cộng dồn qua mọi thế hệ. Công thức của bot này:
  `K = |SESSION_FILTER| x tổng theo label (|prediction sets của label| x |signal specs của label|)`.
- **Ước lượng và cảnh báo**: nếu bê nguyên menu `fx_pairs` (28 linear + 15 GBM x 10 checkpoint = **178 prediction set/label**), 3 nhãn, 3 filter và chỉ 2 signal spec => `K ~ 3 x 3 x 178 x 2 = 3,204`. `exness_fx_d1` với `K = 2,136` có **0/1,068** spec vượt ngưỡng 0.95 trên active return (`BOT.md:83`). Kỳ vọng Sharpe cực đại dưới giả thuyết null tăng theo `log K` (Ch16 `12_dsr_validation`), nên **mỗi chiều sweep thêm vào làm ngưỡng cao lên cho chính mình.**
- **Khuyến nghị cắt sweep TRƯỚC khi chạy, và ghi vào `setup.yaml`**: 2 filter (`london`, `ny`) + 1 book gộp = 3; 3 nhãn; **<= 2 signal spec mỗi nhãn**; giữ nguyên menu model (menu là prior để so với `fx_pairs`). Tuyệt đối không mở lưới kiểu nasdaq (`long_q x max_slots x hold_bars x exit_signal_q` = 108 spec/prediction, `nasdaq100_microstructure/config/setup.yaml:553-559`) trên một bot 2 tài sản.
- **Thiết bị rẻ nhất và mạnh nhất của phase 1 — IC\***: tính `IC* = spread_p90 / sigma(label)` cho từng mã, từng nhãn, **trước khi fit mô hình nào**. `xau_fx_mt5` khai đúng thế: `ic_star_by_symbol: {XAUUSD: 0.008, ...}` (`case_studies/xau_fx_mt5_d1/config/setup.yaml:322-323`). Nó cho biết IC tối thiểu để trả nổi chi phí; nếu IC\* của bạc lớn hơn IC mà bất kỳ nghiên cứu nào trong repo từng đo được, câu hỏi về bạc đã có câu trả lời trước khi tốn một trial nào.

### 2.e Cổng chi phí — cần gì cho XAGUSD

**Vấn đề cơ chế phải sửa trước, nếu không phase 5 sẽ chạy trên chi phí bịa:**

`case_studies/utils/backtest_loaders.py:2297-2299` — `_cfd_pair_class(symbol)` trả `"major_pairs"` khi tên chứa `"USD"`. **XAUUSD và XAGUSD đều chứa "USD"**. Hệ quả: khai `spread_bps: {metals: [...]}` mà không khai `major_pairs` => `_normalize_cfd_costs` **ném lỗi** (`:2313-2318`, "no range for 'major_pairs'"); khai `major_pairs` bằng số của FX => vàng bị định giá theo spread FX **trong im lặng**. Và `:2322` trả `max(tops)` — **một** con số slippage cho cả sổ, không phân biệt mã.

**Việc phải làm (task riêng, có test):** mở rộng `_cfd_pair_class` để nhận lớp `metals` (XAU/XAG), giữ `_normalize_cfd_costs` trả `max` (bảo thủ: cả sổ chịu p90 của bạc), và viết unit test theo mẫu `bots/exness_fx_d1/tests/test_backtest_costs.py` (5 test, `exness_fx_d1/BOT.md:129`). Nhánh này phải được khoá trên `commission_per_lot` như nhánh cũ để **không** làm dịch chuyển `fx_pairs`.

**Viết gì vào `costs` trước khi phép đo spread về:**

```yaml
costs:
  class: material
  components: [spread, commission, swap]
  spread_bps:
    metals:      [1.16, <XAG p90 — CHƯA ĐO>]
    major_pairs: [1.16, <như trên>]   # bắt buộc tới khi _cfd_pair_class biết 'metals'
  spread_bps_provisional: true        # guard của 13_backtest từ chối chạy khi cờ này còn true
  spread_bps_by_session: { measured_on: ..., XAUUSD: {...}, XAGUSD: {...} }
  commission_per_lot: 0               # Pro (user decision, exness_gold_sess/BOT.md:15)
  swap:
    source: symbol_info
    mode: points                      # swap_mode 1
    rollover_utc: "00:00"             # nửa đêm server = 00:00 UTC
    rollover3days: wednesday          # swap_rollover3days == 3, XAUUSD.md:178, XAGUSD.md:176
    measured_on: "2026-09-05"
    points_per_lot_per_night:
      XAUUSD: {long: 0.0, short: 0.0}   # demo; đọc lại trên tài khoản thật trước 16_costs
      XAGUSD: {long: 0.0, short: 0.0}
  contract:
    contract_size: {XAUUSD: 100, XAGUSD: 5000}
    volume_min: 0.01
    volume_step: 0.01
    point: {XAUUSD: 0.001, XAGUSD: 0.001}
```

- **1.16 / 1.79 bps cho XAUUSD** là số **đã đo trên chính tài khoản này** (trường `spread` theo nến, 2017-03-06 -> 2025-08-20, 200 points): `case_studies/xau_fx_mt5/config/setup.yaml:99` và `:105`. Được dùng làm giá trị tạm với thuộc tính nguồn ghi rõ. **Cấm** dùng dải mẫu `metals: [2, 6]` của `mt5-exness-broker.md:44` như một số đo — nó là placeholder tài liệu.
- **XAGUSD: không có con số nào trong repo.** Không được bịa. Trong lúc chờ, để `spread_bps_provisional: true` và cho `13_backtest` **từ chối chạy** khi cờ còn bật — cùng loại guard mà `exness_fx_d1/13_backtest.py` đã có (từ chối khi chi phí engine khác `setup.yaml`, `BOT.md:130`).
- Kèm cảnh báo lịch sử: nến `spread == 0` tồn tại tới 2023-02-03 trên tài khoản này (1,015 nến/mã, `xau_fx_mt5/config/setup.yaml:92-95`); phải điền bằng trung vị lăn point-in-time (`experiments/xau_fx_mt5/data/costs_pit.py:fill_spread_cost`) hoặc loại, **không bao giờ định giá là miễn phí**.

**Cổng chi phí ở phase 6 — `16_costs` phải cho thấy gì:**

1. Đường `cost_grid_bps` (khai trong `backtest.sweep.cost_grid_bps`, mẫu `[0,1,2,3,5,7,10,15,20,30,50]`, `exness_fx_d1/setup.yaml:175`) -> breakeven nội suy tại Sharpe = 0 (`case_studies/utils/strategy_analysis.py:1532-1537`) và **headroom = breakeven / chi phí giả định** (`:1566-1567`).
2. **Ngưỡng phải vượt là p90, không phải trung vị**: `bot-portfolio-exness.md:121` — *"breakeven phải vượt phân vị 90 của spread phiên được chọn, không chỉ trung vị"*. Round trip = 2 lần cắt spread. Với XAGUSD: `breakeven_bps > 2 x p90_XAG(phiên giao dịch)`, và biên an toàn phải được **khai trong `setup.yaml` trước khi chạy** (`roadmap.md:193`).
3. **Bằng chứng phải tách theo mã.** Đường chi phí tính trên sổ; với sổ 2 mã, builder phải chạy thêm bản per-symbol (mẫu: so sánh full-vs-screened universe của nasdaq/sp500_options, `nasdaq100_microstructure/config/setup.yaml:476-479`) cộng bảng chẩn đoán mức 3 (`bot-portfolio-exness.md:91-94`: nhóm trade log theo phiên **và** theo mã, báo số lệnh, hit rate, gộp/ròng, chi phí trung bình, MAE/MFE Ch07 `04`, thời gian giữ — **chỉ để đọc, không để chọn**).
4. **Nếu chân bạc không tự trả nổi p90 của chính nó => bỏ XAGUSD khỏi `universe`.** `bots/assets/XAGUSD.md:144` đã chấp nhận trước kết cục này. Lưu ý hệ quả kỹ thuật: đổi universe => nhãn mới, hash mới, **experiment/generation mới**, và các trial cũ vẫn được đếm vào `K` (mẫu supersede: `exness_fx_d1/BOT.md:137`).
5. **Kiểm tra hạt lot của bạc (dễ bỏ sót):** `execution.share_type: integer` nghĩa là đơn vị của engine = `lots x trade_contract_size` = **ounce**. Với XAUUSD, `volume_min 0.01 x 100 = 1 oz` => mọi số nguyên đều hợp lệ. Với XAGUSD, `0.01 x 5000 = 50 oz` => **backtest có thể khớp 73 oz bạc, một khối lượng không đặt được**; adapter làm tròn xuống (`normalize_lot`, `exness_fx_d1/BOT.md:65`) => phân kỳ backtest/live, đúng loại lỗi Ch25 §25.1. Phải khai ràng buộc này và test nó trong `tests/test_parity.py`.

### 2.f Register feature (`features`) — khai **trước** khi đọc IC

Nguyên tắc: `roadmap.md:118-120` (Ch20 — feature sống sót là *bộ lọc*, không phải bằng chứng chiến lược) và tiền lệ `exness_fx_d1/BOT.md:151` ("one feature register, declared before any IC was read"). Mọi family cần `pattern`, `role`, `hypothesis`, `inputs`, `lookback`, `lag`, `frame`, `representation`, `failure_mode` (mẫu đầy đủ: `exness_fx_d1/setup.yaml:238-328`).

| Family | Lưới | Nguồn / lý do |
|---|---|---|
| `session momentum` (từ mở phiên đến mốc quyết định; 1–4 nến H1) | H1, **chặn tại biên phiên** | `XAUUSD.md:130`; `bot-portfolio-exness.md:72` |
| `session range / position in range` | H1, chặn biên phiên | `XAUUSD.md:130` |
| `overnight vs intraday decomposition` (gap từ đóng phiên trước sang mở phiên này) | H1 + D1 | `XAUUSD.md:131` (Ch08 `01_price_volume_features`) |
| `gold-silver ratio` + z-score, beta chéo | **D1** (cửa sổ > 1 phiên) | `XAGUSD.md:128`, `XAUUSD.md:129` (Ch08 `03_structural_cross_instrument_features`) |
| `long-window state` (momentum 21/63/252, vol GK, drawdown) | **D1**, asof-join theo giờ đóng | `bot-portfolio-exness.md:74`; join theo `exness_fx_d1/setup.yaml:313-318` |
| `session flags` (`london`/`ny`, `edge_open`, `edge_close`, `rollover`, cờ cửa sổ tin NFP/CPI/FOMC +-30 phút) | H1 | `bots/_shared/sessions.py:326-342`; lịch tin ở `XAUUSD.md:87-97`. **role: state/conditioner**, không phải signal |
| `real yields / USD` | **PLANNED — chưa có dữ liệu** | `XAUUSD.md:128` gọi lợi suất thực là động lực chính, nhưng `data/macro/config.yaml` chưa có, `FRED_API_KEY` vắng trong WSL `.env` (`exness_fx_d1/BOT.md:167`). **Khai là planned, đếm +1 trial khi thử** — đúng cách `exness_fx_d1` xử lý carry (`BOT.md:116`) |
| ~~`cross-sectional position` (`rank_*`)~~ | **BỎ** | 2 tài sản (§2.d) |

`ranked: []`, `null_policy_carrier` = cột có chuỗi D1 dài nhất (ví dụ `zscore_252d`), `redundancy_cut: 0.7`.

### 2.g `modeling`

Copy nguyên menu `fx_pairs` (28 linear + 15 GBM mỗi nhãn), **không cắt** (`exness_fx_d1/BOT.md:122`). Thêm trong **experiment** (không phải bản repo): `modeling.gbm.num_threads: 2` nếu chạy trong WSL2 — đo được 78 s ở 8 luồng vs 1.2 s ở 2 luồng do tranh chấp OpenMP (`exness_fx_d1/BOT.md:125`); và `LD_LIBRARY_PATH` trỏ `libgomp.so.1` (`BOT.md:124`). Giữ `max_bin: 255` (`exness_fx_d1/setup.yaml:378`).

---

## 3. Phase 8 và 9

### 3.1 Danh sách file

```
bots/exness_gold_sess/
  BOT.md                     (đã có; builder cập nhật cuối mỗi phase)
  README.md                  sổ tay vận hành: lịch chạy, credential, thủ tục arming, kill switch
  deploy/
    deployment_loop.py       7 bước theo 25_live_trading/02_etfs_deployment_loop.py:
                             1 Refresh Data (:125) -> 2 Load Inputs & Compute Features (:158)
                             -> 3 Build Training Matrix & Refit (:189, kèm Lookahead Guard :213)
                             -> 4 Persist the Deployment Artefact (:253) -> 5 Predict the Live Window (:303)
                             -> 6 Build the Strategy (:328) -> 7 Offline Reference Tape qua
                             ml4t.backtest.Engine (:422) -> staging lệnh qua SafeBroker (:462)
                             -> 9 Reconcile (:616) -> 10 Persist Run Metadata (:676).
                             Mẫu gần nhất cho rổ FX/CFD: 25_live_trading/11_fx_deployment_loop.py
                             (mt5-exness-broker.md:85-87). KHÔNG có bản sao order_send ở đây:
                             import bots/_shared/mt5_broker.py (bot-template.md:60-70)
    risk_config.yaml         LiveRiskConfig + magic + shadow_mode: true (3.3)
    state/                   RiskState đã persist + run record JSON mỗi chu kỳ
  monitor/
    drift.py                 26_mlops_governance/01_drift_monitoring.py: PSI, K-S, rolling IC, hit rate
    online_detectors.py      26/02_online_drift_detection.py: ADWIN, DDM trên luồng lỗi
    circuit_breakers.py      26/04_circuit_breakers.py: CLOSED/OPEN/HALF_OPEN; drawdown, lỗ ngày,
                             chuỗi thua, latency
    rollout.py               26/03_safe_model_rollout.py: shadow -> A/B giới hạn vốn -> staged
  tests/
    test_parity.py           25/08_pipeline_verification.py: data -> features -> predictions -> sizing -> orders
                             + kiểm tra hạt lot XAG 50 oz (2.e mục 5)
    test_lookahead.py        dựng lại feature tại t chỉ với dữ liệu <= t; so với panel batch
    test_sessions.py         mốc quyết định = open+1h ở cả hai mùa; điểm cuối nhãn <= đóng phiên
    test_data_quality.py     mẫu bots/exness_fx_d1/tests/test_data_quality.py
    test_backtest_costs.py   nhánh CFD 'metals' (2.e)

bots/_shared/monitor/        CHƯA TỒN TẠI — phải tạo (bot-portfolio-exness.md:40, :123-128):
    account_breakers.py      breaker cấp TÀI KHOẢN: equity, margin level, tổng lệnh mở,
                             cắt MỌI bot cùng lúc, và phải kích hoạt TRƯỚC stop-out của Exness
                             (mt5-exness-broker.md:92-93)
```

Cấu trúc này là `bot-template.md:76-92` cộng `bot-portfolio-exness.md:42-49`.

### 3.2 Quy tắc magic number

1. **Một magic cho một bot**, truyền vào `MT5Broker(magic=...)`; adapter lọc `positions_get`/`orders_get` theo magic (`bots/_shared/mt5_broker.py:536`, `:630`) và đóng lệnh theo ticket mang magic đó (`:877`). Magic trùng nhau => hai bot reconcile vị thế của nhau => bot này đóng lệnh của bot kia.
2. **Giá trị cho bot này: `260903`** — giữ nguyên đề xuất đã ghi ở `bots/exness_gold_sess/BOT.md:65`. Khác `260901` của `exness_fx_d1` (`BOT.md:106`) và khác `202500` của bot legacy (`exness_gold_sess/BOT.md:74`).
3. **Phải bổ sung**: một bảng magic duy nhất trong `bots/README.md` (bot_id -> magic -> trạng thái) + một test khẳng định tính duy nhất trên mọi `deploy/risk_config.yaml`. Hiện `bots/xau_fx_mt5/BOT.md` **không ghi magic nào** — nếu bot đó lên demo cùng lúc, va chạm là không thể phát hiện được từ code.
4. Magic **không bao giờ** trùng magic của bot legacy, và `.env` của legacy không bao giờ được đọc hay copy vào repo (`BOT.md:74`).

### 3.3 `LiveRiskConfig` — các trường phải khai

Từ cách `25_live_trading/10_safety_risk_demo.py` khởi tạo (`:345-350`, `:404-412`, `:431-439`, `:459-465`) và `mt5-exness-broker.md:79-81`:

```yaml
# bots/exness_gold_sess/deploy/risk_config.yaml
magic: 260903
shadow_mode: true                 # bot-template.md:82; mặc định là dry run
live_risk_config:
  execution_mode: paper           # 'paper' bắt assert_paper_trading; 'live' cần armed_live mỗi phiên
  max_order_shares: <oz>          # ĐƠN VỊ LÀ OUNCE (lots x contract_size). XAU: bội của 1; XAG: bội của 50
  max_order_value: <USD>
  max_position_shares: <oz>       # per symbol
  max_position_value: <USD>       # per symbol
  max_total_exposure: <USD>       # cả bot; phải nhỏ hơn hạn mức cấp tài khoản
  state_file: bots/exness_gold_sess/deploy/state/risk_state.json
```

Thêm hai điều bắt buộc khi vận hành, không nằm trong dataclass: (a) `safe_broker.record_market_snapshot(symbol, price)` từ `symbol_info_tick` trước mỗi lần đặt lệnh, nếu không guard staleness sẽ chặn (`10_safety_risk_demo.py:352-354`, `mt5-exness-broker.md:80-81`); (b) guard margin của adapter ở 20 % free margin, fail-closed (`exness_fx_d1/BOT.md:65`).

### 3.4 Hypothesis — bản nháp để user phê duyệt (theo văn phong `exness_fx_d1`)

> *Vàng và bạc trên tài khoản CFD này được định giá bởi cùng một động lực (lợi suất thực Mỹ và USD; `bots/assets/XAUUSD.md:66`, Ch08 `04_fundamentals_macro_calendar`), tương quan khoảng 0.8 (`XAGUSD.md:140`), nên bot không đi tìm lợi thế xếp hạng cross-section: hai tên không tạo ra một cross-section. Lợi thế của nó, nếu có, là (i) **cấu trúc trong ngày**: dòng tiền tập trung vào giờ mở London và giờ mở New York, và biên độ cùng spread của hai phiên đó khác hẳn phiên Á (`XAUUSD.md:146`), nên một dự báo về lợi nhuận từ mốc mở phiên đến hết phiên có thể mang thông tin mà một dự báo daily làm nhoè; và (ii) **hồi quy của tỷ số vàng/bạc** khi một trong hai chân đi trước (`XAGUSD.md:128`, Ch08 `03`). Người đứng phía bên kia là dòng lệnh phải giao dịch bất kể giá — phòng hộ của thợ mỏ và người dùng công nghiệp, cân bằng lại quỹ ETF, và định giá LBMA hai lần một ngày (`XAUUSD.md:89`). Giả thuyết bị bác bỏ khi, tại số trial cộng dồn ghi trong bảng Trials, không spec nào có DSR trên **active return so với sổ 1/N long hai kim loại** đạt mức đã khai, hoặc khi breakeven cost từ `16_costs` thấp hơn p90 spread đo trên tài khoản Pro thật của phiên được chọn. Bối cảnh phải vượt qua: `exness_fx_d1` chạy 2,136 trial trên 5 cặp FX cùng broker và không có spec nào vượt ngưỡng (`bots/exness_fx_d1/BOT.md:144`); `xau_fx_mt5` chạy 3 trial trên XAUUSD cùng tài khoản và đóng cả hai đường H1 và D1 (`bots/xau_fx_mt5/BOT.md:9-17`). Prior của repo là "chưa có lợi thế nào được phân giải"; nó đứng vững cho tới khi có bằng chứng ngược lại.*

**Mức phải khai kèm (user quyết định):** quy ước notebook của `16_strategy_simulation/12_dsr_validation.py`, `confidence_level = 0.95`, thống kê báo cáo trên **active return** (chiến lược trừ sổ 1/N long hai kim loại) song song với so với SR = 0 — đúng user decision 3 của `exness_fx_d1` (`BOT.md:139`). Với vàng, active return là điều **bắt buộc chứ không phải tuỳ chọn**: mẫu 2022–2026 là một xu hướng tăng, sổ nào thiên về mua cũng đẹp trên raw return.

### 3.5 Kill criteria — bản nháp để user phê duyệt (viết trước lệnh live đầu tiên)

> *Tạm dừng khi (a) drawdown từ đỉnh > 8 % vốn phân bổ cho bot; hoặc (b) hit rate lăn 63 quyết định của sổ đang chạy dưới 0.5 trong 21 phiên liên tiếp (IC trên hai tên không phải một thống kê đọc được); hoặc (c) chi phí round trip thực hiện trên tape demo/live vượt 1.5x chi phí giả định trong `setup.yaml::costs` suốt một tháng dương lịch; hoặc (d) reconciliation của `SafeBroker` tìm thấy một vị thế không giải thích được mang magic 260903; hoặc (e) nến quyết định thiếu trong dung sai — không đặt lệnh phiên đó (quy tắc sự cố; `exness_fx_d1/BOT.md:166`); hoặc (f) swap thực tế mỗi đêm vượt giá trị khai trong `setup.yaml::costs.swap` suốt một tuần; hoặc (g) một mã bị halt/gap qua cửa sổ nghỉ hằng ngày 21:00–22:00 UTC trong khi bot còn giữ lệnh (rủi ro riêng của kim loại, `XAUUSD.md:110`).*
>
> *Nghỉ hưu khi ước lượng breakeven cost mới nhất (`16_costs`) thấp hơn p90 spread đo được của phiên đang giao dịch, hoặc khi PSR trên holdout so với sổ 1/N long hai kim loại dưới 0.5.*
>
> *Breaker cấp tài khoản (equity, margin level, tổng lệnh mở) nằm ở `bots/_shared/monitor/` và cắt mọi bot cùng lúc, trước stop-out của Exness. Ràng buộc riêng của tài khoản này: `xau_fx_mt5` cũng giao dịch XAUUSD trên cùng login, nên hạn mức exposure của bot này phải được đặt sao cho tổng của hai bot vẫn dưới breaker cấp tài khoản.*

Không có con số lợi nhuận kỳ vọng nào trong hai bản nháp trên, và không được thêm vào.

---

## 4. Checklist cổng theo từng phase

| Phase | Cổng phải xanh trước khi sang phase sau |
|---|---|
| **0** | `uv run python scripts/verify_installation.py` PASS trong WSL2; `import ml4t.data, ml4t.diagnostic, ml4t.engineer, ml4t.backtest, ml4t.live` OK; `pytest bots` xanh. (`roadmap.md:41-42`) |
| **B0 (tiền phase 1)** | H1 của **cả** XAUUSD và XAGUSD tải bằng đường count-based; độ sâu thực ghi vào `bots/assets/*.md` §12 và vào Decisions log của bot; mâu thuẫn X1 được ghi ở cả hai BOT.md |
| **1** | `setup.yaml` tồn tại với: `universe.history_start` đo được; **hai** snapshot và quy tắc open+1h; `evaluation.n_splits/train_size/val_size/holdout_start/holdout_end` khai **trước** lần train đầu tiên; `labels.rebalance_step` và `labels.classification_eval_label` (2 khoá pinned) tác giả một lần trên `run_log/` rỗng; `costs` có cờ `provisional` cho tới khi đo xong. `01_feasibility_analysis` exit 0 và khẳng định: mỗi hàng quyết định có nến, không nến nào đóng sau giờ đóng phiên, số fold sinh đúng, **holdout untouched**. `02_labels` exit 0 với digest sidecar và bảng `N_eff`. Hypothesis + kill criteria **user đã duyệt**. IC\* mỗi mã mỗi nhãn đã tính. (`roadmap.md:72-73`) |
| **2** | Test lookahead tự động (dựng lại feature tại >= 5 thời điểm mẫu, max diff 0.0) và test data-quality chạy được lặp lại, **chỉ trên cửa sổ development**; báo cáo nến thiếu / nến spread 0 / outlier spread; loader đã đăng ký trong `backtest_loaders.py` và `analytics.py`. (`roadmap.md:93-94`) |
| **3** | Register khai trước khi đọc IC; `03`/`04`/`05` exit 0; `triage_ledger.parquet` có ICIR và tỷ lệ fold dương; không cột nào tính từ dữ liệu tương lai; mọi hình trong `03` vẽ trên development. Nhắc lại Ch20: sống sót triage là bộ lọc, không phải bằng chứng. (`roadmap.md:118-120`) |
| **4** | Mọi lựa chọn dựa trên validation; số training run và prediction set ghi trong registry; baseline tuyến tính có mặt cạnh GBM; **không** cắt menu theo cảm giác. (`roadmap.md:146-147`) |
| **5** | Mỗi `SESSION_FILTER` là một hash riêng **và** cho ra `weights.parquet` khác nhau (guard §2.a); `resolve_rebalance_timestamps` đã được in ra và kiểm trên lưới đã lọc; Sharpe validation đi kèm **DSR tại K cộng dồn thực**, trên raw **và** active return; không con số nào từ holdout; `avg_turnover` không được dùng nếu chưa recompute — đọc từ ledger (`exness_fx_d1/BOT.md:173`). (`roadmap.md:167-168`) |
| **6** | breakeven > 2 x p90 spread của phiên giao dịch, với biên an toàn đã khai trước; bảng chẩn đoán mức 3 theo phiên **và** theo mã; quyết định giữ/bỏ XAGUSD; swap đọc lại trên tài khoản Pro thật trước khi `16_costs` chạy; mọi overlay rủi ro gắn với một hành động định trước. (`roadmap.md:193-194`, `bot-portfolio-exness.md:121`) |
| **7** | Holdout chấm **một lần** bằng `17_holdout_predictions` / `18_holdout_backtest`; `19_strategy_analysis` sinh `strategy_assessment.json`; nếu quay lại sửa thiết kế thì holdout coi như đã "đốt" và phải ghi. (`roadmap.md:215`) |
| **8** | Parity tape (replay offline) khớp lệnh paper; kill switch thử được; reconciliation chạy lúc khởi động; magic duy nhất đã test; `assert_paper_trading` từ chối tài khoản không phải demo; checklist vận hành Ch25 §25.7 đã ký. (`roadmap.md:237-238`) |
| **9** | Bot tự dừng khi vi phạm ngưỡng mà không cần người; breaker hai tầng (bot + tài khoản) thử được; mọi mô hình mới đi qua shadow trước khi nhận vốn; run record lưu mỗi chu kỳ. (`roadmap.md:255-257`) |

**Rủi ro lớn nhất của kế hoạch này**, và nó là rủi ro về *chứng cứ*, không phải về code: nếu builder chốt `universe.history_start: 2022-10-25` từ con số của loader dùng chung, toàn bộ bot — nhãn, fold, holdout, DSR, mọi hash trong registry — sẽ được dựng trên **2.85 năm của đúng một chế độ thị trường vàng tăng giá**, trong khi một bot anh em đã chứng minh 9.5 năm cùng mã lấy được từ cùng terminal. Sai lầm đó không lộ ra ở bất kỳ cổng nào cho tới phase 5, và sửa nó thì phải làm lại từ `02_labels`. Một lệnh tải count-based ở B0 đóng rủi ro này. Rủi ro lớn thứ hai, cùng loại: `_cfd_pair_class` trả `major_pairs` cho XAUUSD/XAGUSD (`backtest_loaders.py:2297-2299`) — hoặc stage chết, hoặc kim loại bị định giá bằng spread FX trong im lặng.

---

**Next step**: chạy task B0 — tải lại H1 cho `XAUUSD` và `XAGUSD` bằng đường count-based (`copy_rates_from` + epoch seconds, mẫu `experiments/xau_fx_mt5/data/build_panel.py:333-352`), ghi độ sâu thực đo được của **cả hai** mã vào `bots/assets/XAUUSD.md` §12 và `bots/assets/XAGUSD.md` §12, rồi mới viết `case_studies/exness_gold_sess/config/setup.yaml` theo §2.

**Read**:
1. `.claude/skills/ml4t-quant-bot-mentor/references/bot-portfolio-exness.md` §4 "Backtest theo phiên: ba mức" (dòng 75–98) và §5 "Quy tắc giữ lệnh và thoát theo phiên" (dòng 100–109) — cùng `bots/xau_fx_mt5/BOT.md:672` cho lý do H1 count-based.
2. PDF Tập 2 > Ch07 > `03_label_methods` (triple-barrier và `LabelingConfig.triple_barrier`, `atr_triple_barrier_labels`, `measure_n_eff`) và Tập 2 > Ch16 > `12_dsr_validation` (Deflated Sharpe với số trial thực).
