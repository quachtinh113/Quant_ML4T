# Hướng Dẫn Vận Hành Hệ Thống 4 Bot Exness (Windows Task Scheduler)

Không gian làm việc chuyên dụng để vận hành tự động 4 bot định lượng trên cùng một máy tính:
- **`exness_btc_8h`** (BTCUSD - Magic: `260904`)
- **`exness_fx_d1`** (5 cặp tiền FX - Magic: `260901`)
- **`exness_usidx_sess`** (US500 & USTEC - Magic: `260902`)
- **`exness_gold_sess`** (XAUUSD & XAGUSD - Magic: `260903`)

---

## 1. Cấu Trúc Thư Mục

```text
D:/05_Quant/ML_4_Bot/
├── runners/                       # Các script kích hoạt chu kỳ từng bot
│   ├── run_btc_8h.bat             # Chạy chu kỳ BTC 8h
│   ├── run_fx_d1.bat              # Chạy chu kỳ FX Daily
│   ├── run_usidx_sess.bat         # Chạy chu kỳ Chỉ số Mỹ
│   ├── run_gold_sess.bat          # Chạy chu kỳ Vàng & Bạc
│   ├── session_gate.py            # Cổng giờ phiên (DST): quyết định chân nào đến giờ, hoặc bỏ qua
│   └── check_fleet_status.bat     # Xem nhanh trạng thái & log 4 bot
├── logs/                          # Nhật ký chạy từng bot (tự động cập nhật)
├── setup_tasks.ps1                # Tự động đăng ký 4 tác vụ vào Task Scheduler
├── remove_tasks.ps1               # Gỡ bỏ 4 tác vụ khỏi Task Scheduler
└── .env                           # Cấu hình tài khoản & đường dẫn 4 bot
```

---

## 1.1 Cấu Hình 4 Tài Khoản MT5 Độc Lập

Hệ thống sử dụng **4 thư mục MetaTrader 5 riêng biệt** để 4 bot chạy độc lập, không xung đột tài khoản:

| Bot | Thư Mục Terminal | Tài Khoản Login | Server | File Mở Nhanh |
|---|---|:---:|:---:|---|
| **BTC 8h** | `D:\05_Quant\ML_4_Bot\terminals\MT5_BTC\terminal64.exe` | **`416351011`** | `Exness-MT5Trial14` | `runners\open_terminal_btc.bat` |
| **FX D1** | `D:\05_Quant\ML_4_Bot\terminals\MT5_FX\terminal64.exe` | **`463960816`** | `Exness-MT5Trial17` | `runners\open_terminal_fx.bat` |
| **USIDX Sess** | `D:\05_Quant\ML_4_Bot\terminals\MT5_USIDX\terminal64.exe` | **`463960820`** | `Exness-MT5Trial17` | `runners\open_terminal_usidx.bat` |
| **Gold Sess** | `D:\05_Quant\ML_4_Bot\terminals\MT5_GOLD\terminal64.exe` | **`463960823`** | `Exness-MT5Trial17` | `runners\open_terminal_gold.bat` |

> [!TIP]
> Nhấp đúp vào `runners\open_all_terminals.bat` để mở cả 4 terminal cùng một lúc!
> Mở từng terminal lên, chọn **File ➔ Login to Trade Account**, nhập Password `87u3D1$6` và tích chọn **Save password** một lần duy nhất.

---

## 2. Kích Hoạt Tự Động Hóa Bằng Windows Task Scheduler

Mở **PowerShell** (Run as Administrator) và chạy lệnh:

```powershell
cd D:\05_Quant\ML_4_Bot
.\setup_tasks.ps1
```

Script sẽ tự động đăng ký 4 tác vụ vào Windows Task Scheduler (chạy trễ 1 phút sau khi nến đóng để đảm bảo server MT5 đã hoàn tất nến):

| Task Name | Bot | Giờ Chạy (Giờ VN - UTC+7) | Giờ Quyết Định (UTC) | Ngày (giờ VN) |
|---|---|---|---|---|
| **`Exness_Bot_BTC_8h`** | `exness_btc_8h` | `07:01`, `15:01`, `23:01` | `00:00`, `08:00`, `16:00` cố định | Hằng ngày (7 ngày/tuần) |
| **`Exness_Bot_FX_D1`** | `exness_fx_d1` | `03:01` (sáng hôm sau) | `20:00` cố định | Thứ 3 ➔ Thứ 7 |
| **`Exness_Bot_USIDX_Sess`** | `exness_usidx_sess` | `21:01`, `22:01` (mở phiên) và `03:01`, `04:01` (đóng phiên) | Mở NYSE +30' làm tròn giờ: `14:00` hè / `15:00` đông. Đóng NYSE: `20:00` hè / `21:00` đông | Mở: Thứ 2 ➔ Thứ 6. Đóng: Thứ 3 ➔ Thứ 7 |
| **`Exness_Bot_Gold_Sess`** | `exness_gold_sess` | `15:01`, `16:01` (London) và `20:01`, `21:01` (New York) | London mở +1h: `08:00` hè / `09:00` đông. NY mở +1h: `13:00` hè / `14:00` đông | Thứ 2 ➔ Thứ 6 |

### Vì sao Gold và USIDX có hai giờ cho mỗi chân

Task Scheduler chạy theo giờ máy (giờ VN, không đổi quanh năm), còn hai bot này quyết định theo giờ của sàn London và New York, vốn đổi giờ hai lần mỗi năm (Anh: cuối tháng 3 và cuối tháng 10; Mỹ: giữa tháng 3 và đầu tháng 11). Vì vậy mỗi chân được đăng ký ở cả giờ mùa hè lẫn giờ mùa đông. Khi task chạy, file `.bat` gọi `runners\session_gate.py` trước:

- Nếu giờ hiện tại đúng là giờ quyết định của một chân, cổng in tên chân (`london`/`ny` hoặc `intraday`/`overnight`) và bot chạy chu kỳ.
- Nếu không, cổng trả mã 10, `.bat` ghi một dòng `Skipped` vào log và kết thúc. Deployment loop không được gọi, không có run record nào được tạo.

Lần chạy lệch giờ vì vậy chỉ tốn một dòng log. Để xem giờ quyết định thật của tuần này:

```powershell
py -3.12 runners\session_gate.py gold --table
py -3.12 runners\session_gate.py usidx --table
```

Cổng chỉ biết múi giờ và ngày trong tuần. Ngày nghỉ lễ của sàn và phiên rút ngắn do deployment loop tự xử lý (nó sở hữu lịch giao dịch) và sẽ báo `NotReady` trong log như trước.

---

## 3. Kiểm Tra & Giám Sát Hoạt Động

### Xem Trạng Thái Nhanh:
Bấm đúp chuột vào file:
```cmd
D:\05_Quant\ML_4_Bot\runners\check_fleet_status.bat
```
Hoặc xem trực tiếp các file log trong thư mục `D:\05_Quant\ML_4_Bot\logs\`:
- `btc_8h.log`
- `fx_d1.log`
- `usidx_sess.log`
- `gold_sess.log`

Để xem cả 4 bot cùng lúc (evidence, execution mode, run history, Task Scheduler, đèn tín hiệu) trên một trang HTML đọc-only, chạy `runners\run_dashboard.bat` hoặc xem `dashboard\README.md`.

---

## 4. Kiểm Soát Rủi Ro & Dừng Khẩn Cấp (Circuit Breakers)

### Dừng Khẩn Cấp Cả 4 Bot Lập Tức:
Tạo một file rỗng có tên:
```text
D:\05_Quant\machine-learning-for-trading\data\mt5\monitor\account_halt.json
```
Khi file này tồn tại, cả 4 bot ở chu kỳ kế tiếp sẽ **lập tức từ chối đặt lệnh mới** và ghi log cảnh báo `ACCOUNT HALTED`.

### Xóa Dừng Khẩn Cấp:
Chỉ cần xóa file `account_halt.json` sau khi đã kiểm tra an toàn tài khoản.

---

## 5. Chuyển Sang Chế Độ Đặt Lệnh Thật (Arming)

Mặc định, cả 4 bot chạy ở chế độ **shadow** (đọc nến thật, tính tín hiệu, giả lập vị thế, không gửi lệnh ra MT5). Các file `.bat` không có cờ `--dry-run` để thay; chúng gọi loop với `--cycle` hoặc `--spec`, và shadow là trạng thái mặc định của từng loop.

**Trạng thái hiện tại (2026-09-12): không bot nào được phép arm.** Cả 4 bot đều kết thúc Phase 5 với 0 survivor, nên không có model deployable. Việc thêm `--arm` vào lúc này không mở được lệnh, và cũng không được phép theo BOT.md của từng bot.

Khi một bot có survivor được mentor xác nhận và Phase 6/7 mở, arming cần **đủ các điều kiện sau**, không chỉ một cờ:

| Bot | Điều kiện phải thỏa trước khi `--arm` có tác dụng |
|---|---|
| `exness_btc_8h` | `bots/exness_btc_8h/deploy/risk_config.yaml` phải đặt `shadow_mode: false`. Cờ `--arm` không ghi đè giá trị này. |
| `exness_fx_d1` | Phải có model đăng ký và truyền `--model <training_hash>`; loop dừng ở bước parity khi không có model deployable. Thêm `--arm` cho phiên chạy đó. |
| `exness_usidx_sess` | `risk_config.yaml::breakers.pending_user_approval` phải là `false` (trader đã duyệt kill criteria, ghi ngày duyệt vào BOT.md). Loop từ chối `--arm` khi còn `true`. |
| `exness_gold_sess` | Như usidx: `breakers.pending_user_approval: false`, và `shadow_mode: false` trong risk_config. |

Sau khi các điều kiện trên đã thỏa:
1. Mở file `.bat` tương ứng trong `runners/` và thêm `--arm` vào câu lệnh Python, ví dụ:
   ```cmd
   py -3.12 -m bots.exness_btc_8h.deploy.deployment_loop --cycle --arm >> "%LOG_FILE%" 2>&1
   ```
2. Terminal MT5 phải đang mở, đã đăng nhập đúng tài khoản, và bật `Tools -> Options -> Expert Advisors -> Allow Algo Trading`. Loop tự đối chiếu login, server và trade_mode với cấu hình; sai một trong ba là dừng.
3. Chế độ `paper` chỉ chấp nhận tài khoản demo. Chuyển sang tài khoản thật là một quyết định riêng, phải đổi `execution_mode` trong risk_config và một người truyền `armed_live` cho phiên đó.
4. Chạy thử một chu kỳ bằng tay (nhấp đúp `.bat`) và đọc log trước khi để Task Scheduler chạy tự động.

Để quay lại shadow: bỏ `--arm` khỏi `.bat`. Để dừng khẩn cấp cả fleet: tạo file `account_halt.json` như mục 4.
