// ESP32-S3
// motor dataset collection rig
//
// IDE settings:
//   Tools -> "USB CDC On Boot: DISABLED"  (Serial = UART0 -> CH343 bridge)
//
// note!!
// the BTS7960s on the IBT-2 have internal pull-downs on IN and INH, 
// so ESP32 pins floating during boot or flashing read LOW at the driver and the bridge stays disabled
//
//   POWER SEQUENCING RULE: plug in the USB first, then turn on the power supply. never flash with 24V present

#include <SPI.h>
#include <Wire.h>
#include <Adafruit_MLX90614.h>
#include <driver/i2s.h>
#include <esp_task_wdt.h>
#include <esp_system.h>
#include <driver/gpio.h>

// GY-906 (MLX90614) - temperature (I2C)
#define MLX90614_SCL 4
#define MLX90614_SDA 5

// vibration (SPI). Two ST boards, both in the path:
//   STEVAL-MKI208V1K  carries the IIS3DWB itself, sensor soldered centre-board,
//                     adhesive-mounted on the motor frame. This is the part the
//                     datasheet numbers refer to.
//   STEVAL-MKIGIBV2   DIL24 adapter it cables into; what the PCB actually
//                     footprints (see edge/hardware/*/sensors.kicad_sch).
#define IIS3DWB_CS  13
#define IIS3DWB_SCL 12  // SCK
#define IIS3DWB_SDA 11  // MOSI
#define IIS3DWB_SDO 10  // MISO

// INMP441 - acoustic (I2S)
#define I2S_SD      18
#define I2S_SCK     17
#define I2S_WS      16
#define I2S_PORT    I2S_NUM_0

// IBT_2 - motor driver
#define MOTOR_RPWM 41
#define MOTOR_LPWM 40
#define MOTOR_R_EN 38
#define MOTOR_L_EN 39

// run config
static const float    TRIP_TEMP_C     = 70.0f;                      // kills system at 70 deg (based on the ISO-13732-1 burn threshold)
static const uint32_t TEMP_STALE_MS   = 5000;                       // sensor-lost timeout
static const uint32_t MAX_RUN_MS      = 6UL * 60UL * 60UL * 1000UL; // 6 h cap
static const int      MOTOR_SPEED     = 255;                        // -255 to 255
static const uint32_t WDT_TIMEOUT_MS  = 5000;                       // watchdog timeout

// Accelerometer full scale. CHANGED 2026-07-30 from +/-16 g to +/-4 g.
//
// Measured on the three 6 h pilot sessions recorded at +/-16 g: running RMS is
// 0.043 g (15 V) to 0.056 g (24 V) with a worst-case peak of 1.16 g. At
// +/-16 g (0.488 mg/LSB) an RMS of 0.056 g is only ~115 LSB, i.e. about 7 of
// the 16 available bits were carrying signal.
//
// +/-4 g gives 0.122 mg/LSB -- 4x finer resolution, ~9 bits at the same RMS --
// while still leaving 3.4x headroom over the observed 1.16 g peak. +/-2 g was
// rejected as too tight for the impulsive transients that seeded faults are
// expected to produce.
//
// IF THIS IS CHANGED AGAIN: data recorded either side of the change is not
// resolution-comparable. data_catcher.py writes ACCEL_FS_G into every
// session.json so analysis code scales each session correctly; the two values
// must be kept in step (ml/config.py documents both).
//
// IIS3DWB CTRL1_XL (10h). Verified against DS12569 Rev 8, Table 28 (register
// layout) and Table 30 (full-scale selection), p.32:
//   bit 7..5  XL_EN[2:0]   101 = accelerometer enabled, 000 = power-down
//   bit 4     must be 0
//   bit 3..2  FS[1:0]_XL   00 = +/-2 g (default), 01 = +/-16 g,
//                          10 = +/-4 g,           11 = +/-8 g
//   bit 1     LPF2_XL_EN   0 = first filtering stage output
//   bit 0     must be 0
//   0xA4 = 1010_0100 -> FS_XL=01 -> +/-16 g   (pilot sessions 2026-07-16..20)
//   0xA8 = 1010_1000 -> FS_XL=10 -> +/-4 g    (current)
static const uint8_t  IIS3DWB_CTRL1_XL_VAL = 0xA8;                  // XL on @ 26.7 kHz, FS +/-4 g
static const float    ACCEL_FS_G           = 4.0f;                  // must match the register above

// CTRL1_XL as read back from the device after configuration; 0xFF until
// setupIIS3DWB() has run. Shipped in every status packet so the host records
// the full scale the hardware reports rather than the one it was told to
// expect. Four sessions were recorded at +/-16 g while every session.json
// claimed +/-4 g, because the constants above were edited but the build was
// never flashed and nothing compared intent against hardware.
static volatile uint8_t iisCtrl1XL = 0xFF;
#define REQUIRE_HOST_ARM 1       

// fault handler
enum FaultCode : uint8_t {
  FAULT_NONE          = 0,
  FAULT_OVERTEMP      = 1,
  FAULT_TEMP_SENSOR   = 2,   // GY-906 stopped answering
  FAULT_RUNTIME_DONE  = 3,   // planned end of run
  FAULT_BAD_RESET     = 4,   // rebooted via WDT/panic/brownout (stay off)
  FAULT_VIB_INIT      = 5,   // IIS3DWB WHO_AM_I or CTRL1_XL readback failed
  FAULT_HOST_STOP     = 6,   // logger sent 'X' or disconnected intentionally
};

static volatile uint8_t faultCode = FAULT_NONE;

// binary packet protocol
enum PacketType : uint8_t { PKT_VIB = 0, PKT_AUDIO = 1, PKT_STATUS = 2 };

// flags bits
#define FLAG_FIFO_OVERRUN 0x01
#define FLAG_FAULTED      0x02

struct __attribute__((packed)) PacketHeader {
  uint16_t magic;        // 0xA55A
  uint8_t  type;         // PacketType
  uint8_t  flags;
  uint32_t seq;          // per-stream sequence number
  uint16_t payloadLen;   // bytes
  uint16_t dropped;      // samples lost since previous packet (0xFFFF = unknown)
};

struct __attribute__((packed)) StatusPayload {
  float    objTempC;
  float    ambTempC;
  uint8_t  motorRunning;
  uint8_t  faultCode;
  uint32_t uptimeS;
  uint32_t vibPackets;
  uint32_t audioPackets;
  uint16_t fifoOverruns;
  uint32_t txDropped;    // whole packets skipped bcs USB TX buffer was full
  uint8_t  ctrl1XL;      // CTRL1_XL read back from the sensor; host derives FS
};

static SemaphoreHandle_t serialMutex;
static uint32_t seqVib = 0, seqAudio = 0, seqStatus = 0;
static volatile uint16_t fifoOverrunCount = 0;
static volatile uint32_t txDroppedPackets = 0;   // whole packets skipped bcs USB full

static uint16_t crc16ccitt(const uint8_t *d, size_t n, uint16_t crc = 0xFFFF) {
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)d[i] << 8;
    for (int b = 0; b < 8; b++)
      crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : (crc << 1);
  }
  return crc;
}

static void sendPacket(uint8_t type, uint8_t flags, uint32_t seq,
                       const uint8_t *payload, uint16_t len, uint16_t dropped) {
  PacketHeader h;
  h.magic = 0xA55A;
  h.type = type;
  h.flags = flags | (faultCode != FAULT_NONE ? FLAG_FAULTED : 0);
  h.seq = seq;
  h.payloadLen = len;
  h.dropped = dropped;

  uint16_t crc = crc16ccitt((const uint8_t *)&h, sizeof(h));
  crc = crc16ccitt(payload, len, crc);

  const size_t total = sizeof(h) + len + 2;
  xSemaphoreTake(serialMutex, portMAX_DELAY);
  size_t w = Serial.write((const uint8_t *)&h, sizeof(h));
  if (w == sizeof(h)) {
    w += Serial.write(payload, len);
    w += Serial.write((const uint8_t *)&crc, 2);
  }
  if (w != total) txDroppedPackets++;
  xSemaphoreGive(serialMutex);
}

// motor control
// pins are forced safe before anything else happens in setup()
static volatile bool motorRunning = false;

static const gpio_num_t MOTOR_PINS[4] = {
  (gpio_num_t)MOTOR_R_EN, (gpio_num_t)MOTOR_L_EN,
  (gpio_num_t)MOTOR_RPWM, (gpio_num_t)MOTOR_LPWM
};

static void motorPinsSafe() {
  // release a pad hold from a previous fault bcs it survives soft resets
  // otherwise the pinMode/digitalWrite functions silently do nothing
  for (auto p : MOTOR_PINS) gpio_hold_dis(p);
  pinMode(MOTOR_R_EN, OUTPUT); digitalWrite(MOTOR_R_EN, LOW);
  pinMode(MOTOR_L_EN, OUTPUT); digitalWrite(MOTOR_L_EN, LOW);
  pinMode(MOTOR_RPWM, OUTPUT); digitalWrite(MOTOR_RPWM, LOW);
  pinMode(MOTOR_LPWM, OUTPUT); digitalWrite(MOTOR_LPWM, LOW);
}

static void motorKill(uint8_t code) {
  digitalWrite(MOTOR_R_EN, LOW);
  digitalWrite(MOTOR_L_EN, LOW);
  analogWrite(MOTOR_RPWM, 0);
  analogWrite(MOTOR_LPWM, 0);
  // clamp the pads LOW in hardware so nothing short of a power cycle can re-enable the driver after a fault
  for (auto p : MOTOR_PINS) gpio_hold_en(p);
  motorRunning = false;
  if (faultCode == FAULT_NONE) faultCode = code;  // first fault wins, latched
}

static void motorStart(int speed) {          // -255 to 255
  if (faultCode != FAULT_NONE) return;       // latched, no restart until reset
  digitalWrite(MOTOR_R_EN, HIGH);
  digitalWrite(MOTOR_L_EN, HIGH);
  if (speed >= 0) { analogWrite(MOTOR_LPWM, 0);      analogWrite(MOTOR_RPWM, speed);  }
  else            { analogWrite(MOTOR_RPWM, 0);      analogWrite(MOTOR_LPWM, -speed); }
  motorRunning = true;
}

// IIS3DWB
// hardware FIFO, continuous mode, drained in bursts
#define IIS3DWB_WHO_AM_I    0x0F  // must read 0x7B
#define IIS3DWB_FIFO_CTRL3  0x09
#define IIS3DWB_FIFO_CTRL4  0x0A
#define IIS3DWB_CTRL1_XL    0x10
#define IIS3DWB_CTRL3_C     0x12
#define IIS3DWB_FIFO_STAT1  0x3A
#define IIS3DWB_FIFO_STAT2  0x3B
#define IIS3DWB_FIFO_TAG    0x78  // tag + 6 data bytes follow (auto-increment)

static const SPISettings iisSPI(8000000, MSBFIRST, SPI_MODE0);

static void iisWrite(uint8_t reg, uint8_t val) {
  SPI.beginTransaction(iisSPI);
  digitalWrite(IIS3DWB_CS, LOW);
  SPI.transfer(reg & 0x7F);
  SPI.transfer(val);
  digitalWrite(IIS3DWB_CS, HIGH);
  SPI.endTransaction();
}

static uint8_t iisRead(uint8_t reg) {
  SPI.beginTransaction(iisSPI);
  digitalWrite(IIS3DWB_CS, LOW);
  SPI.transfer(reg | 0x80);
  uint8_t v = SPI.transfer(0x00);
  digitalWrite(IIS3DWB_CS, HIGH);
  SPI.endTransaction();
  return v;
}

static bool setupIIS3DWB() {
  pinMode(IIS3DWB_CS, OUTPUT);
  digitalWrite(IIS3DWB_CS, HIGH);
  SPI.begin(IIS3DWB_SCL, IIS3DWB_SDO, IIS3DWB_SDA, IIS3DWB_CS);
  delay(20);

  iisWrite(IIS3DWB_CTRL3_C, 0x01);            // SW_RESET
  delay(20);
  if (iisRead(IIS3DWB_WHO_AM_I) != 0x7B) return false;

  iisWrite(IIS3DWB_CTRL3_C,   0x44);          // BDU + IF_INC
  iisWrite(IIS3DWB_FIFO_CTRL3, 0x0A);         // batch XL at 26.7 kHz
  iisWrite(IIS3DWB_FIFO_CTRL4, 0x06);         // FIFO continuous mode
  iisWrite(IIS3DWB_CTRL1_XL,  IIS3DWB_CTRL1_XL_VAL);   // XL on @ 26.7 kHz, FS per ACCEL_FS_G

  // Verify the full scale actually took. A blind write here is invisible until
  // someone measures 1 g and finds it in the wrong place, by which point the
  // sessions are already recorded. Failing init latches FAULT_VIB_INIT, which
  // holds the motor off, so a mis-scaled run stops before it starts.
  delay(5);
  iisCtrl1XL = iisRead(IIS3DWB_CTRL1_XL);
  if (iisCtrl1XL != IIS3DWB_CTRL1_XL_VAL) return false;

  return true;
}

// returns unread FIFO words
// sets *overrun if the FIFO wrapped (data lost)
static uint16_t fifoStatus(bool *overrun) {
  SPI.beginTransaction(iisSPI);
  digitalWrite(IIS3DWB_CS, LOW);
  SPI.transfer(IIS3DWB_FIFO_STAT1 | 0x80);
  uint8_t s1 = SPI.transfer(0x00);
  uint8_t s2 = SPI.transfer(0x00);
  digitalWrite(IIS3DWB_CS, HIGH);
  SPI.endTransaction();
  *overrun = (s2 & 0x40) != 0;                // FIFO_OVR_IA
  return ((uint16_t)(s2 & 0x03) << 8) | s1;   // DIFF_FIFO[9:0]
}

// CORE 1 vibration task
// drains the FIFO in chunk-word bursts and ships each burst as one packet
// sample loss will show up as the overrun flag + dropped counter instead of just silently vanishing
static void vibTask(void *) {
  esp_task_wdt_add(NULL);
  const int CHUNK = 64;                        // 64 samples = 2.4 ms of data
  static int16_t buf[CHUNK * 3];

  for (;;) {
    esp_task_wdt_reset();
    bool ovr = false;
    uint16_t avail = fifoStatus(&ovr);
    if (ovr) fifoOverrunCount++;

    while (avail >= CHUNK) {
      int n = 0;
      SPI.beginTransaction(iisSPI);
      for (int i = 0; i < CHUNK; i++) {
        digitalWrite(IIS3DWB_CS, LOW);
        SPI.transfer(IIS3DWB_FIFO_TAG | 0x80);
        uint8_t tag = SPI.transfer(0x00);
        uint8_t d[6];
        for (int j = 0; j < 6; j++) d[j] = SPI.transfer(0x00);
        digitalWrite(IIS3DWB_CS, HIGH);
        if (((tag >> 3) & 0x1F) == 0x02) {     // accelerometer data word
          buf[n * 3 + 0] = (int16_t)((d[1] << 8) | d[0]);
          buf[n * 3 + 1] = (int16_t)((d[3] << 8) | d[2]);
          buf[n * 3 + 2] = (int16_t)((d[5] << 8) | d[4]);
          n++;
        }
      }
      SPI.endTransaction();

      sendPacket(PKT_VIB,
                 ovr ? FLAG_FIFO_OVERRUN : 0,
                 seqVib++,
                 (const uint8_t *)buf, n * 6,
                 ovr ? 0xFFFF : 0);            // exact loss unknown on overrun
      ovr = false;
      esp_task_wdt_reset();
      avail = fifoStatus(&ovr);
      if (ovr) fifoOverrunCount++;
    }
    vTaskDelay(1);                           
  }
}

// INMP441
// blocking read lives on core 0 so it doesnt stall the vibration stream
static void setupI2S() {
  const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 512,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  const i2s_pin_config_t pin_config = {
    .mck_io_num = I2S_PIN_NO_CHANGE,   // omitting this zero-inits it = GPIO0!
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_start(I2S_PORT);
}

static void audioTask(void *) {
  esp_task_wdt_add(NULL);
  // 192 samples = 12 ms per packet = 398 bytes on the wire
  // same size class as the vib packets, so all three streams share the link fairly and each packet fits comfortably within one bounded-timeout write
  const int AUDIO_SAMPLES = 192;
  static int32_t raw[AUDIO_SAMPLES];
  static int16_t out[AUDIO_SAMPLES];

  for (;;) {
    esp_task_wdt_reset();
    size_t bytesIn = 0;
    i2s_read(I2S_PORT, raw, sizeof(raw), &bytesIn, portMAX_DELAY);
    int n = bytesIn / sizeof(int32_t);
    for (int i = 0; i < n; i++)
      out[i] = (int16_t)(raw[i] >> 16);        // 24-bit MSB-aligned
    sendPacket(PKT_AUDIO, 0, seqAudio++, (const uint8_t *)out, n * 2, 0);
  }
}

// GY-906
// runs in loop() on core 1
Adafruit_MLX90614 mlx = Adafruit_MLX90614();

static bool  tempSensorOk   = false;
static float objTempC       = NAN;
static float ambTempC       = NAN;
static uint32_t lastTempOkMs = 0;

static void setupMLX90614() {
  Wire.begin(MLX90614_SDA, MLX90614_SCL);
  Wire.setTimeOut(50);                         // a wedged bus must not hang us
  tempSensorOk = mlx.begin();
  lastTempOkMs = millis();
}

// setup
void setup() {
  motorPinsSafe();                      
  serialMutex = xSemaphoreCreateMutex();

  Serial.setTxBufferSize(4096);   // enlarge the TX buffer to smooth out packet bursts
  Serial.begin(3000000);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 5000) delay(10);   // dont wait forever

  // watchdog
  // if any task hangs, the chip resets and the reset-reason check below keeps the motor OFF after that reset instead of blindly restarting
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  esp_task_wdt_config_t wdtCfg = {
    .timeout_ms = WDT_TIMEOUT_MS,
    .idle_core_mask = 0,
    .trigger_panic = true,
  };
  esp_task_wdt_reconfigure(&wdtCfg);
#else
  esp_task_wdt_init(WDT_TIMEOUT_MS / 1000, true);
#endif
  esp_task_wdt_add(NULL);                      // loop() task

  esp_reset_reason_t rr = esp_reset_reason();
  bool cleanBoot = (rr == ESP_RST_POWERON || rr == ESP_RST_SW ||
                    rr == ESP_RST_USB || rr == ESP_RST_UNKNOWN);
  if (!cleanBoot) faultCode = FAULT_BAD_RESET; // WDT/panic/brownout (stay off)

  if (!setupIIS3DWB()) faultCode = (faultCode == FAULT_NONE) ? FAULT_VIB_INIT : faultCode;
  setupMLX90614();
  setupI2S();

  xTaskCreatePinnedToCore(vibTask,   "vib", 6144, NULL, 3, NULL, 1);
  xTaskCreatePinnedToCore(audioTask, "aud", 6144, NULL, 2, NULL, 0);
}

// loop
// motor starts here only when:
//    faultCode == FAULT_NONE (clean boot)   AND   host has sent 'G' (or REQUIRE_HOST_ARM == 0)    AND   the GY-906 has produced at least one valid reading
void loop() {
  static uint32_t lastStatusMs = 0;
  static bool armRequested = (REQUIRE_HOST_ARM == 0);
  static bool tempProven = false;
  static bool motorWasStarted = false;
  esp_task_wdt_reset();

  // host commands
  while (Serial.available()) {
    int c = Serial.read();
    if (c == 'G' || c == 'g') armRequested = true;
    if (c == 'X' || c == 'x') motorKill(FAULT_HOST_STOP);
    if (c == 'R' || c == 'r') {
      // pads stay held LOW through the restart and ESP_RST_SW counts as a clean boot, 
      // so the fault latch is released without touching the physical RESET button
      motorKill(FAULT_HOST_STOP);
      delay(50);
      esp_restart();
    }
  }

  // gated start (armed + interlock proven alive + no fault, once only)
  if (!motorWasStarted && armRequested && tempProven && faultCode == FAULT_NONE) {
    motorStart(MOTOR_SPEED);
    motorWasStarted = true;
  }

  // 1 Hz (temperature + interlocks)
  if (millis() - lastStatusMs >= 1000) {
    lastStatusMs = millis();

    float t = mlx.readObjectTempC();
    float a = mlx.readAmbientTempC();
    if (!isnan(t)) {
      objTempC = t;
      ambTempC = isnan(a) ? ambTempC : a;
      lastTempOkMs = millis();
      tempProven = true;

      // NOTE: this trips on a SINGLE over-threshold reading, with no debounce.
      // That is fail-safe in the correct direction (a spurious low reading
      // cannot suppress a trip -- the next 1 Hz sample would still catch it).
      //
      // But the pilot sessions show the MLX90614 producing roughly 2 spurious
      // single-sample excursions per 6 h run, in which obj and amb move
      // together by 1.5-2.6 C for exactly one sample and then return. At the
      // present ~25 C operating point a 2.6 C excursion is nowhere near the
      // 70 C threshold, so no spurious trip is possible today. Once a
      // mechanical load raises the steady-state case temperature, consider
      // requiring 2 consecutive over-threshold readings before latching --
      // one extra second above the threshold is thermally negligible, and it
      // stops a single glitch from ending an overnight run.
      // Not implemented yet: it is a change to safety-critical behaviour and
      // should be a deliberate decision, not a side effect.
      if (motorRunning && objTempC >= TRIP_TEMP_C)
        motorKill(FAULT_OVERTEMP);
    }

    // stop the motor if the interlock sensor goes silent
    if (motorRunning && millis() - lastTempOkMs > TEMP_STALE_MS)
      motorKill(FAULT_TEMP_SENSOR);

    if (motorRunning && millis() > MAX_RUN_MS)
      motorKill(FAULT_RUNTIME_DONE);

    // status packet
    StatusPayload s;
    s.objTempC      = objTempC;
    s.ambTempC      = ambTempC;
    s.motorRunning  = motorRunning ? 1 : 0;
    s.faultCode     = faultCode;
    s.uptimeS       = millis() / 1000;
    s.vibPackets    = seqVib;
    s.audioPackets  = seqAudio;
    s.fifoOverruns  = fifoOverrunCount;
    s.txDropped     = txDroppedPackets;
    s.ctrl1XL       = iisCtrl1XL;
    sendPacket(PKT_STATUS, 0, seqStatus++, (const uint8_t *)&s, sizeof(s), 0);
  }

  vTaskDelay(pdMS_TO_TICKS(100));
}
