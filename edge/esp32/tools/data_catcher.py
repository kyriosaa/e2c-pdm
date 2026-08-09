# data catcher for the motor data collection rig
# 
# streams:
#   type 0 VIB    payload = N * (int16 x, int16 y, int16 z)  raw counts
#                 scale depends on ACCEL_FS_G in config.py (must match firmware):
#                   +/-4 g  -> 0.122 mg/LSB   (current)
#                   +/-16 g -> 0.488 mg/LSB   (pilot sessions 2026-07-16..19)
#                 g = raw * ACCEL_FS_G / 32768 ;  m/s^2 = g * 9.80665
#                 the value used is recorded in session.json as accel_fs_g,
#                 read back from the sensor's CTRL1_XL rather than declared
#   type 1 AUDIO  payload = N * int16, 16 kHz mono
#   type 2 STATUS objTempC(f32) ambTempC(f32) motorRunning(u8) faultCode(u8)
#                 uptimeS(u32) vibPackets(u32) audioPackets(u32)
#                 fifoOverruns(u16) txDropped(u32) ctrl1XL(u8)
#
# outputs:
#   vibration.bin       raw int16 LE, 3 columns  -> np.fromfile(f, '<i2').reshape(-1,3)
#   audio.wav           16 kHz 16-bit mono
#   status.csv          1 Hz temperature / fault / counter log
#   packet_index.csv    host arrival time of every packet (for cross-stream alignment)
#   session.json        metadata + integrity summary (writes after session end)

import serial
import struct
import csv
import json
import signal
import time
import os
import wave
from datetime import datetime

# every knob for this rig lives in config.py -- edit that, not this file
from config import (SERIAL_PORT, BAUD_RATE, RAW_ROOT, SESSION_LABEL,
                    RUN_SECONDS, ARM_DELAY_S, ACCEL_FS_G, VIB_ODR_NOMINAL_HZ,
                    AUDIO_SR_HZ, MOTOR_VOLTAGE_V, MECHANICAL_LOAD,
                    FAULT_CONDITION, ROOM_CONDITION, NOTES)

if hasattr(signal, 'SIGBREAK'):
    def _sigbreak_to_kbint(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGBREAK, _sigbreak_to_kbint)

try:
    from private import COLOR as C
except ImportError:
    C = {k: '' for k in ('RED', 'GREEN', 'BLUE', 'WHITE', 'YELLOW',
                         'MAGENTA', 'CYAN', 'GRAY', 'ORANGE', 'RESET')}

try:
    import notifier
except ImportError:
    class notifier:  # phone alerts will be silently disabled if notifier.py is missing 
        motor_started = motor_stopped = fault = session_summary = \
            staticmethod(lambda *a, **k: None)

MAGIC        = 0xA55A
HDR_FMT      = '<HBBIHH'          # magic, type, flags, seq, payloadLen, dropped
HDR_SIZE     = struct.calcsize(HDR_FMT)
STATUS_FMT   = '<ffBBIIIHIB'      # ... + ctrl1XL read back from the IIS3DWB
STATUS_SIZE  = struct.calcsize(STATUS_FMT)
STATUS_FMT_V2  = '<ffBBIIIHI'     # older firmware without the CTRL1_XL readback
STATUS_SIZE_V2 = struct.calcsize(STATUS_FMT_V2)
STATUS_FMT_V1  = '<ffBBIIIH'      # ... and without the txdrop field
STATUS_SIZE_V1 = struct.calcsize(STATUS_FMT_V1)

# IIS3DWB CTRL1_XL (10h) bits [3:2] = FS[1:0]_XL.
# DS12569 Rev 8, Table 30, p.32.
FS_XL_TO_G = {0b00: 2.0, 0b01: 16.0, 0b10: 4.0, 0b11: 8.0}


def fs_g_from_ctrl1(ctrl1: int):
    """Full scale the sensor reports, or None if the firmware did not send it."""
    if ctrl1 is None or ctrl1 == 0xFF:      # 0xFF = firmware never read it back
        return None
    return FS_XL_TO_G[(ctrl1 >> 2) & 0b11]
MAX_PAYLOAD  = 4096               # sanity bound for resync

PKT_VIB, PKT_AUDIO, PKT_STATUS = 0, 1, 2

FAULT_NAMES = {
    0: 'NONE',
    1: 'OVERTEMP',
    2: 'TEMP_SENSOR_LOST',
    3: 'RUNTIME_CAP_REACHED',
    4: 'BAD_RESET_MOTOR_HELD_OFF',
    5: 'VIB_SENSOR_INIT_FAILED',
    6: 'HOST_STOP',
}

FLAG_FIFO_OVERRUN = 0x01
FLAG_FAULTED      = 0x02


# table CRC16-CCITT
# the bit loop version costs around 0.8ms per packet in Python
# at ~900 packets/s would eat most of a core and there might be some host-side data loss
# table ver is around 20x faster
_CRC16_TABLE = []
for _i in range(256):
    _c = _i << 8
    for _ in range(8):
        _c = ((_c << 1) ^ 0x1021) & 0xFFFF if _c & 0x8000 else (_c << 1) & 0xFFFF
    _CRC16_TABLE.append(_c)


def crc16_ccitt(data, crc: int = 0xFFFF) -> int:
    tbl = _CRC16_TABLE
    for b in data:
        crc = ((crc << 8) & 0xFF00) ^ tbl[((crc >> 8) ^ b) & 0xFF]
    return crc


# session outputs
started_at = datetime.now()
# Keep `stamp` a bare %Y%m%d_%H%M%S: it goes into session.json as "started",
# and ml/sessions.py resolves the legacy +/-16 g full scale by STRING-comparing
# that value against ACCEL_FS_CHANGED_AFTER. Any prefix breaks the ordering and
# the wrong full scale is silently applied.
stamp = started_at.strftime('%Y%m%d_%H%M%S')
# Weekday is for the directory name only, and goes AFTER the label: ml's
# operating_point() keys on the first '_'-separated field, so the label has to
# stay first. Overnight runs start one day and end the next -- this is the start.
session_dir = os.path.join(
    RAW_ROOT, f'{SESSION_LABEL}_{stamp}_{started_at.strftime("%a")}')
os.makedirs(session_dir, exist_ok=True)

vib_file = open(os.path.join(session_dir, 'vibration.bin'), 'wb')

audio_wav = wave.open(os.path.join(session_dir, 'audio.wav'), 'wb')
audio_wav.setnchannels(1)
audio_wav.setsampwidth(2)
audio_wav.setframerate(AUDIO_SR_HZ)

status_csv = open(os.path.join(session_dir, 'status.csv'), 'w', newline='')
status_writer = csv.writer(status_csv)
status_writer.writerow(['host_time', 'obj_temp_c', 'amb_temp_c', 'motor_running',
                        'fault', 'uptime_s', 'vib_packets', 'audio_packets',
                        'fifo_overruns', 'tx_dropped_packets'])

index_csv = open(os.path.join(session_dir, 'packet_index.csv'), 'w', newline='')
index_writer = csv.writer(index_csv)
index_writer.writerow(['host_time', 'type', 'seq', 'n_samples', 'flags', 'dropped'])

# CH343 UART bridge
# on the bridge, the DTR/RTS lines are held LOW bcs they are wired to the board's reset circuit
# holding them low prevents connect-time resets
def open_port(port, retries=2):
    for attempt in range(retries + 1):
        s = serial.Serial()
        s.port = port
        s.baudrate = BAUD_RATE
        s.timeout = 1
        s.write_timeout = 2        # timeout for dead handle
        s.dtr = False              # bridge: hold reset lines inactive
        s.rts = False
        try:
            s.open()
            time.sleep(2)          # let any settle/boot finish
            _ = s.in_waiting       # probe: raises if the port is dead
            return s
        except (serial.SerialException, OSError) as e:
            try:
                s.close()
            except Exception:
                pass
            if attempt < retries:
                print(f'Port not ready ({e.__class__.__name__}); retrying in 3 s...')
                time.sleep(3)
    return None


ser = open_port(SERIAL_PORT)
if ser is None:
    print(f'Error: could not get a working connection on {SERIAL_PORT}.')
    print('Unplug/replug the USB (or press RESET), close any serial monitors, retry.')
    raise SystemExit(1)

print(f'Connected to {SERIAL_PORT}.')
ser.reset_input_buffer()
arm_time = time.time() + ARM_DELAY_S
print(f'Motor will arm in {ARM_DELAY_S} s — turn on the 24 V supply and leave '
      f'the room. Sensors are recording now.')

# stats
stats = {
    'packets':      {PKT_VIB: 0, PKT_AUDIO: 0, PKT_STATUS: 0},
    'seq_gaps':     {PKT_VIB: 0, PKT_AUDIO: 0, PKT_STATUS: 0},
    'crc_errors':   0,
    'resyncs':      0,
    'fifo_overrun_packets': 0,
    'tx_dropped_packets': 0,
    'device_resets': 0,
    'ctrl1_xl': None,             # CTRL1_XL as the sensor reports it
    'accel_fs_g_reported': None,  # full scale derived from it
    'vib_samples':  0,
    'audio_samples': 0,
    'last_fault':   'NONE',
}
last_seq = {}


def note_seq(ptype: int, seq: int):
    prev = last_seq.get(ptype)
    if prev is not None and seq != (prev + 1) & 0xFFFFFFFF:
        if seq < prev:
            # sequence went backwards = the device rebooted, not packet loss.
            # Counted so it lands in session.json: every recording so far has
            # one of these in the first ~3 s (the port-open reset), and a reset
            # is a hard discontinuity in the sample stream that downstream
            # windowing has to split on.
            stats['device_resets'] += 1
            print(f"{C['YELLOW']}!! stream {ptype} sequence restarted — "
                  f"device rebooted{C['RESET']}")
        else:
            gap = (seq - prev - 1) & 0xFFFFFFFF
            stats['seq_gaps'][ptype] += 1
            print(f"{C['RED']}!! seq gap on stream {ptype}: "
                  f"missed ~{gap} packet(s) after seq {prev}{C['RESET']}")
    last_seq[ptype] = seq


def handle_packet(ptype, flags, seq, dropped, payload, host_time):
    note_seq(ptype, seq)
    stats['packets'][ptype] = stats['packets'].get(ptype, 0) + 1
    n_samples = 0

    if ptype == PKT_VIB:
        vib_file.write(payload)
        n_samples = len(payload) // 6
        stats['vib_samples'] += n_samples
        if flags & FLAG_FIFO_OVERRUN:
            stats['fifo_overrun_packets'] += 1
            print(f"{C['YELLOW']}!! IIS3DWB FIFO overrun (seq {seq}) — "
                  f"samples lost on-device{C['RESET']}")

    elif ptype == PKT_AUDIO:
        audio_wav.writeframes(payload)
        n_samples = len(payload) // 2
        stats['audio_samples'] += n_samples

    elif ptype == PKT_STATUS and len(payload) in (STATUS_SIZE, STATUS_SIZE_V2,
                                                  STATUS_SIZE_V1):
        ctrl1 = None
        if len(payload) == STATUS_SIZE:
            (obj_t, amb_t, running, fault, uptime,
             vib_pkts, aud_pkts, overruns, tx_drops,
             ctrl1) = struct.unpack(STATUS_FMT, payload)
        elif len(payload) == STATUS_SIZE_V2:
            (obj_t, amb_t, running, fault, uptime,
             vib_pkts, aud_pkts, overruns, tx_drops) = struct.unpack(STATUS_FMT_V2, payload)
        else:
            (obj_t, amb_t, running, fault, uptime,
             vib_pkts, aud_pkts, overruns) = struct.unpack(STATUS_FMT_V1, payload)
            tx_drops = 0
        if ctrl1 is None and not stats.get('_old_fw_warned'):
            stats['_old_fw_warned'] = True
            print(f"{C['YELLOW']}!! board is running older firmware (no CTRL1_XL "
                  f"readback) — the full scale in session.json is DECLARED, not "
                  f"measured. Reflash edge_asset_monitor.ino{C['RESET']}")

        # The declared ACCEL_FS_G is what this file was told to expect; ctrl1 is
        # what the sensor reports. They diverge when the firmware constant is
        # edited but the build is never flashed, which is how the 2026-08-01..02
        # sessions came to claim +/-4 g while recording at +/-16 g. Caught here
        # the run has not armed yet (ARM_DELAY_S), so nothing is lost.
        fs_reported = fs_g_from_ctrl1(ctrl1)
        if fs_reported is not None:
            stats['ctrl1_xl'] = ctrl1
            stats['accel_fs_g_reported'] = fs_reported
            if fs_reported != ACCEL_FS_G and not stats.get('_fs_mismatch'):
                stats['_fs_mismatch'] = True
                print(f"{C['RED']}!! FULL-SCALE MISMATCH — aborting before arm.\n"
                      f"   config.py ACCEL_FS_G = +/-{ACCEL_FS_G:g} g\n"
                      f"   sensor CTRL1_XL = 0x{ctrl1:02X} -> +/-{fs_reported:g} g\n"
                      f"   The board is not running the firmware you think it is. "
                      f"Reflash, then re-run.{C['RESET']}")

        fault_name = FAULT_NAMES.get(fault, f'UNKNOWN({fault})')
        stats['last_fault'] = fault_name

        # alerts go off only on state transitions so no spam
        prev_running = stats.get('_last_running')
        stats['_last_running'] = running
        if prev_running is not None and running != prev_running:
            if running:
                notifier.motor_started(obj_t, amb_t, uptime)
            else:
                notifier.motor_stopped(obj_t, amb_t, uptime)
        prev_fault = stats.get('_last_fault_code', 0)
        stats['_last_fault_code'] = fault
        if fault != 0 and fault != prev_fault:
            notifier.fault(fault_name, obj_t)

        # firmware counters run since BOOT, not since session start
        # baseline the first reading so a board that hasn't power-cycled doesn't show stale drops from an earlier session as a current problem
        if '_tx0' not in stats:
            stats['_tx0'] = tx_drops
        if tx_drops < stats['_tx0']:          # counter restarted: device rebooted
            stats['_tx0'] = 0
        tx_sess = tx_drops - stats['_tx0']
        stats['tx_dropped_packets'] = tx_sess

        status_writer.writerow([f'{host_time:.3f}', f'{obj_t:.2f}', f'{amb_t:.2f}',
                                running, fault_name, uptime, vib_pkts, aud_pkts,
                                overruns, tx_drops])
        status_csv.flush()

        colour = C['GREEN'] if fault == 0 else C['RED']
        print(f"{C['GRAY']}[{uptime:>6}s]{C['RESET']}  "
              f"{C['YELLOW']}obj {obj_t:5.1f}C{C['RESET']}  "
              f"{C['CYAN']}amb {amb_t:5.1f}C{C['RESET']}  "
              f"{C['MAGENTA']}motor {'ON ' if running else 'OFF'}{C['RESET']}  "
              f"{colour}{fault_name}{C['RESET']}  "
              f"{C['BLUE']}vib {vib_pkts}{C['RESET']}  "
              f"{C['GREEN']}aud {aud_pkts}{C['RESET']}  "
              f"{C['RED']}ovr {stats['fifo_overrun_packets']}{C['RESET']}  "
              f"{C['ORANGE']}txdrop {tx_sess}{C['RESET']}")
        if tx_sess:
            print(f"{C['YELLOW']}!! firmware skipped {tx_sess} whole packet(s) "
                  f"this session: USB TX buffer full — link is saturating"
                  f"{C['RESET']}")
        if stats.get('_fs_mismatch'):
            raise RuntimeError(
                f'full-scale mismatch: ACCEL_FS_G=+/-{ACCEL_FS_G:g} g but sensor '
                f'reports +/-{stats["accel_fs_g_reported"]:g} g — reflash the board')
        if running == 0 and fault == 0 and time.time() >= arm_time:
            try:
                ser.write(b'G')   # arm/re-arm until the motor confirms running
            except serial.SerialTimeoutException:
                pass
        if fault not in (0,):
            print(f"{colour}>> firmware fault latched: {fault_name} — "
                  f"motor is held off until hardware reset{C['RESET']}")

    index_writer.writerow([f'{host_time:.4f}', ptype, seq, n_samples, flags, dropped])

print(f'Recording to {session_dir}')
print('Ctrl+C to stop.\n')

buf = bytearray()
start_time = time.time()
last_flush = time.time()

try:
    while time.time() - start_time < RUN_SECONDS:
        try:
            chunk = ser.read(4096)
        except (serial.SerialException, OSError):
            # USB dropped mid-run
            # retry the port until time runs out
            print(f"{C['RED']}!! serial link lost — reconnecting (outage data "
                  f"is lost and will show as seq gaps){C['RESET']}")
            try:
                ser.close()
            except Exception:
                pass
            buf.clear()
            ser = None
            while ser is None and time.time() - start_time < RUN_SECONDS:
                time.sleep(5)
                ser = open_port(SERIAL_PORT, retries=0)
            if ser is None:
                print('Could not reconnect before session end.')
                break
            print('Reconnected.')
            if time.time() >= arm_time:
                try:
                    ser.write(b'G')        # re-arm in case the board rebooted
                except Exception:
                    pass
            continue

        if chunk:
            buf.extend(chunk)

        # periodic flush
        if time.time() - last_flush >= 5:
            last_flush = time.time()
            vib_file.flush()
            index_csv.flush()

        pos = 0
        blen = len(buf)
        while True:
            idx = buf.find(b'\x5A\xA5', pos)     # 0xA55A little-endian
            if idx < 0:
                pos = max(pos, blen - 1)         # keep last byte (split magic)
                break
            if idx > pos:
                stats['resyncs'] += 1            # skipped garbage before magic
            pos = idx
            if blen - pos < HDR_SIZE:
                break                            # wait for full header

            magic, ptype, flags, seq, plen, dropped = struct.unpack_from(HDR_FMT, buf, pos)
            if plen > MAX_PAYLOAD or ptype > PKT_STATUS:
                stats['resyncs'] += 1            # false magic, keep scanning
                pos += 2
                continue
            total = HDR_SIZE + plen + 2
            if blen - pos < total:
                break                            # wait for the rest

            payload = bytes(buf[pos + HDR_SIZE:pos + HDR_SIZE + plen])
            (rx_crc,) = struct.unpack_from('<H', buf, pos + HDR_SIZE + plen)
            calc = crc16_ccitt(buf[pos:pos + HDR_SIZE])
            calc = crc16_ccitt(payload, calc)
            if calc != rx_crc:
                stats['crc_errors'] += 1
                pos += 2                         # bad frame, rescan from next byte
                continue

            handle_packet(ptype, flags, seq, dropped, payload, time.time())
            pos += total

        if pos:
            del buf[:pos]                        # one compaction per chunk

except KeyboardInterrupt:
    print('\nRecording manually stopped.')
except Exception as e:
    # Never die without finalizing the WAV header and writing session.json.
    print(f'\nUnexpected error: {e!r} — closing out session files.')

# end + summary
# if end command is sent (Ctrl+C, timeout, or error), stop the motor if possible
if ser is not None:
    try:
        ser.write(b'X')
        time.sleep(0.5)   # let the stop land and the final status packets drain
    except Exception:
        pass
vib_file.close()
audio_wav.close()
status_csv.close()
index_csv.close()
if ser is not None:
    try:
        ser.close()
    except Exception:
        pass

elapsed = time.time() - start_time
summary = {
    'session': SESSION_LABEL,
    'started': stamp,
    'elapsed_s': round(elapsed, 1),

    # --- acquisition provenance (see notes at the top of this file)
    # accel_fs_g is the value the SENSOR reported via CTRL1_XL whenever the
    # firmware is new enough to send it, and only falls back to the declared
    # constant for older firmware. accel_fs_g_source says which, so a session
    # recorded against unverified firmware is identifiable after the fact.
    'accel_fs_g': (stats['accel_fs_g_reported']
                   if stats['accel_fs_g_reported'] is not None else ACCEL_FS_G),
    'accel_fs_g_declared': ACCEL_FS_G,
    'accel_fs_g_source': ('hardware_readback'
                          if stats['accel_fs_g_reported'] is not None else 'declared'),
    'ctrl1_xl': (None if stats['ctrl1_xl'] is None
                 else f"0x{stats['ctrl1_xl']:02X}"),
    'accel_mg_per_lsb': round(
        (stats['accel_fs_g_reported']
         if stats['accel_fs_g_reported'] is not None else ACCEL_FS_G)
        / 32768.0 * 1000.0, 6),
    'vib_odr_nominal_hz': VIB_ODR_NOMINAL_HZ,
    'audio_sr_hz': AUDIO_SR_HZ,
    'motor_voltage_v': MOTOR_VOLTAGE_V,
    'mechanical_load': MECHANICAL_LOAD,
    'fault_condition': FAULT_CONDITION,
    'room_condition': ROOM_CONDITION,
    'notes': NOTES,

    'vib_samples': stats['vib_samples'],
    # NOTE: this ratio divides by the NOMINAL ODR, so it also absorbs any error
    # in the sensor's actual output data rate. The pilot sessions read ~0.995
    # here, which looked like 0.5 % packet loss but was almost entirely the
    # IIS3DWB running ~0.46 % slow (measured 26533-26547 Hz). Real transport
    # loss was ~0.001 %. For a true integrity figure use the sequence-derived
    # ratio from ml/sessions.py, not this one.
    'vib_expected_at_26k7': int(elapsed * VIB_ODR_NOMINAL_HZ),
    'vib_capture_ratio': round(
        stats['vib_samples'] / max(1, elapsed * VIB_ODR_NOMINAL_HZ), 4),
    'audio_samples': stats['audio_samples'],
    'audio_expected_at_16k': int(elapsed * AUDIO_SR_HZ),
    'packets': {str(k): v for k, v in stats['packets'].items()},
    'seq_gaps': {str(k): v for k, v in stats['seq_gaps'].items()},
    'crc_errors': stats['crc_errors'],
    'resyncs': stats['resyncs'],
    'device_resets': stats['device_resets'],
    'fifo_overrun_packets': stats['fifo_overrun_packets'],
    'tx_dropped_packets': stats['tx_dropped_packets'],
    'last_fault': stats['last_fault'],
}
with open(os.path.join(session_dir, 'session.json'), 'w') as f:
    json.dump(summary, f, indent=2)
notifier.session_summary(summary)   # blocking send bcs script is exiting anyway

print(json.dumps(summary, indent=2))
print(f'Saved to {session_dir}')