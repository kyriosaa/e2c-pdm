# Every setting for this rig that changes between runs, in one place.
#
# Edit this file, not the scripts. data_catcher.py, start_handler.py and
# export_sessions.py all read from here, so they can no longer disagree with
# each other -- which they did: start_handler.py sat on COM3 while
# data_catcher.py had been moved to COM6, and COM3 opens perfectly well on this
# machine (it is Intel AMT serial-over-LAN), so 'start' reported that it had
# rebooted the board while writing 'R' into nothing.
#
# What is deliberately NOT here: the wire protocol (packet header layout, status
# struct, fault codes, CTRL1_XL bit decoding) and the export manifest format.
# Those are contracts with the firmware and with files already on disk. They
# change when the firmware or the format changes, not when you set up a run, and
# they belong beside the code that parses them.
#
# SMTP credentials stay in private.py, which is gitignored. This file is
# tracked, so nothing secret goes in it.

import os

# ---------------------------------------------------------------------------
# Serial link                            data_catcher.py + start_handler.py
# ---------------------------------------------------------------------------
# List the ports with:  python -m serial.tools.list_ports -v
# The board is the CH343 USB bridge. Do not guess from the number -- COM1 and
# COM3 on this machine are a motherboard port and Intel AMT SOL, and both open
# without error while being attached to nothing.
SERIAL_PORT = 'COM6'
BAUD_RATE   = 3000000

# ---------------------------------------------------------------------------
# Where recordings land                  data_catcher.py + export_sessions.py
# ---------------------------------------------------------------------------
# Anchored to this file so it resolves the same regardless of cwd. The catcher
# writes here; export_sessions.py reads from here.
RAW_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'raw'))

# ---------------------------------------------------------------------------
# One session                                             data_catcher.py
# ---------------------------------------------------------------------------
# Motor replaced 2026-08-09. The new part is rated 12 V / 20 000 rpm nominal,
# 1.4 A no-load, 5.5 A starting. Everything recorded before that date is a
# DIFFERENT MOTOR and is not a baseline for anything recorded after it.
SESSION_LABEL = 'healthy12V'
# SESSION_LABEL = 'test'
RUN_SECONDS   = int(2 * 3600)       # 2hr run -- short enough that a cell sits
                                    # inside one ambient condition. A 6 h run
                                    # started in the evening ends in the small
                                    # hours and spans both.
ARM_DELAY_S   = 60                  # 60s wait time before motor starts running

# ---------------------------------------------------------------------------
# Unattended back-to-back sessions                       start_handler.py
# ---------------------------------------------------------------------------
REBOOT_WAIT_S = 4                   # settle time after 'R' before launching

# Gap between the end of one auto session and the start of the next. Lets the
# motor and the driver cool back to ambient so every session starts from the
# same thermal state instead of inheriting the previous run's heat.
AUTO_GAP_S = int(1 * 3600)

# A session that ends this fast did not run -- data_catcher aborted (bad port,
# full-scale mismatch, latched fault). Without this, auto mode would happily
# restart a broken run every hour all night and produce nothing.
AUTO_MIN_SESSION_S = 300

# How often the auto thread checks whether the session has ended. Keep it well
# below AUTO_MIN_SESSION_S or a short failed run gets rounded up past the guard.
AUTO_POLL_S = 2

# ---------------------------------------------------------------------------
# Acquisition provenance written into session.json        data_catcher.py
# ---------------------------------------------------------------------------
# ACCEL_FS_G MUST match ACCEL_FS_G / IIS3DWB_CTRL1_XL_VAL in the firmware.
# Recorded per session so analysis code can scale each session correctly instead
# of hardcoding a full scale that silently goes stale when the register changes.
# The six pilot sessions of 2026-07-16..20 predate this field and were recorded
# at +/-16 g; ml/sessions.py falls back for those by start timestamp.
#
# This is the DECLARED value. Since 2026-08-04 the firmware reads CTRL1_XL back
# and ships it in every status packet, and session.json records that instead;
# this constant now serves as the cross-check that aborts the run when the two
# disagree. The 2026-08-01..02 sessions are why: they declare +/-4 g and were
# recorded at +/-16 g, because this constant was edited and the board was not
# reflashed. A declared constant is not evidence of anything.
ACCEL_FS_G    = 4.0
VIB_ODR_NOMINAL_HZ = 26667.0        # IIS3DWB datasheet nominal
AUDIO_SR_HZ   = 16000

# ---------------------------------------------------------------------------
# Physical setup notes                                    data_catcher.py
# ---------------------------------------------------------------------------
# Free-text notes that belong with the recording rather than in a lab notebook.
# Fill these in before each run -- they end up in session.json and are the only
# record of the physical setup.
MOTOR_VOLTAGE_V = 12.0              # rated voltage of the 2026-08-09 motor
MECHANICAL_LOAD = 'none'            # e.g. 'none', 'eddy_brake_gap8mm'
                                    # record the magnet gap -- it IS the load
                                    # setting. NOT a flywheel: pure inertia adds
                                    # no steady-state torque.
FAULT_CONDITION = 'healthy'         # e.g. 'healthy', 'imbalance_l1', 'bearing_worn'
ROOM_CONDITION  = 'unlabelled'      # quiet | noise | unlabelled
NOTES           = ''

# Fixed vocabulary, not free text. It has already drifted once -- the first
# provenanced sessions were written 'quiet_night'/'noisy_day_hvac' and the
# vocabulary settled on 'quiet'/'noise' on 2026-08-04. session.json is the
# authoritative label, so a typo here is a mislabelled session that nobody
# notices until analysis. To add a level, add it here deliberately.
#
# 'unlabelled' is the default and is deliberate: back-to-back auto runs cross
# ambient conditions on their own schedule, so the room is judged per session
# afterwards rather than asserted up front. It reads as "pending", which an
# empty string does not -- that reads as "forgot".
ROOM_CONDITIONS = ('quiet', 'noise', 'unlabelled')
if ROOM_CONDITION not in ROOM_CONDITIONS:
    raise SystemExit(f'ROOM_CONDITION={ROOM_CONDITION!r} is not one of '
                     f'{ROOM_CONDITIONS} -- fix it before recording.')

# ---------------------------------------------------------------------------
# Transfer to the analysis machine                      export_sessions.py
# ---------------------------------------------------------------------------
COPY_CHUNK = 8 * 1024 * 1024   # 8 MiB: large enough that USB throughput, not
                               # syscall overhead, is the limit

# ---------------------------------------------------------------------------
# Email alerts                                                notifier.py
# ---------------------------------------------------------------------------
# Credentials are NOT here -- see private.py. This is just the network timeout.
SMTP_TIMEOUT_S = 15

# ---------------------------------------------------------------------------
# Terminal colour            data_catcher.py + edge/rpi/pi_edge_twin.py
# ---------------------------------------------------------------------------
# These lived in private.py until 2026-08-09. Nothing about them is secret, and
# keeping them in an untracked file is what let them drift out of step with the
# code: private.py defined six of these keys while data_catcher.py's status line
# indexed nine, so the first status packet of every run raised KeyError('GRAY')
# and ended the session ~1 s in. Tracked here, a missing key is visible in the
# diff instead of at 3 a.m. Add a key here BEFORE using it in a format string.
COLOR = {
    'RED':     '\033[91m',
    'GREEN':   '\033[92m',
    'BLUE':    '\033[94m',
    'YELLOW':  '\033[93m',
    'WHITE':   '\033[37m',
    'CYAN':    '\033[96m',
    'MAGENTA': '\033[95m',
    'GRAY':    '\033[90m',
    'ORANGE':  '\033[38;5;208m',
    'RESET':   '\033[0m',
}

# Set False to strip the escape codes -- for a terminal that does not handle
# them, or when piping a run to a log file. Replaces the old behaviour where
# colour was implicitly off whenever private.py was missing.
USE_COLOR = True
if not USE_COLOR:
    COLOR = {k: '' for k in COLOR}
