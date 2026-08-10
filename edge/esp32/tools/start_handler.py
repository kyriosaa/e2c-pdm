# start/stop controller for the motor data collection rig
#
# commands:
#   start   reboot the ESP32 over serial ('R'), then launch data_catcher.py.
#           the reboot clears any latched fault (so no need to press RESET) and
#           zeroes the firmware uptime, so device uptime and host session
#           timestamps start together
#   stop    stop data_catcher properly (it finalizes audio.wav / status.csv /
#           session.json and sends 'X' so the motor stops). if the catcher
#           isn't running, sends 'X' directly to confirm the motor is off
#   status  show whether data_catcher is running
#   auto    run sessions back to back, unattended: start one, wait for it to
#           finish, wait AUTO_GAP_S, start the next. Repeats until 'stop' or
#           'quit'. The prompt stays live while it runs, so 'status' works
#   quit    stop everything and exit (Ctrl+C does the same thing btw)

import os
import signal
import subprocess
import sys
import threading
import time

import serial

# every knob for this rig lives in config.py -- edit that, not this file.
# data_catcher.py reads the same SERIAL_PORT, so the two cannot drift apart.
from config import (SERIAL_PORT, BAUD_RATE, REBOOT_WAIT_S, AUTO_GAP_S,
                    AUTO_MIN_SESSION_S, AUTO_POLL_S)

HERE    = os.path.dirname(os.path.abspath(__file__))
CATCHER = os.path.join(HERE, 'data_catcher.py')

proc = None                # either running data_catcher subprocess or None
proc_lock = threading.RLock()          # auto thread and prompt both touch proc
auto_thread = None
auto_cancel = threading.Event()


def send_cmd(cmd: bytes, why: str) -> bool:
    s = serial.Serial()
    s.port = SERIAL_PORT
    s.baudrate = BAUD_RATE
    s.timeout = 1
    s.write_timeout = 2
    s.dtr = False         
    s.rts = False
    try:
        s.open()
        s.write(cmd)
        s.flush()
        return True
    except (serial.SerialException, OSError) as e:
        print(f'!! could not open {SERIAL_PORT} to {why}: {e.__class__.__name__}: {e}')
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def reap():
    """Notice a catcher that exited on its own (run timeout, crash, ...)."""
    global proc
    with proc_lock:
        if proc is not None and proc.poll() is not None:
            print(f'(data_catcher exited on its own, code {proc.returncode})')
            proc = None


def catcher_running() -> bool:
    reap()
    with proc_lock:
        return proc is not None


def do_start(quiet: bool = False) -> bool:
    """Reboot the board and launch one data_catcher. True if it started."""
    global proc
    if catcher_running():
        print(f"data_catcher is already running (pid {proc.pid}) — 'stop' it first.")
        return False
    if not quiet:
        print('Rebooting ESP32 (clears any latched fault, uptime restarts at 0)...')
    if not send_cmd(b'R', 'reboot the board'):
        print("Close any serial monitor / stray data_catcher and try 'start' again.")
        return False
    time.sleep(REBOOT_WAIT_S)
    with proc_lock:
        proc = subprocess.Popen(
            [sys.executable, '-u', CATCHER],
            cwd=HERE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        print(f'data_catcher started (pid {proc.pid}).')
    return True


def _fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    return f'{seconds // 3600}h{(seconds % 3600) // 60:02d}m'


def auto_loop():
    """Run sessions back to back until cancelled or something goes wrong."""
    n = 0
    while not auto_cancel.is_set():
        n += 1
        print(f'\n[auto] session {n} starting...')
        started = time.time()
        if not do_start(quiet=True):
            print('[auto] could not start a session — auto mode off.')
            break

        # Wait for this session to finish. Poll rather than proc.wait() so the
        # sleep stays cancellable from the prompt.
        while not auto_cancel.is_set():
            with proc_lock:
                p = proc
            if p is None or p.poll() is not None:
                break
            auto_cancel.wait(AUTO_POLL_S)
        if auto_cancel.is_set():
            break

        ran = time.time() - started
        reap()
        if ran < AUTO_MIN_SESSION_S:
            print(f'[auto] session {n} lasted only {_fmt_hms(ran)} — that is not a '
                  f'real run. Auto mode off; check the last data_catcher output.')
            break

        print(f'[auto] session {n} done ({_fmt_hms(ran)}). '
              f'Next starts in {_fmt_hms(AUTO_GAP_S)} '
              f'at {time.strftime("%H:%M", time.localtime(time.time() + AUTO_GAP_S))}.')
        if auto_cancel.wait(AUTO_GAP_S):
            break
    print('[auto] stopped.')


def do_auto():
    global auto_thread
    if auto_thread is not None and auto_thread.is_alive():
        print("auto mode is already running — 'stop' to end it.")
        return
    if catcher_running():
        print("a session is already running — 'stop' it first, then 'auto'.")
        return
    auto_cancel.clear()
    auto_thread = threading.Thread(target=auto_loop, daemon=True)
    auto_thread.start()
    print(f'Auto mode on: session, then {_fmt_hms(AUTO_GAP_S)} gap, repeat. '
          f"'stop' ends it.")


def stop_auto():
    global auto_thread
    if auto_thread is not None and auto_thread.is_alive():
        auto_cancel.set()
        auto_thread.join(timeout=10)
    auto_thread = None


def do_stop():
    global proc
    stop_auto()
    if not catcher_running():
        print("data_catcher is not running — sending 'X' to confirm the motor is off.")
        if send_cmd(b'X', 'stop the motor'):
            print("Stop confirmed. Motor is held off until the next 'start'.")
        return
    with proc_lock:
        p = proc
    print(f'Stopping data_catcher (pid {p.pid})...')
    p.send_signal(signal.CTRL_BREAK_EVENT)
    try:
        p.wait(timeout=30)
        print('data_catcher stopped cleanly.')
    except subprocess.TimeoutExpired:
        print('!! data_catcher did not exit within 30 s — killing it.')
        p.kill()
        p.wait()
        send_cmd(b'X', 'stop the motor')   # catcher died before it could send it
    with proc_lock:
        proc = None


def main():
    print('Motor rig controller — commands: start | auto | stop | status | quit')
    while True:
        try:
            cmd = input('> ').strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            cmd = 'quit'
        if cmd == 'start':
            do_start()
        elif cmd == 'auto':
            do_auto()
        elif cmd == 'stop':
            do_stop()
        elif cmd == 'status':
            auto_on = auto_thread is not None and auto_thread.is_alive()
            if catcher_running():
                print(f'data_catcher running (pid {proc.pid})'
                      f'{"  [auto mode on]" if auto_on else ""}')
            elif auto_on:
                print('data_catcher not running — auto mode on, waiting out the gap')
            else:
                print('data_catcher not running')
        elif cmd in ('quit', 'exit', 'q'):
            stop_auto()
            if catcher_running():
                do_stop()
            return
        elif cmd:
            print('commands: start | auto | stop | status | quit')


if __name__ == '__main__':
    main()
