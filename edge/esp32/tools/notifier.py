# ============================================================================
# Email alerts for the motor rig — Gmail SMTP, stdlib only
#
# One-time setup (Gmail blocks plain passwords for SMTP, you need an
# "app password"):
#   1. Google Account -> Security -> turn on 2-Step Verification (if not on).
#   2. Google Account -> Security -> "App passwords" -> create one for "Mail".
#      You get a 16-character password like "abcd efgh ijkl mnop".
#   3. Put it in private.py next to this script (untracked via .gitignore, so
#      it never reaches git). Spaces in the password are fine:
#        SMTP_HOST         = 'smtp.gmail.com'
#        SMTP_PORT         = 465
#        EMAIL_ADDRESS     = 'you@gmail.com'      # sender AND recipient
#        SMTP_APP_PASSWORD = 'abcd efgh ijkl mnop'
#
# Test from a terminal:
#   python notifier.py                      -> sends a test email
#   python notifier.py session <path.json>  -> emails a session.json summary
#
# Design notes for callers (data_catcher.py):
#   - send() never raises and never blocks the capture loop: delivery runs on
#     a daemon thread and all network errors are swallowed with a console note.
#   - Pass wait=True only at shutdown (session summary), where blocking a few
#     seconds is fine and a daemon thread would be killed before delivering.
#   - With no app password configured everything is a no-op (one warning).
# ============================================================================

import json
import os
import smtplib
import sys
import threading
from email.message import EmailMessage
try:
    from private import SMTP_HOST, SMTP_PORT, EMAIL_ADDRESS, SMTP_APP_PASSWORD
except ImportError:
    SMTP_HOST, SMTP_PORT = 'smtp.gmail.com', 465
    EMAIL_ADDRESS = ''
    SMTP_APP_PASSWORD = ''

from config import SMTP_TIMEOUT_S as TIMEOUT_S

_warned_unconfigured = False

# fault codes that are intentional endings, not real faults
# (firmware still latches the motor off, but nothing went wrong)
CLEAN_ENDINGS = {'NONE', 'RUNTIME_CAP_REACHED', 'HOST_STOP'}

def _post(subject, body):
    msg = EmailMessage()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_ADDRESS
    msg['Subject'] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=TIMEOUT_S) as s:
        s.login(EMAIL_ADDRESS, SMTP_APP_PASSWORD.replace(' ', ''))
        s.send_message(msg)


def _try_post(subject, body):
    try:
        _post(subject, body)
    except Exception as e:
        print(f'notifier: could not deliver "{subject}" '
              f'({e.__class__.__name__}: {e})')


def send(subject, body, wait=False):
    global _warned_unconfigured
    if not (SMTP_APP_PASSWORD and EMAIL_ADDRESS):
        if not _warned_unconfigured:
            _warned_unconfigured = True
            print('notifier: email alerts disabled — create private.py with '
                  'EMAIL_ADDRESS and SMTP_APP_PASSWORD (see header comment).')
        return
    if wait:
        _try_post(subject, body)
    else:
        threading.Thread(target=_try_post, args=(subject, body),
                         daemon=True).start()


# ---------------------------------------------------------------------------
# Rig events — called by data_catcher.py
# ---------------------------------------------------------------------------
def motor_started(obj_temp_c, amb_temp_c, uptime_s):
    send('[motor rig] Motor started',
         f'obj {obj_temp_c:.1f} °C / amb {amb_temp_c:.1f} °C, '
         f'board uptime {uptime_s} s')


def motor_stopped(obj_temp_c, amb_temp_c, uptime_s):
    send('[motor rig] Motor stopped',
         f'obj {obj_temp_c:.1f} °C / amb {amb_temp_c:.1f} °C, '
         f'board uptime {uptime_s} s')


def fault(fault_name, obj_temp_c):
    if fault_name in CLEAN_ENDINGS:
        send(f'[motor rig] Run ended: {fault_name}',
             f'Firmware ended the run as planned ({fault_name}) at obj '
             f'{obj_temp_c:.1f} °C.\n'
             f'Motor is held off until hardware reset.')
    else:
        send(f'[motor rig] FAULT: {fault_name}',
             f'Firmware latched {fault_name} at obj {obj_temp_c:.1f} °C.\n'
             f'Motor is held off until hardware reset.')


def session_summary(summary):
    faulted = summary.get('last_fault', 'NONE') not in CLEAN_ENDINGS
    prefix = 'FAULTED session' if faulted else 'Session complete'
    send(f"[motor rig] {prefix} — {summary.get('session', '?')}",
         json.dumps(summary, indent=2),
         wait=True)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'session':
        with open(sys.argv[2]) as f:
            session_summary(json.load(f))
    else:
        send('[motor rig] Test alert', 'notifier.py is working.', wait=True)
    if SMTP_APP_PASSWORD:
        print('Done (check your inbox).')
