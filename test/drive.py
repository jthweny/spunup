"""Drive the spunup TUI in a pseudo-terminal: python3 test/drive.py "<keys>|<wait>|..."
Keys: DOWN UP ENTER ESC LEFT RIGHT or literal text. Prints the screen after each step."""
import os, pty, sys, time, select, re
KEYS = {"DOWN": "\x1b[B", "UP": "\x1b[A", "ENTER": "\r", "ESC": "\x1b", "LEFT": "\x1b[D", "RIGHT": "\x1b[C", "BS": "\x7f"}
def strip(s):
    s = re.sub(r"\x1b\][^\x07]*\x07", "", s)
    s = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", s)
    return s
pid, fd = pty.fork()
if pid == 0:
    os.environ["SPUNUP_HOME"] = os.environ.get("SPUNUP_HOME", "/tmp/spunup-test-home")
    os.environ["COLUMNS"] = "120"; os.environ["LINES"] = "45"
    os.execvp("node", ["node", os.environ.get("SPUNUP_ENTRY", "dist/cli.js")])
import fcntl, termios, struct
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 45, 120, 0, 0))
def read(t):
    out = b""; end = time.time() + t
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try: out += os.read(fd, 65536)
            except OSError: break
    return out.decode("utf8", "replace")
buf = read(4)
steps = sys.argv[1].split("|") if len(sys.argv) > 1 else []
def show(label, s):
    # show the last full frame: Ink redraws, so keep the tail
    s = strip(s)
    print(f"===== {label}\n" + "\n".join(s.splitlines()[-40:]))
show("start", buf)
for st in steps:
    wait = 1.5
    if st.startswith("WAIT"):
        show(st, read(float(st[4:] or 2))); continue  # keep draining the pty while waiting
    os.write(fd, KEYS.get(st, st).encode())
    show(st, read(wait))
os.write(fd, b"\x03"); time.sleep(0.3)
