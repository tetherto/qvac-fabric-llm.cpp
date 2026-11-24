#!/bin/bash
# Forced command for a localhost-only SSH key: cron runs "ssh -i ~/.ssh/bench_localhost localhost weekly", so the
# bench runs at the priority sshd sessions get. Jobs cron starts directly decode about 25-30% slower on macOS.
# Setup: ssh-keygen -t ed25519 -N '' -f ~/.ssh/bench_localhost, then add its .pub to ~/.ssh/authorized_keys as
#   from="127.0.0.1,::1",command="/bin/bash <bench dir>/ssh-entry.sh",no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty <key>
W=${BENCH_DIR:-$HOME/bench-weekly}
TV=${TV_DIR:-$HOME/bench-turbovec}
case "${SSH_ORIGINAL_COMMAND:-}" in
  weekly)
    /bin/bash $W/weekly.sh >> $W/cron.log 2>&1
    # optional: the TurboVec CPU bench after the LLM bench, if it is installed
    [ -f $TV/turbovec.sh ] && /bin/bash $TV/turbovec.sh --after-llm >> $TV/cron.log 2>&1 ;;
  *) echo "ssh-entry.sh: unknown command" >&2; exit 1 ;;
esac
