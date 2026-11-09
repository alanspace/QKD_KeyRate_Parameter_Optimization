#!/bin/bash
# Mac side: run one remote tick every 120 s. A failed connection just skips that tick.
SSH=(ssh -i "$HOME/.ssh/id_EFP" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o LogLevel=ERROR -o "CertificateFile=$HOME/.ssh/id_login.leonardo.cineca.it-cert.pub" -o BatchMode=yes
     -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 sleung00@login.leonardo.cineca.it)
fails=0; last=""
for i in $(seq 1 240); do            # at most 8 h
  out=$("${SSH[@]}" 'bash $HOME/qkd_driver_tick.sh' </dev/null 2>&1); rc=$?
  if [ $rc -ne 0 ] || ! grep -q '^STATUS' <<<"$out"; then
    fails=$((fails+1)); echo "$(date +%T) tick failed rc=$rc ($fails consecutive): $(head -c 200 <<<"$out")"
    [ $fails -ge 15 ] && { echo "GIVING UP after 15 consecutive failures"; exit 2; }
  else
    fails=0
    grep '^LOG' <<<"$out"
    s=$(grep '^STATUS' <<<"$out"); key=$(cut -d' ' -f3- <<<"$s")
    [ "$key" != "$last" ] && echo "$s"; last=$key
    grep -q 'done=1' <<<"$s" && { echo "ALL DONE"; exit 0; }
    grep -q 'abort=none' <<<"$s" || { echo "ABORT"; exit 1; }
  fi
  sleep 120
done
echo "loop cap reached"; exit 3
