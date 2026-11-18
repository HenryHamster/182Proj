"""flip-vs-srqrr-001, GPEF below 4 bits/weight (Henry 2026-11-10 'add it'): gpef@e0 with codebook groups of 64/128/256
(team_grp.py; 3 / 2.5 / 2.25 b/w), same protocol as the other methods. Waits for campaign_grp.py to finish (GPU budget).
Pilot: seed-0 grid .01/.03/.09 (gpef@e0 best was .03; edge extension x3 up to 2), seeds 1,2. 10M: grid
{best/3, best, 3 best} around the pilot best (1 extension), seeds 1,2."""
import subprocess, time
while subprocess.run(['pgrep', '-f', '[c]ampaign_grp.py'], capture_output=True).returncode == 0:
    time.sleep(60)
import campaign as C
M = ('gpef@e0g64', 'gpef@e0g128', 'gpef@e0g256')
C.PAR['pilot'] = 4
C.phase('pilot', {m: [0.01, 0.03, 0.09] for m in M}, ext=2)
print('GSZ PILOT DONE', flush=True)
g = {}
for m in M:
    b, _ = C.best('pilot', m)
    g[m] = [round(b / 3, 6), b, round(b * 3, 6)]
print('10M gsz grid', g, flush=True)
C.PAR['10M'] = 4
C.phase('10M', g, ext=1)
print('GSZ DONE', flush=True)
