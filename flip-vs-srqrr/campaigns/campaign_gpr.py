"""flip-vs-srqrr-001, rotation on GPEF's refit codebooks (Henry 2026-11-11 'we should prob run rotation on the refit
codebooks'). 10M, campaign.py protocol (seed-0 grid of 3, one x3 edge extension, seeds 1,2 at the seed-0 best).
gpefr@g256_after / _before at 2.25 b/w vs gpef@e0g256 2.220 and srq_rot@b0 2.334; gpefr@g32_before at 4 b/w vs
gpef@e0 1.889 and gpef_rot@e0 (= gpefr@g32_after) 2.261. Starts after campaign_gpx.py exits (GPU sharing)."""
import subprocess, time
import campaign as C

while subprocess.run(["pgrep", "-f", "[c]ampaign_gpx.py"], capture_output=True).returncode == 0:
    time.sleep(60)
C.PAR['10M'] = 4
G3 = [0.01, 0.03, 0.09]
C.phase('10M', {'gpefr@g256_after': G3, 'gpefr@g256_before': G3, 'gpefr@g32_before': G3}, ext=1)
print('GPR DONE', flush=True)
