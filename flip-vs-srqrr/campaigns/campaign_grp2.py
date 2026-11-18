"""flip-vs-srqrr-001: gpef_sr@e0 at 10M had its best seed-0 lr (.027) at the edge after the one extension; run .081
(and .243 if that wins), then seeds 1,2 at the resulting best, so its reported lr is interior like every other method."""
import campaign as C
C.PAR['10M'] = 1
C.phase('10M', {'gpef_sr@e0': [0.081]}, ext=1)
print('GRP2 DONE', C.best('10M', 'gpef_sr@e0'), flush=True)
