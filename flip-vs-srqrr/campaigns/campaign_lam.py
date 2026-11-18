import campaign as C
C.PAR['10M'] = 4
G3 = [0.01, 0.03, 0.09]
C.phase('10M', {f'gpefr@g256_after_l{l}': G3 for l in ('0.1', '0.3')}, ext=1)
print('LAM DONE', flush=True)
