# Toy run (real output)

`python -m tenor.demo`, run on a 32-core Linux machine with Python 3.13 and NumPy 2.3:

```text
TENOR toy: pull requests (toy), 30 past changes, 1500 runs/day/version, landmark 14 days
slope gamma 1.109: the index explains 90% of an effect
margin kappa 2.45 pts, screen tolerance tau 3.49 pts; 25 past changes pass the screen
past changes the screen flags: 3, 4, 9, 18, 19

release  kind       effect   TENOR                 wait for the outcome  proxy only
31       honest      -2.2    escalate              block    20.5d  ok    block     3.3d  ok
32       honest      +5.3    ship      4.2d  ok    ship     15.2d  ok    ship      0.5d  ok
33       honest      +3.7    ship     27.7d  ok    ship     15.2d  ok    ship      0.8d  ok
34       hacked      -4.4    block    26.8d  ok    block    14.9d  ok    ship      0.1d  WRONG
35       unhack      +1.9    escalate              ship     18.5d  ok    block     0.5d  WRONG
36       honest     +21.2    ship      2.9d  ok    ship     14.2d  ok    ship      0.1d  ok
37       latent      -5.6    ship      4.8d  WRONG block    14.5d  ok    ship      0.5d  WRONG

wrong decisions: TENOR 1, waiting 0, proxy 3 (of 7); 3.5s on CPU
```
