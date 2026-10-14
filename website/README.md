# RoTA-Bench website

The bilingual portal contains the task game, leaderboard, HTML paper figures,
method documentation, and the signed paper PDF. It renders on CPU and needs no
model API key. Formal evaluation remains separate in `../src/`.

```bash
pip install -r website/requirements.txt
python website/prepare_tasks.py
python website/app.py
```

The default port is 7860 (override with `PORT`). The projection renderer listens
internally on localhost:8788. Run one portal process per container. The game
uses the fixed test split, six rotation directions, 30/10/5/1 degree angles,
automatic success at IoU >= 0.90, and a 48-decision limit. Answer paths are shown
only as demo references, not as formal evaluation inputs.

See `MODELSCOPE_DEPLOY.md` for the private free-CPU Studio configuration.
