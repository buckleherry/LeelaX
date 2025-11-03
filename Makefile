.PHONY: venv self train eval tb
venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
self:
	. .venv/bin/activate && python -m leelax.selfplay.worker --episodes 50
train:
	. .venv/bin/activate && python -m leelax.train.loop --steps 2000
eval:
	. .venv/bin/activate && python -m leelax.eval.arena --games 40
tb:
	. .venv/bin/activate && tensorboard --logdir runs

