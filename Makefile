run-sim:
	poetry run python run_sim.py

train-v1:
	poetry run python -m v1.train $(ARGS)
