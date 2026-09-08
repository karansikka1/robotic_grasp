run-sim:
	poetry run python run_sim.py

train-v1:
	poetry run python -m v1.train $(ARGS)

teacher-v4:
	poetry run python -m v4.record_trajectory $(ARGS)

split-v4:
	poetry run python -m v4.data $(ARGS)

train-v4-bc:
	poetry run python -m v4.train_bc $(ARGS)
