run_etl:
	python -m src.main run_etl

etl_mnist:
	PYTHONPATH=. python src/scripts/etl_process/run_etl.py mnist

run_treevi:
	PYTHONPATH=. python src/scripts/reconstruction/run_treevi_training.py

run_geovae:
	PYTHONPATH=. python src/scripts/reconstruction/run_geovae_training.py

run_vqvae:
	PYTHONPATH=. python src/scripts/reconstruction/run_vqvae_training.py

run_vae:
	PYTHONPATH=. python src/scripts/reconstruction/run_vae_training.py

jupyter:
	uv run --with jupyter --active jupyter lab

reproduce: run_etl run_barlow_twins run_byol run_barlow_twins_finetune run_byol_finetune  run_treevi run_pygtreevi run_vqvae run_vae
