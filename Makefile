run_etl:
	PYTHONPATH=. python src/scripts/etl_process/run_etl.py

etl_mnist:
	PYTHONPATH=. python src/scripts/etl_process/run_etl.py mnist

etl_fmnist:
	PYTHONPATH=. python src/scripts/etl_process/run_etl.py fmnist

etl_stl10:
	PYTHONPATH=. python src/scripts/etl_process/run_etl.py stl10

etl_celeba:
	PYTHONPATH=. python src/scripts/etl_process/run_etl.py thetthetyee/celaba-face-recognition

run_treevi:
	PYTHONPATH=. python src/scripts/reconstruction/run_treevi_training.py

run_geovae:
	PYTHONPATH=. python src/scripts/reconstruction/run_geovae_training.py

run_dummy_geovae:
	PYTHONPATH=. python src/scripts/reconstruction/run_dummy_geovae_training.py

run_vqvae:
	PYTHONPATH=. python src/scripts/reconstruction/run_vqvae_training.py

run_vae:
	PYTHONPATH=. python src/scripts/reconstruction/run_vae_training.py

jupyter:
	uv run --with jupyter --active jupyter lab

reproduce: 
	run_etl run_barlow_twins run_byol run_barlow_twins_finetune run_byol_finetune  run_treevi run_pygtreevi run_vqvae run_vae

test:
	PYTHONPATH=. python src/scripts/test_reconstruction/evaluate_all_models.py --models_dir models/reconstruction/ --output_csv geo_vae_results.csv

run_multiple:
	PYTHONPATH=. python src/scripts/reconstruction/run_multiple.py