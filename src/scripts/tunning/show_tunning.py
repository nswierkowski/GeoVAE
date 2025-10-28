import optuna
import pandas as pd

# === Load study ===
storage = "sqlite:///models/reconstruction/geovae_optuna/optuna_study.db"
study_name = "GeoVAE_BayesOpt"
study = optuna.load_study(study_name=study_name, storage=storage)

# === Summary ===
print(f"\n=== OPTUNA STUDY SUMMARY ===")
print(f"Study name: {study_name}")
print(f"Number of trials: {len(study.trials)}")
print(f"Best trial number: {study.best_trial.number}")
print(f"Best value (val_loss): {study.best_value:.4f}\n")

# === Print all trials with results ===
print("=== ALL TRIALS ===")
records = []
for trial in study.trials:
    if trial.state.name != "COMPLETE":
        continue  # skip pruned or failed trials
    record = {"trial_number": trial.number, "val_loss": trial.value}
    record.update(trial.params)
    records.append(record)

df = pd.DataFrame(records).sort_values(by="val_loss")
print(df.to_string(index=False))

# === Optionally, save to CSV ===
out_path = "models/reconstruction/geovae_optuna/optuna_all_results.csv"
df.to_csv(out_path, index=False)
print(f"\n✅ All trial results saved to: {out_path}")

# === Print best trial details again (explicitly) ===
print("\n=== BEST TRIAL DETAILS ===")
print(f"Trial number: {study.best_trial.number}")
print(f"Validation loss: {study.best_value:.6f}")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")
