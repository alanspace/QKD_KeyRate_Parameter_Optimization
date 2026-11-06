# Legacy NN loadability check (git `4a61c4d0e6`, torch 2.15.0, separate env `.venv-nn`)

| checkpoint | outputs (fc4 rows) | loads into the 5-output net used by all legacy evaluation scripts |
|---|---|---|
| `NeuralNetwork/models/bb84_nn_model.pth` | 6 | FAILS: Error(s) in loading state_dict for BB84Network5: |
| `NeuralNetwork/models/bb84_nn_model_jax.pth` | 5 | ok |
| `NeuralNetwork/models/bb84_nn_model_legacy.pth` | 5 | ok |
| `models/bb84_nn_model.pth` | 5 | ok |

| scaler | type | n_features_in | n_samples_seen |
|---|---|---|---|
| `NeuralNetwork/models/scaler.pkl` | StandardScaler | 4 | 2117 |
| `NeuralNetwork/models/y_scaler.pkl` | MinMaxScaler | 6 | 2117 |
| `models/scaler.pkl` | StandardScaler | 4 | 2117 |
| `models/y_scaler.pkl` | MinMaxScaler | 5 | 2117 |

Unpickling warnings: Trying to unpickle estimator MinMaxScaler from version 1.7.2 when using version 1.9.1; Trying to unpickle estimator StandardScaler from version 1.7.2 when using version 1.9.1

No file records which scaler pair belongs to which checkpoint; see NN_RETRAIN_PLAN.md.
