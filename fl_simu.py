import torch
import flwr as fl
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold, KFold

from clients.cr_client import CRClient
from clients.hcr_client import HCRClient
from clients.ts_client import TSClient
from net_archs import MLP, LSTMModel, MLP2, LSTMModel2

def mlp_simulation(split, x_train, y_train, x_test, y_test, dir='default', num_epoch=256, batch_size=16, num_rounds=15):
  client_fn = fl_split(split, x_train, y_train, x_test, y_test, MLP(22, 21), CRClient, num_epoch, batch_size, 5e-5, 1e-4)
  return _run_simulation(client_fn, len(split), MLP(22, 21), upper_dir='model_checkpoints', dir=dir, num_rounds=num_rounds)

def lstm_simulation(split, x_train, y_train, x_test, y_test, dir='default', num_epoch=256, batch_size=4, num_rounds=20):
  client_fn = fl_split(split, x_train, y_train, x_test, y_test, LSTMModel(23, 21), CRClient, num_epoch, batch_size, 1e-4, 1e-5)
  return _run_simulation(client_fn, len(split), LSTMModel(23, 21), upper_dir='lstm_model_checkpoints', dir=dir, num_rounds=num_rounds)

def mlp_simulation2(split, x_train, y_train, x_test, y_test, dir='default', num_epoch=256, batch_size=16, num_rounds=15, dropout=True, dropout_rate=0.2):
  client_fn = fl_split(split, x_train, y_train, x_test, y_test, MLP2(120, 1, dropout, dropout_rate), HCRClient, num_epoch, batch_size, 5e-5, 5e-5)
  return _run_simulation(client_fn, len(split), MLP2(120, 1, dropout, dropout_rate), upper_dir='model_checkpoints2', dir=dir, num_rounds=num_rounds)

def lstm_simulation2(split, nts_train, ts1_train, ts2_train, ts3_train, ts4_train, y_train, nts_test, ts1_test, ts2_test, ts3_test, ts4_test, y_test, dir='default', num_epoch=256, batch_size=16, num_rounds=15, dropout=True, dropout_rate=0.2):
  client_fn = fl_split2(split, nts_train, ts1_train, ts2_train, ts3_train, ts4_train, y_train, nts_test, ts1_test, ts2_test, ts3_test, ts4_test, y_test, LSTMModel2(dropout, dropout_rate), TSClient, num_epoch, batch_size, 5e-6, 5e-5)
  return _run_simulation(client_fn, len(split), LSTMModel2(dropout, dropout_rate), upper_dir='lstm_model_checkpoints2', dir=dir, num_rounds=num_rounds)

def fl_split2(split, nts_train, ts1_train, ts2_train, ts3_train, ts4_train, y_train, nts_test, ts1_test, ts2_test, ts3_test, ts4_test, y_test, net_arch, client, num_epoch, batch_size, lr, weight_decay):
  nts_eq = []
  ts1_eq = []
  ts2_eq = []
  ts3_eq = []
  ts4_eq = []
  y_eq = []
  # skf = StratifiedKFold(n_splits=10)
  skf = KFold(n_splits=10)
  skf.get_n_splits(nts_train, y_train)
  for i, (_, test_index) in enumerate(skf.split(nts_train, y_train)):
      nts_eq.append(nts_train[test_index])
      ts1_eq.append(ts1_train[test_index])
      ts2_eq.append(ts2_train[test_index])
      ts3_eq.append(ts3_train[test_index])
      ts4_eq.append(ts4_train[test_index])
      y_eq.append(y_train[test_index])

  x_split = []
  y_split = []

  acc = 0
  for s in split:
      x_split.append((
          torch.cat(nts_eq[acc:acc+int(s*10)], 0),
          torch.cat(ts1_eq[acc:acc+int(s*10)], 0),
          torch.cat(ts2_eq[acc:acc+int(s*10)], 0),
          torch.cat(ts3_eq[acc:acc+int(s*10)], 0),
          torch.cat(ts4_eq[acc:acc+int(s*10)], 0)))
      y_split.append(torch.cat(y_eq[acc:acc+int(s*10)], 0))
      acc += int(s*10)

  def client_fn(cid):
      net = net_arch
      optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
      return TSClient(net,
                      optimizer,
                      x_train=x_split[int(cid)],
                      y_train=y_split[int(cid)],
                      x_test=(nts_test, ts1_test, ts2_test, ts3_test, ts4_test),
                      y_test=y_test,
                      cid=cid,
                      num_epoch=num_epoch,
                      batch_size=batch_size)
  return client_fn

def fl_split(split, x_train, y_train, x_test, y_test, net_arch, client, num_epoch, batch_size, lr, weight_decay):
    x_eq = []
    y_eq = []
    skf = StratifiedKFold(n_splits=10)
    # skf = KFold(n_splits=10)
    skf.get_n_splits(x_train, y_train)
    for i, (_, test_index) in enumerate(skf.split(x_train, y_train)):
        x_eq.append(x_train[test_index])
        y_eq.append(y_train[test_index])

    x_split = []
    y_split = []

    acc = 0
    for s in split:
        x_split.append(torch.cat(x_eq[acc:acc+int(s*10)], 0))
        y_split.append(torch.cat(y_eq[acc:acc+int(s*10)], 0))
        acc += int(s*10)

    def client_fn(cid):
        # net = net_arch(22, 21)
        net = net_arch
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        return client(net,
                      optimizer,
                      x_train=x_split[int(cid)],
                      y_train=y_split[int(cid)],
                      x_test=x_test,
                      y_test=y_test,
                      cid=cid,
                      num_epoch=num_epoch,
                      batch_size=batch_size)
    return client_fn

def _run_simulation(client_fn, num_clients, net_arch, upper_dir='default', dir='default', num_rounds=15):
    model_fl = net_arch

    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round,
            results,
            failures,
        ):
            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super(
            ).aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                print(f"Saving round {server_round} aggregated_parameters...")

                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays = fl.common.parameters_to_ndarrays(
                    aggregated_parameters)

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(
                    model_fl.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict(
                    {k: torch.tensor(v) for k, v in params_dict})
                model_fl.load_state_dict(state_dict, strict=True)

                # Save the model
                torch.save(model_fl.state_dict(
                ), f"{upper_dir}/{dir}/model_round_{server_round}.pth")

            return aggregated_parameters, aggregated_metrics

    # Create FedAvg strategy
    strategy = SaveModelStrategy(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=num_clients,
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None

    fl.simulation.start_simulation(
        client_fn=client_fn,
        clients_ids=[str(x) for x in range(num_clients)],
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={'log_to_driver': False}
    )

    latest_round_file = f'{upper_dir}/{dir}/model_round_{num_rounds}.pth'
    state_dict = torch.load(latest_round_file)
    model_fl.load_state_dict(state_dict)

    return model_fl