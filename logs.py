import pickle


def read_log(log_name: str):
    with open(log_name + '.pkl', 'rb') as log_file:
        data = pickle.load(log_file)
    return data


def log_data(data, log_name: str) -> None:
    with open(log_name + '.pkl', 'wb') as log_file:
        pickle.dump(data, log_file)
