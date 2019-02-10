from bang import bang


if __name__ == "__main__":
    bang(input_path='./data/datasets/rcv1.train.vw', quiet=False, ignore_namespaces=[])
    import pickle

    # with open("./data/models/model1.pkl", "rb") as f:
    #     d = pickle.load(f)
    # d
    bang()
