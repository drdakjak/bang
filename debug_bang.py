from bang import bang


if __name__ == "__main__":
    with open("./data/datasets/rcv1.train.vw") as f:
        rows = f.readlines()  # [:500000]
    bang()
    import pickle

    # with open("./data/models/model1.pkl", "rb") as f:
    #     d = pickle.load(f)
    # d
    bang()
