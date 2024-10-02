import numpy as np
from torchvision import datasets, transforms
import copy




# Split the entire data into public data and users' data

def mnist_noniidcmm(dataset, num_users, num_commondata, alpha):

    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users = {i: np.array([]) for i in range(1, num_users+1)}

    dict_users[0] = set(np.random.choice(all_idxs, num_commondata, replace=False))

    #Exclude the public data from local device
    all_idxs = list(set(all_idxs) - dict_users[0])
    total_data = len(all_idxs)
    
    dict_common = dict_users[0]
    idxs_labels = list(set(all_idxs) - dict_users[0])
    train_labels = dataset.train_labels.numpy()
    dict_users = dirichlet_split_noniid(len(dataset.classes),train_labels, alpha, num_users)

   

    return dict_users, dict_common



def dirichlet_split_noniid(n_classes, train_labels, alpha, n_clients):

    label_distribution = np.random.dirichlet([alpha]*n_clients,n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):

        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            # print(idcs)
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def cifar_noniidcmm(dataset, num_users, num_commondata):

    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i:np.array([]) for i in range(num_users+1)}
    print("type dict users[0]", type(dict_users[0]))
    idxs = np.arange(num_shards * num_imgs)
   

    idxs = list(set(idxs))
    total_data = len(idxs)
    num_shards, num_imgs = 10, total_data//10

    b = []
    for i in idxs:
        b.append(dataset[i][1])


    labels = np.array(b)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # common data
    idx_set = set(range(11))
    for rand in idx_set:
        dict_users[0] = np.concatenate((dict_users[0], idxs[rand*num_imgs:rand*num_imgs+50]), axis=0)
    print("len dict[user]",len(dict_users[0]))
    print("type dict users[0]", type(dict_users[0]))
    print("type dict idxs", type(idxs))
    idxs = list(set(idxs) - set(dict_users[0]))
    dict_common = copy.deepcopy(dict_users[0])
    b = []
    for i in idxs:
        b.append(dataset[i][1])


    labels = np.array(b)


    train_labels = labels
    alpha = 1.0
    dict_users = dirichlet_split_noniid(len(dataset.classes),train_labels, alpha, num_users)


    return dict_users, dict_common




def gtsrb_noniidcmm(dataset, num_users, num_commondata,alpha):
    num_shards, num_imgs = 200, len(dataset)//200

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i:np.array([]) for i in range(num_users+1)}
    print("type dict users[0]", type(dict_users[0]))
    idxs = np.arange(num_shards * num_imgs)
   

    # Exclude the public data from local device
    idxs = list(set(idxs))
    total_data = len(idxs)

    b = []
    for i in idxs:
        b.append(dataset[i][1])


    labels = np.array(b)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # common data
    idx_set = set(range(44))
    for rand in idx_set:
        dict_users[0] = np.concatenate((dict_users[0], idxs[rand*num_imgs:rand*num_imgs+10]), axis=0)

    print("len dict[user]",len(dict_users[0]))
    print("type dict users[0]", type(dict_users[0]))
    print("type dict idxs", type(idxs))
    idxs = list(set(idxs) - set(dict_users[0]))
    b = []
    for i in idxs:
        b.append(dataset[i][1])
    dict_common = copy.deepcopy(dict_users[0])

    labels = np.array(b)

    train_labels = labels
    alpha = 1.0
    print("dataset ")
    dict_users = dirichlet_split_noniid(43,train_labels, alpha, num_users)


    return dict_users, dict_common





if __name__ == '__main__':
    if __name__ == '__main__':
        dataset_train = datasets.MNIST('../data/mnist', train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
