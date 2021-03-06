def readfile(filename):
    data = []
    with open(filename, "r") as r:
        while True:
            l1 = r.readline().strip()
            l2 = r.readline().strip()
            l3 = r.readline().strip()
            if l1 == "":
                break
            data.append((l1.split(), [w.lower() for w in l1.split()], l2.split(), l3.split()))
    return data



def get_from_ix(w, to_ix, unk):
    if w in to_ix:
        return to_ix[w]

    assert unk != -1, "no unk supported"
    return unk


def data2instance(trn_data, ixes):
    instances = []
    for one in trn_data:
        instances.append([])
        for i in range(len(ixes)):
            instances[-1].append([get_from_ix(w, ixes[i][0], ixes[i][1]) for w in one[i]])
    return instances


def data2instance_orig(trn_data, ixes):
    instances = []
    wix, wunk = ixes[0]
    pix, punk = ixes[1]
    lix, lunk = ixes[2]
    tix, tunk = ixes[3]

    for words, pretrains, lemmas, tags in trn_data:  # one[0] for words; one[1] for pretrains, one[2] for lemmas; one[4] for tags
        instance = [[] for i in range(4)]

        assert len(words) == len(pretrains) == len(lemmas) == len(tags)

        for i in range(len(words)):
            tmp = []
            ws = words[i].split("~")
            for w in ws:
                tmp.append(get_from_ix(w, wix, wunk))
            instance[0].append(tmp)

            tmp = []
            ps = pretrains[i].split("~")
            for p in ps:
                tmp.append(get_from_ix(p, pix, punk))
            instance[1].append(tmp)

            tmp = []
            ls = lemmas[i].split("~")
            for l in ls:
                tmp.append(get_from_ix(l, lix, lunk))
            instance[2].append(tmp)

            instance[3].append(get_from_ix(tags[i], tix, tunk))
        instances.append(instance)
    return instances


def packed_data(trn_instances, batch_size):
    packed_instances = []
    packed_instance = []
    packed_lengths = []
    packed_length = []
    packed_idx = 0
    max_length = len(trn_instances[0][0])
    for instance in trn_instances:
        if packed_idx != 0 and packed_idx % 100 == 0:
            packed_instances.append(packed_instance)
            packed_lengths.append(packed_length)
            packed_instance = []
            packed_length = []
        if len(packed_instance) == 0:
            max_length = len(instance[0])

        packed_length.append(len(instance[0]))
        instance[0] += [0 for i in range(max_length - len(instance[0]))]
        instance[1] += [0 for i in range(max_length - len(instance[1]))]
        instance[2] += [0 for i in range(max_length - len(instance[2]))]
        instance[3] += [0 for i in range(max_length - len(instance[3]))]
        packed_instance.append(instance)

        packed_idx += 1
    if len(packed_instance) != 0:
        packed_instances.append(packed_instance)
        packed_lengths.append(packed_length)
    return packed_instances, packed_lengths
