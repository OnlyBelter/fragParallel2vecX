import pandas as pd


ELEMENTS = ['S', 'Br', 'O', 'C', 'F', 'P', 'N', 'I', 'Cl', 'H']
BONDS = ['DOUBLE', 'SINGLE', 'TRIPLE']
SELECTED_MD = ['nN', 'nS', 'nO', 'nX', 'nBondsD', 'nBondsT', 'naRing', 'nARing']
PRMIER_NUM = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def get_class_md_combination(frag_info, selected_md=None, min_number=3):
    """
    get unique class depends on different molecular descriptors
    frag_info: a dataframe which contains fragment smiles, selected_md
    selected_md: selected molecular descriptors
    min_number: the minimal number of fragment in each class
    :return: fragment, class(the combination of different MD, such as 10001010),
             class_id(0 to n), class_num(count each class)
    """
    if not selected_md:
        selected_md = SELECTED_MD
    # md_num = len(selected_md)
    # if md_num <= len(PRMIER_NUM):
    #     unique_code = PRMIER_NUM[:md_num]
    # else:
    #     raise Exception('Please give more primer number to PRMIER_NUM...')
    # frag_info = frag_info.set_index('fragment')
    frag_info = frag_info.loc[:, selected_md].copy()
    frag_info[frag_info >= 1] = 1
    # frag_info = frag_info.apply(lambda x: np.multiply(x, unique_code), axis=1)
    # frag_info[frag_info == 0] = 1
    frag2class = frag_info.apply(lambda x: ''.join([str(i) for i in x]), axis=1)
    frag2class = pd.DataFrame(frag2class, columns=['class'])

    frag_class2num = {}
    for c in frag2class.index:
        class_num = frag2class.loc[c, 'class']
        if class_num not in frag_class2num:
            frag_class2num[class_num] = 0
        frag_class2num[class_num] += 1
    frag_class2num_df = pd.DataFrame.from_dict(frag_class2num, orient='index', columns=['class_num'])
    frag2class = frag2class.merge(frag_class2num_df, left_on='class', right_index=True)
    frag2class = frag2class[frag2class['class_num'] >= min_number].copy()
    print('  >the shape of frag2class after filtered: {}'.format(frag2class.shape))

    unique_class = sorted(frag2class['class'].unique())
    code2id = {unique_class[i]: i for i in range(len(unique_class))}
    print(code2id)
    frag2class['class_id'] = frag2class['class'].apply(lambda x: code2id[x])

    # depth = len(code2id)
    # y_one_hot = tf.one_hot(frag2class_filtered.class_id.values, depth=depth)
    # print('  >the shape of one hot y: {}'.format(y_one_hot.shape))
    return frag2class
