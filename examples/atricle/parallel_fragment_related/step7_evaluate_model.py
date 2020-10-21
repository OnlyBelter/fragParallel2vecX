import os
import pandas as pd
from fragpara2vec.mlp import predict, evaluate


if __name__ == '__main__':
    root_dir = r'F:\github\fragParallel2vecX\big_data'
    sub_dir1 = '08_train_mol_vec_by_mlp'
    result_dir = '09_evaluate_model'
    # selected_mol2md_file_path = os.path.join(result_dir, 'selected_cid2md.csv')
    test_set_file_path = os.path.join(root_dir, sub_dir1, 'test_set.csv')
    pred_acc_list = []
    for frag_type in ['tandem', 'parallel', 'Mol2vec', 'random']:
        print('>>> evaluate the performance of model {}'.format(frag_type))
        mlp_model_result_dir = os.path.join(root_dir, sub_dir1, 'pre_trained_model',
                                            frag_type)
        mlp_model_file_path = os.path.join(mlp_model_result_dir, 'model_reg_{}.h5'.format(frag_type))
        frag_method = 'tree_decomposition'
        if frag_type == 'Mol2vec':
            frag_method = 'fingerprinting'
            mol2vec_file_name = 'mol_vec_{}_frag.csv'.format(frag_type)
        elif frag_type == 'random':
            frag_method = 'random'
            mol2vec_file_name = 'mol_vec_{}_frag.csv'.format(frag_type)
        else:
            mol2vec_file_name = 'mol_vec_{}_frag_minn_{}_maxn_{}.csv'.format(frag_type, 1, 2)

        mol2vec_file_path = os.path.join(root_dir, sub_dir1, frag_method, mol2vec_file_name)
        y_pred = predict(mlp_model_file_path, test_set_file_path, mol2vec_file_path)
        y_true = pd.read_csv(test_set_file_path, usecols=['cid', 'md_class'], index_col=0, dtype={'md_class': str})
        pred_acc = evaluate(y_true=y_true, y_pred=y_pred, model_name=frag_type)
        pred_acc.to_csv(os.path.join(root_dir, result_dir, 'pred_acc_{}.csv'.format(frag_type)))
        pred_acc_list.append(pred_acc)
    pred_acc_all = pd.concat(pred_acc_list, axis=1)
    pred_acc_all['>parallel'] = pred_acc_all['pred_acc_Mol2vec'] > pred_acc_all['pred_acc_parallel']
    pred_acc_all['>tandem'] = pred_acc_all['pred_acc_Mol2vec'] > pred_acc_all['pred_acc_tandem']
    pred_acc_all.to_csv(os.path.join(root_dir, result_dir, 'pred_acc_all.csv'))
