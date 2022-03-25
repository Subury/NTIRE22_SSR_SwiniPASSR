

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()


    # -----------------------------------------
    # super-resolution
    # -----------------------------------------
    if dataset_type in ['sr', 'super-resolution']:
        from data.dataset_sr import DatasetSR as D
    elif dataset_type in ['ssr', 'stereo-super-resolution']:
        from data.dataset_ssr import DatasetSSR as D
    elif dataset_type in ['pssr', 'pyramid-stereo-super-resolution']:
        from data.dataset_pssr import DatasetPSSR as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
