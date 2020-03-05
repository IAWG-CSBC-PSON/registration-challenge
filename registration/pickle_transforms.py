import SimpleITK as sitk
from pathlib import Path
import pickle

registration_dir = '/home/nhp/Desktop/hackathon/masked_reg'


def pickle_transforms(registration_dir,
                      project_name='NormalBreast',
                      output_name='elx_tforms.pkl'):

    # for NormalBreast data
    reg_paths = sorted(Path(registration_dir).glob("*"))

    elx_tform_dict = {}
    for idx, reg_path in enumerate(reg_paths):
        round_name = '{}_R{}'.format(project_name, reg_path.stem.split('_')[0])
        elx_tform_dict[round_name] = {}
        rig_tform = sorted(reg_path.glob('*.0.txt'))[0]
        nl_tform = sorted(reg_path.glob('*.1.txt'))[0]

        if len(rig_tform) > 0 and len(nl_tform) > 0:
            rig = sitk.ReadParameterFile(str(rig_tform))
            nl = sitk.ReadParameterFile(str(nl_tform))
            elx_tform_dict[round_name]['rigid'] = {}
            elx_tform_dict[round_name]['nl'] = {}

            for k, v in rig.items():
                elx_tform_dict[round_name]['rigid'][k] = v
            for k, v in nl.items():
                elx_tform_dict[round_name]['nl'][k] = v

    with open('elx_tforms.pkl', 'ab') as f:
        pickle.dump(elx_tform_dict, f)


def load_elx_pickle_tforms(pkled_file):
    with open(pkled_file, 'rb') as f:
        tforms = pickle.load(f)
    tform_dict = {}
    for k, v in tforms.items():
        tform_dict[k] = {}
        tform_dict[k]['rigid'] = sitk.ParameterMap()
        tform_dict[k]['nl'] = sitk.ParameterMap()

        for kr, vr in tforms[k]['rigid'].items():
            tform_dict[k]['rigid'][kr] = vr

        for kl, vl in tforms[k]['nl'].items():
            tform_dict[k]['nl'][kl] = vl

apply_tforms(im_dict, tform_dict):
    return
