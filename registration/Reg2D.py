import SimpleITK as sitk
from pathlib import Path
import pickle


def register_2D_images(source_image, source_image_res, target_image,
                       target_image_res, reg_models, reg_output_fp):
    """ Register 2D images with multiple models and return a list of elastix
        transformation maps.

    Parameters
    ----------
    source_image : str
        file path to the image that will be aligned
    source_image_res : float
        pixel resolution of the source image(e.g., 0.25 um /px)
    target_image : str
        file path to the image to which source_image will be aligned
    source_image_res : float
        pixel resolution of the target image(e.g., 0.25 um /px)
    reg_models : list
        python list of file paths to elastix paramter files
    reg_output_fp : type
        where to place elastix registration data: transforms and iteration info

    Returns
    -------
    list
        list of elastix transforms for aligning subsequent images

    """

    source = sitk.ReadImage(source_image)
    target = sitk.ReadImage(target_image)

    source.SetSpacing((source_image_res, source_image_res))
    target.SetSpacing((target_image_res, target_image_res))

    try:
        selx = sitk.SimpleElastix()
    except AttributeError:
        selx = sitk.ElastixImageFilter()

    selx.LogToConsoleOn()

    selx.SetOutputDirectory(reg_output_fp)

    # if source_image_type == target_image_type:
    #     matcher = sitk.HistogramMatchingImageFilter()
    #     matcher.SetNumberOfHistogramLevels(64)
    #     matcher.SetNumberOfMatchPoints(7)
    #     matcher.ThresholdAtMeanIntensityOn()
    #     source.image = matcher.Execute(source.image, target.image)

    selx.SetMovingImage(source)
    selx.SetFixedImage(target)

    for idx, model in enumerate(reg_models):
        if idx == 0:
            pmap = sitk.ReadParameterFile(model)
            pmap["WriteResultImage"] = ("false", )
            #            pmap['MaximumNumberOfIterations'] = ('10', )

            selx.SetParameterMap(pmap)
        else:
            pmap = sitk.ReadParameterFile(model)
            pmap["WriteResultImage"] = ("false", )
            #            pmap['MaximumNumberOfIterations'] = ('10', )

            selx.AddParameterMap(pmap)

    selx.LogToFileOn()

    # execute registration:
    selx.Execute()

    return list(selx.GetTransformParameterMap())


def transform_2D_image(source_image,
                       source_image_res,
                       transformation_maps,
                       im_output_fp,
                       write_image=False):
    """Transform 2D images with multiple models and return the transformed image
        or write the transformed image to disk as a .tif file.

    Parameters
    ----------
    source_image : str
        file path to the image that will be transformed
    source_image_res : float
        pixel resolution of the source image(e.g., 0.25 um /px)
    transformation_maps : list
        python list of file paths to elastix parameter files
    im_output_fp : str
        output file path
    write_image : bool
        whether to write image or return it as python object

    Returns
    -------
    type
        Description of returned object.

    """

    try:
        transformix = sitk.SimpleTransformix()
    except AttributeError:
        transformix = sitk.TransformixImageFilter()

    if isinstance(source_image, sitk.Image):
        image = source_image
    else:
        print("reading image from file")
        image = sitk.ReadImage(source_image)
        image.SetSpacing((source_image_res, source_image_res))

    for idx, tmap in enumerate(transformation_maps):

        if idx == 0:
            if isinstance(tmap, sitk.ParameterMap) is False:
                tmap = sitk.ReadParameterFile(tmap)
            tmap["InitialTransformParametersFileName"] = (
                "NoInitialTransform", )
            transformix.SetTransformParameterMap(tmap)
            tmap["ResampleInterpolator"] = (
                "FinalNearestNeighborInterpolator", )
        else:
            if isinstance(tmap, sitk.ParameterMap) is False:
                tmap = sitk.ReadParameterFile(tmap)
            tmap["InitialTransformParametersFileName"] = (
                "NoInitialTransform", )
            tmap["ResampleInterpolator"] = (
                "FinalNearestNeighborInterpolator", )

            transformix.AddTransformParameterMap(tmap)

    #take care for RGB images
    pixelID = image.GetPixelID()

    transformix.LogToConsoleOn()
    transformix.LogToFileOn()
    transformix.SetOutputDirectory(str(Path(im_output_fp).parent))

    if pixelID in list(range(1, 13)) and image.GetDepth() == 0:
        transformix.SetMovingImage(image)
        image = transformix.Execute()
        image = sitk.Cast(image, pixelID)

    elif pixelID in list(range(1, 13)) and image.GetDepth() > 0:
        images = []
        for chan in range(image.GetDepth()):
            transformix.SetMovingImage(image[:, :, chan])
            images.append(sitk.Cast(transformix.Execute(), pixelID))
        image = sitk.JoinSeries(images)
        image = sitk.Cast(image, pixelID)

    elif pixelID > 12:
        images = []
        for idx in range(image.GetNumberOfComponentsPerPixel()):
            im = sitk.VectorIndexSelectionCast(image, idx)
            pixelID_nonvec = im.GetPixelID()
            transformix.SetMovingImage(im)
            images.append(sitk.Cast(transformix.Execute(), pixelID_nonvec))
            del im

        image = sitk.Compose(images)
        image = sitk.Cast(image, pixelID)

    if write_image is True:
        sitk.WriteImage(image, im_output_fp + "_registered.tif", True)
        return
    else:
        return image


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
    return tform_dict


# im_dir = '/home/nhp/Desktop/hackathon/NormalBreast'
# round_tag = 'R2'
# tforms = load_elx_pickle_tforms(
#     '/home/nhp/Desktop/hackathon/repo/registration-challenge/normalbreast.pkl')
# round_tforms = [
#     tforms['NormalBreast_RR2']['rigid'], tforms['NormalBreast_RR2']['nl']
# ]
# source_res = 0.2645833333


def apply_transform_to_round(im_dir,
                             source_res,
                             round_tforms,
                             out_folder,
                             round_tag='R0'):

    dir_path = Path(im_dir)

    round_ims = sorted(dir_path.glob('{}_*'.format(round_tag)))
    round_ims = [str(im) for im in round_ims]

    for round_im in round_ims:
        im = transform_2D_image(
            str(round_im),
            source_res,
            round_tforms,
            im_output_fp='',
            write_image=False)

        out_im_path = Path(round_im).stem + '_registered.tiff'

        output = str(Path(out_folder) / out_im_path)

        sitk.WriteImage(im, output, True)


#preprocessed
datapath = Path('/home/nhp/Desktop/hackathon/NormalBreast')

#preprocessed
outpath = Path('/home/nhp/Desktop/hackathon/masked_reg')

# grab all channel 2 images
#raw
ims = sorted(datapath.glob("*_c2_*"))

#preprocesseds
ims = sorted(datapath.glob("*"))

# select the first index image as the target
target = str(ims[0])

#reg models
reg_models = [
    'registration/elx_reg_params/rigid_largersteps.txt',
    'registration/elx_reg_params/nl.txt'
]

for idx, image in enumerate(ims):
    out_folder = outpath / image.stem
    if out_folder.is_dir() is False:
        out_folder.mkdir()
    if idx > 0:
        register_2D_images(
            str(image),
            0.2645833333,
            str(target),
            0.2645833333,
            reg_models,
            str(out_folder),
        )

        rig_tform = sorted(Path(out_folder).glob('*.0.txt'))
        nl_tform = sorted(Path(out_folder).glob('*.1.txt'))
        tform_list = [str(rig_tform[0]), str(nl_tform[0])]
        im = transform_2D_image(
            str(image),
            0.2645833333,
            tform_list,
            im_output_fp='',
            write_image=False)
        out_im_path = image.stem + '_registered.tiff'
        str(out_folder / out_im_path)
        sitk.WriteImage(im, str(out_folder / out_im_path), True)
