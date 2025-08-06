import torch
import warnings
import os
import rasterio as rio
import numpy as np
import shapely
import geopandas as gpd
import argparse
from utils import datasets
from utils.torch_utils import select_device
from utils.general import check_file, check_img_size, check_imshow, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from models.experimental import attempt_load
from utils.torch_utils import torch_distributed_zero_first
from tqdm import tqdm
import pandas as pd
from torchvision.ops import nms

def boxes_to_shapefile(df, root_dir, projected=True, flip_y_axis=False):
    """
    Convert from image coordinates to geographic coordinates
    Note that this assumes df is just a single plot being passed to this function
    Args:
       df: a pandas type dataframe with columns: name, xmin, ymin, xmax, ymax. Name is the relative path to the root_dir arg.
       root_dir: directory of images to lookup image_path column
       projected: If True, convert from image to geographic coordinates, if False, keep in image coordinate system
       flip_y_axis: If True, reflect predictions over y axis to align with raster data in QGIS, which uses a negative y origin compared to numpy. See https://gis.stackexchange.com/questions/306684/why-does-qgis-use-negative-y-spacing-in-the-default-raster-geotransform
    Returns:
       df: a geospatial dataframe with the boxes optionally transformed to the target crs
    """
    # Raise a warning and confirm if a user sets projected to True when flip_y_axis is True.
    if flip_y_axis and projected:
        warnings.warn(
            "flip_y_axis is {}, and projected is {}. In most cases, projected should be False when inverting y axis. Setting projected=False"
            .format(flip_y_axis, projected), UserWarning)
        projected = False

    plot_names = df.image_path.unique()
    if len(plot_names) > 1:
        raise ValueError("This function projects a single plots worth of data. "
                         "Multiple plot names found {}".format(plot_names))
    else:
        plot_name = plot_names[0]

    rgb_path = "{}/{}".format(root_dir, plot_name)
    with rio.open(rgb_path) as dataset:
        bounds = dataset.bounds
        pixelSizeX, pixelSizeY = dataset.res
        crs = dataset.crs
        transform = dataset.transform

    if projected:
        # Convert image pixel locations to geographic coordinates
        xmin_coords, ymin_coords = rio.transform.xy(transform=transform,
                                                         rows=df.ymin,
                                                         cols=df.xmin,
                                                         offset='center')

        xmax_coords, ymax_coords = rio.transform.xy(transform=transform,
                                                         rows=df.ymax,
                                                         cols=df.xmax,
                                                         offset='center')

        # One box polygon for each tree bounding box
        # Careful of single row edge case where
        # xmin_coords comes out not as a list, but as a float
        if type(xmin_coords) == float:
            xmin_coords = [xmin_coords]
            ymin_coords = [ymin_coords]
            xmax_coords = [xmax_coords]
            ymax_coords = [ymax_coords]

        box_coords = zip(xmin_coords, ymin_coords, xmax_coords, ymax_coords)
        box_geoms = [
            shapely.geometry.box(xmin, ymin, xmax, ymax)
            for xmin, ymin, xmax, ymax in box_coords
        ]

        geodf = gpd.GeoDataFrame(df, geometry=box_geoms)
        geodf.crs = crs

        return geodf

    else:
        if flip_y_axis:
            # See https://gis.stackexchange.com/questions/306684/why-does-qgis-use-negative-y-spacing-in-the-default-raster-geotransform
            # Numpy uses top left 0,0 origin, flip along y axis.
            df['geometry'] = df.apply(
                lambda x: shapely.geometry.box(x.xmin, -x.ymin, x.xmax, -x.ymax), axis=1)
        else:
            df['geometry'] = df.apply(
                lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
        df = gpd.GeoDataFrame(df, geometry="geometry")

        return df

def mosiac(boxes, windows, sigma=0.5, thresh=0.001, iou_threshold=0.1):
    # transform the coordinates to original system
    for index, _ in enumerate(boxes):
        xmin, ymin, xmax, ymax = windows[index].getRect()
        boxes[index].xmin += xmin
        boxes[index].xmax += xmin
        boxes[index].ymin += ymin
        boxes[index].ymax += ymin

    predicted_boxes = pd.concat(boxes)
    print(
        f"{predicted_boxes.shape[0]} predictions in overlapping windows, applying non-max supression"
    )
    # move prediciton to tensor
    boxes = torch.tensor(predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
                         dtype=torch.float32)
    scores = torch.tensor(predicted_boxes.score.values, dtype=torch.float32)
    labels = predicted_boxes.label.values
    # Performs non-maximum suppression (NMS) on the boxes according to
    # their intersection-over-union (IoU).
    bbox_left_idx = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)

    bbox_left_idx = bbox_left_idx.numpy()
    new_boxes, new_labels, new_scores = boxes[bbox_left_idx].type(
        torch.int), labels[bbox_left_idx], scores[bbox_left_idx]

    # Recreate box dataframe
    image_detections = np.concatenate([
        new_boxes,
        np.expand_dims(new_labels, axis=1),
        np.expand_dims(new_scores, axis=1)
    ],
                                      axis=1)

    mosaic_df = pd.DataFrame(image_detections,
                             columns=["xmin", "ymin", "xmax", "ymax", "label", "score"])

    print(f"{mosaic_df.shape[0]} predictions kept after non-max suppression")

    return mosaic_df

def predict_tile(model=None,
                 dsm_path=None,
                 rgb_path=None,
                 image=None,
                 patch_size=400,
                 patch_overlap=0.05,
                 iou_threshold=0.15,
                 conf_thres=0.1,
                 device=None,
                 batch_size=1,
                 half=False,
                 label_dict: dict = {"Tree": 0},
                 return_plot=False,
                 mosaic=True,
                 sigma=0.5,
                 thresh=0.001,
                 color=None,
                 thickness=1):
    """For images too large to input into the model, predict_tile cuts the
    image into overlapping windows, predicts trees on each window and
    reassambles into a single array.

    Args:
        raster_path: Path to image on disk
        image (array): Numpy image array in BGR channel order
            following openCV convention
        patch_size: patch size for each window.
        patch_overlap: patch overlap among windows.
        iou_threshold: Minimum iou overlap among predictions between
            windows to be suppressed.
            Lower values suppress more boxes at edges.
        return_plot: Should the image be returned with the predictions drawn?
        mosaic: Return a single prediction dataframe (True) or a tuple of image crops and predictions (False)
        sigma: variance of Gaussian function used in Gaussian Soft NMS
        thresh: the score thresh used to filter bboxes after soft-nms performed
        color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
        thickness: thickness of the rectangle border line in px

    Returns:
        boxes (array): if return_plot, an image.
        Otherwise a numpy array of predicted bounding boxes, scores and labels
    """
    numeric_to_label_dict = {v: k for k, v in label_dict.items()}
    model.eval()
    # model.nms_thresh = self.config["nms_thresh"]

    # # if more than one GPU present, use only a the first available gpu
    # if torch.cuda.device_count() > 1:
    #     # Get available gpus and regenerate trainer
    #     warnings.warn(
    #         "More than one GPU detected. Using only the first GPU for predict_tile.")
    #     self.config["devices"] = 1
    #     self.create_trainer()

    if (rgb_path and rgb_path is None) and (image is None):
        raise ValueError(
            "Both tile and tile_path are None. Either supply a path to a tile on disk, or read one into memory!"
        )

    if dsm_path and rgb_path is None:
        image = image
    else:
        dsm_raster = rio.open(dsm_path).read()
        rgb_raster = rio.open(rgb_path).read()
        image = np.concatenate((rgb_raster, dsm_raster), axis=0)
        # self.image = rio.open(raster_path).read()
        image = np.moveaxis(image, 0, 2)

    ds = datasets.TileDataset(tile=image,
                             patch_overlap=patch_overlap,
                             patch_size=patch_size)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    # Flatten list from batched prediction
    results = []
    for i, image in enumerate(tqdm(dataloader, desc='Predicting tiles')):
        img = image[:, 0:3, ...].to(device)
        img2 = image[:, 3:6, ...].to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img2 = img2.half() if half else img2.float()  # uint8 to fp16/32
        # img2 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img2.ndimension() == 3:
            img2 = img2.unsqueeze(0)

        # Inference
        # t1 = time_synchronized()
        pred = model(img, img2, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_threshold, classes=None, agnostic=False)

        for batch in pred:
            batch_pd = pd.DataFrame(batch.cpu().numpy(), columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
            results.append(batch_pd)

    if mosaic:
        results = mosiac(results,
                         ds.windows,
                         sigma=sigma,
                         thresh=thresh,
                         iou_threshold=iou_threshold)
        results["label"] = results.label.apply(
            lambda x: numeric_to_label_dict[x])
        if rgb_path:
            results["image_path"] = os.path.basename(rgb_path)

    return results


def main(args):
    batch_size = args.batch_size
    patch_size = args.patch_size
    patch_overlap = args.patch_overlap
    iou_threshold = args.iou_threshold
    conf_thres = args.conf_thres
    return_plot = args.return_plot
    sigma = args.sigma
    thresh = args.thresh
    color = args.color
    thickness = args.thickness
    device = select_device(args.device, batch_size=batch_size)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(args.model_path, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    patch_size = check_img_size(patch_size, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    dsm_path = args.dsm_path
    rgb_path = args.rgb_path
    boxes = predict_tile(model=model, dsm_path=dsm_path, rgb_path=rgb_path, patch_size=patch_size,
                       patch_overlap=patch_overlap, iou_threshold=iou_threshold, return_plot=return_plot,
                       sigma=sigma, thresh=thresh, color=color, thickness=thickness, conf_thres=conf_thres,
                       batch_size=batch_size, device=device, half=half, label_dict={"Tree": 0})

    rio_src = rio.open(rgb_path)
    image = rio_src.read()
    PATH_TO_DIR = "/home/wdblink/文档/20240521_hulushan/DOM"
    # Create a shapefile, in this case img data was unprojected
    shp = boxes_to_shapefile(boxes, root_dir=PATH_TO_DIR, projected=True)
    # Get name of image and save a .shp in the same folder
    basename = os.path.splitext(os.path.basename(rgb_path))[0]
    shp.to_file("{}/{}.shp".format(PATH_TO_DIR, basename))
    # #dataloader:read RGB and DSM raster from raster_path, and convert to dataset
    # dataloader = datasets.TileDataset(tile_path=raster_path,
    #                                    patch_overlap=patch_overlap,
    #                                    patch_size=patch_size)



if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser(description='Predict trees in a tile')
    argparser.add_argument('--model_path', type=str, default='/home/wdblink/github/multispectral-object-detection/runs/train/exp28/weights/best.pt', help='Path to trained model')
    argparser.add_argument('--dsm_path', type=str, default='/home/wdblink/文档/20240521_hulushan/DSM/3channel/Production_orth_DSM_merge.tif', help='Path to raster tile')
    argparser.add_argument('--rgb_path', type=str, default='/home/wdblink/文档/20240521_hulushan/DOM/Production_orth_ortho_merge.tif', help='Path to raster tile')
    argparser.add_argument('--data', type=str, default='./data/multispectral/tree_dsm.yaml', help='*.data path')
    argparser.add_argument('--device', type=str, default='0', help='Device to use for inference (cpu, cuda, cuda:0, etc.)')
    argparser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    argparser.add_argument('--patch_size', type=int, default=640, help='Patch size for each window')
    argparser.add_argument('--patch_overlap', type=float, default=0.5, help='Patch overlap among windows')
    argparser.add_argument('--iou_threshold', type=float, default=0.35, help='Minimum iou overlap among predictions between windows to be suppressed')
    argparser.add_argument('--conf_thres', type=float, default=0.5, help='Confidence threshold for predictions')
    argparser.add_argument('--return_plot', action='store_true', help='Should the image be returned with the predictions drawn?')
    argparser.add_argument('--sigma', type=float, default=0.5, help='Variance of Gaussian function used in Gaussian Soft NMS')
    argparser.add_argument('--thresh', type=float, default=0.001, help='The score thresh used to filter bboxes after soft-nms performed')
    argparser.add_argument('--color', type=tuple, default=(0, 165, 255), help='Color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)')
    argparser.add_argument('--thickness', type=int, default=1, help='Thickness of the rectangle border line in px')
    args = argparser.parse_args()
    main(args)