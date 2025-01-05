import torch
import os
from typing import Optional
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from unimatch.unimatch import UniMatch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

dependencies = ["torch", "numpy"] 

def _load_state_dict(local_file_path: Optional[str] = None):
    if local_file_path is not None and os.path.exists(local_file_path):
        # Load state_dict from local file
        state_dict = torch.load(local_file_path, map_location=torch.device("cpu"))
    else:
        # Load state_dict from the default URL
        file_name = "gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth"
        url = f"https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, file_name=file_name, map_location=torch.device("cpu"))

    return state_dict['model']

class Predictor:
    def __init__(self, model, task) -> None:
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.task = task
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def infer_cv2(self, image1, image2):
        import cv2
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        return self.infer_pil(image1, image2)
    
    def infer_pil(self, image1, image2):
        padding_factor = 32
        attn_type = 'swin' if self.task == 'flow' else 'self_swin2d_cross_swin1d'
        attn_splits_list = [2, 8]
        corr_radius_list = [-1, 4]
        prop_radius_list = [-1, 1]
        num_reg_refine = 6 if self.task == 'flow' else 3

        # smaller inference size for faster speed
        max_inference_size = [384, 768] if self.task == 'flow' else [640, 960]

        transpose_img = False

        image1 = np.array(image1).astype(np.float32)
        image2 = np.array(image2).astype(np.float32)

        if len(image1.shape) == 2:  # gray image
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        if self.task == 'flow':
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        else:
            val_transform_list = [transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

            val_transform = transforms.Compose(val_transform_list)

            image1 = val_transform(image1).unsqueeze(0).to(self.device)  # [1, 3, H, W]
            image2 = val_transform(image2).unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # the model is trained with size: width > height
        if self.task == 'flow' and image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True

        nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

        inference_size = [min(max_inference_size[0], nearest_size[0]), min(max_inference_size[1], nearest_size[1])]

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)

        results_dict = self.model(image1, image2,
                            attn_type=attn_type,
                            attn_splits_list=attn_splits_list,
                            corr_radius_list=corr_radius_list,
                            prop_radius_list=prop_radius_list,
                            num_reg_refine=num_reg_refine,
                            task=self.task,
                            )

        flow_pr = results_dict['flow_preds'][-1]  # [1, 2, H, W] or [1, H, W]

        # resize back
        if self.task == 'flow':
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                        align_corners=True)
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]
        else:
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                pred_disp = F.interpolate(flow_pr.unsqueeze(1), size=ori_size,
                                        mode='bilinear',
                                        align_corners=True).squeeze(1)  # [1, H, W]
                pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        if self.task == 'flow':
            if transpose_img:
                flow_pr = torch.transpose(flow_pr, -2, -1)

            output = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        else:
            output = pred_disp[0].cpu().numpy()
        return output

def UniMatchStereo(local_file_path: Optional[str] = None):
    state_dict = _load_state_dict(local_file_path)
    model = UniMatch(feature_channels=128,
                     num_scales=2,
                     upsample_factor=4,
                     ffn_dim_expansion=4,
                     num_transformer_layers=6,
                     reg_refine=True,
                     task='stereo')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return Predictor(model, task='stereo')


def _test_run():
    import argparse
    import torch.nn.functional as F
    import numpy as np

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input image file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output image file")
    parser.add_argument("--remote", action="store_true", help="use remote repo")
    parser.add_argument("--reload", action="store_true", help="reload remote repo")
    parser.add_argument("--pil", action="store_true", help="use PIL instead of OpenCV")
    args = parser.parse_args()


    predictor = torch.hub.load(".", "UniMatchStereo", source="local", trust_repo=True)
        
    import PIL
    import torchvision.transforms.functional as TF
    
    image1 = PIL.Image.open(os.path.join(args.input, 'im0.png')).convert("RGB")
    image2 = PIL.Image.open(os.path.join(args.input, 'im1.png')).convert("RGB")
    with torch.inference_mode():
        disp = predictor.infer_pil(image1, image2) # (H, W, 3)
    
if __name__ == "__main__":
    _test_run()