from easydict import EasyDict as edict
import PIL.Image as I
from numpy import *
from simplejson import loads
from IPython.display import Image, display_png
import IPython.display as dis

from tf_pose.estimator import TfPoseEstimator  
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import common

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class HumanCropper:
    def __init__(self, model='mobilenet_thin'):
        self.d = edict(camera=2, model=model, resize='432x368', resize_out_ratio=4.0, show_process=False, tensorrt='False') 
        self.e = TfPoseEstimator(
                        get_graph_path(self.d.model),
                        target_size=model_wh(self.d.resize), 
                        trt_bool=str2bool(self.d.tensorrt))
    
    def _rotateIfRequired(self, im):
        self.w, self.h = im.size
        if self.w > self.h:
            im = im.rotate(-90, expand=True)
        self.w, self.h = im.size
        return im
        
        
    def _setImage(self, path):
        self.path = path
        self.img = common.read_imgfile(path, None, None)
        print(self.img.shape)
        self.im = self._rotateIfRequired(I.open(path))
    def _predict(self):
        self.humans = self.e.inference(self.img, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=self.d.resize_out_ratio)
        
    def _draw(self,):
        return self.e.draw_humans(self.img, self.humans, imgcopy=False)
    
    def _get_cropped_image(self, body_parts, offset=100):
            b = body_parts
            center = (int(b.x * self.w + 0.5), int(b.y * self.h + 0.5))
            o = offset
            return self.im.crop((center[0] - o, center[1] - o, center[0] + o, center[1] + o))
        
    def showCroppedImage(self,): 
        for human in self.humans:
            [dis.display(self._get_cropped_image(human.body_parts[i])) for i in range(len(human.body_parts))]
    
    def __call__(self, path):
        self._setImage(path)
        self._predict()
        self.img = self._draw()
        return self.img, self.humans 
    
    
