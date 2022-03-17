import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        print('-------------- save DIR -----------------------')
        print(directory)
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)

    def visualize_validation_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:10].clone().cpu().data, 10, normalize=True)
        writer.add_image('Val Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:10], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 10, normalize=False, range=(0, 255))
        writer.add_image('Val Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:10], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 10, normalize=False, range=(0, 255))
        writer.add_image('Val Groundtruth label', grid_image, global_step)

    def visualize_test_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image.clone().cpu().data, image.size(0), normalize=True)
        writer.add_image('Test Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output.float(), 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), output.size(0), normalize=False, range=(0, 255))
        writer.add_image('Test Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target, 1).detach().cpu().numpy(),
                                                       dataset=dataset), target.size(0), normalize=False, range=(0, 255))
        writer.add_image('Test Groundtruth label', grid_image, global_step)