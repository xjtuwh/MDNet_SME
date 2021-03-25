import sys
import cv2
import os
import numpy as np
import time
import argparse
import yaml, json
from PIL import Image

import matplotlib.pyplot as plt

import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append('/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh2020/pyMDNet/')

from modules.model import MDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from modules.utils import overlap_ratio
from data_prov import RegionExtractor
from bbreg import BBRegressor
from gen_config import gen_config
from tracking.vot import VOT, Rectangle

opts = yaml.safe_load(open('/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh2020/pyMDNet/tracking/options.yaml','r'))

def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts)
    for i, regions in enumerate(extractor):
        if opts['use_gpu']:
            regions = regions.cuda()
        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), 0)
    return feats

def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for i in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats[pos_cur_idx]
        batch_neg_feats = neg_feats[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()

def rect_to_poly(rect):
    x0 = rect[0]
    y0 = rect[1]
    x1 = rect[0] + rect[2]
    y1 = rect[1]
    x2 = rect[0] + rect[2]
    y2 = rect[1] + rect[3]
    x3 = rect[0]
    y3 = rect[1] + rect[3]
    return [x0, y0, x1, y1, x2, y2, x3, y3]

def parse_sequence_name(image_path):
    idx = image_path.find('/color/')
    return image_path[idx - image_path[:idx][::-1].find('/'):idx], idx

def parse_frame_name(image_path, idx):
    frame_name = image_path[idx + len('/color/'):]
    return frame_name[:frame_name.find('.')]

# MAIN
handle = VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

gt_rect = [selection.x, selection.y, selection.width, selection.height]

np.random.seed(0)
torch.manual_seed(0)

target_bbox = np.array(gt_rect)
model = MDNet(opts['model_path'])
if opts['use_gpu']:
    model = model.cuda()

# Init criterion and optimizer
criterion = BCELoss()
model.set_learnable_params(opts['ft_layers'])
init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])
# Load first image
image = Image.open(imagefile).convert('RGB')
# Draw pos/neg samples
pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
    target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

neg_examples = np.concatenate([
    SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
    SampleGenerator('whole', image.size)(
        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
neg_examples = np.random.permutation(neg_examples)

# Extract pos/neg features
pos_feats = forward_samples(model, image, pos_examples)
neg_feats = forward_samples(model, image, neg_examples)

# Initial training
train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
del init_optimizer, neg_feats
torch.cuda.empty_cache()

# Train bbox regressor
bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'], opts['aspect_bbreg'])(
    target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
bbreg_feats = forward_samples(model, image, bbreg_examples)
bbreg = BBRegressor(image.size)
bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
del bbreg_feats
torch.cuda.empty_cache()

# Init sample generators for update
sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

# Init pos/neg features for update
neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
neg_feats = forward_samples(model, image, neg_examples)
pos_feats_all = [pos_feats]
neg_feats_all = [neg_feats]
i=0
while True:
    imagefile = handle.frame()
    i=i+1
    if not imagefile:
        break

    # Load image
    image = Image.open(imagefile).convert('RGB')

    # Estimate target bbox
    samples = sample_generator(target_bbox, opts['n_samples'])
    sample_scores = forward_samples(model, image, samples, out_layer='fc6')

    top_scores, top_idx = sample_scores[:, 1].topk(5)
    top_idx = top_idx.cpu()
    target_score = top_scores.mean()
    target_bbox = samples[top_idx]
    if top_idx.shape[0] > 1:
        target_bbox = target_bbox.mean(axis=0)
    success = target_score > 0

    # Expand search area at failure
    if success:
        sample_generator.set_trans(opts['trans'])
    else:
        sample_generator.expand_trans(opts['trans_limit'])

    # Bbox regression
    if success:
        bbreg_samples = samples[top_idx]
        if top_idx.shape[0] == 1:
            bbreg_samples = bbreg_samples[None, :]
        bbreg_feats = forward_samples(model, image, bbreg_samples)
        bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
        bbreg_bbox = bbreg_samples.mean(axis=0)
    else:
        bbreg_bbox = target_bbox

    pred_poly = Rectangle(bbreg_bbox[0], bbreg_bbox[1], bbreg_bbox[2], bbreg_bbox[3])
    confidence = target_score.item()
    # Data collect
    if success:
        pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
        pos_feats = forward_samples(model, image, pos_examples)
        pos_feats_all.append(pos_feats)
        if len(pos_feats_all) > opts['n_frames_long']:
            del pos_feats_all[0]

        neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
        neg_feats = forward_samples(model, image, neg_examples)
        neg_feats_all.append(neg_feats)
        if len(neg_feats_all) > opts['n_frames_short']:
            del neg_feats_all[0]

    # Short term update
    if not success:
        nframes = min(opts['n_frames_short'], len(pos_feats_all))
        pos_data = torch.cat(pos_feats_all[-nframes:], 0)
        neg_data = torch.cat(neg_feats_all, 0)
        train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

    # Long term update
    elif i % opts['long_interval'] == 0:
        pos_data = torch.cat(pos_feats_all, 0)
        neg_data = torch.cat(neg_feats_all, 0)
        train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

    torch.cuda.empty_cache()
    handle.report(pred_poly, confidence)
