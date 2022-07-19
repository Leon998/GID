import argparse
import time
from pathlib import Path
import os
import shutil
import sys

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.myutils import *
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(save_img=False):
    weights, view_img, save_txt, imgsz, trace = opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    object, ground_truth, step, dist, thres, project = opt.object, opt.object, opt.step, opt.dist, opt.thres, opt.project
    classes = [32, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 64, 65, 67, 74, 76]
    name = object
    ground_truth = object
    # prepare the sources
    filePath = '/home/shixu/My_env/Dataset/object/' + object
    name_list = os.listdir(filePath)
    name_list.sort()
    source_list = []
    for i in name_list:
        i = filePath + '/' + i
        source_list.append(i)
    # Directories
    save_dir = Path(project) / name  # evals/object
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # ============================================= initialize evaluation results===================================== #
    instance = 0
    seq_length = 75
    sum_seq = [0] * seq_length  # saving to SP.txt
    sum_seq_acc = []  # saving to SP_acc.txt
    sum_inst = []  # saving to IP.txt
    sum_grasp = []  # saving to GP.txt
    sum_NPC = []  # saving to NPC.txt

    # ================================================== Hyper-parameters ============================================ #
    step = step  # 累积投票的时候，往前看几步
    if dist:
        Box_thres = dist2thres(dist)  # Thres differ from each class
        print('Using distance thresholds, dist=', dist)
    if thres:
        Box_thres = [thres for idx in range(80)]  # All class thres are the same
        print('Using regular thresholds, thres=', thres)
    
    for source in source_list:
        not_trigger = 1
        # For every video, run the process
        vid_name = get_vid_name(source)
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        # ===============================initialize evaluation results for each time/source=========================== #
        # Sequence folder
        eval_seq = []
        seq_dir = save_dir / 'seq'
        seq_dir.mkdir(parents=True, exist_ok=True)
        seq_path = str(seq_dir / str('eval_seq' + vid_name + '.txt'))
        (seq_dir / 'acc').mkdir(parents=True, exist_ok=True)
        seq_acc_path = str(seq_dir / 'acc' / str('eval_seq_acc' + vid_name + '.txt'))
        # Instance folder
        eval_inst = 0
        inst_dir = save_dir / 'inst'
        inst_dir.mkdir(parents=True, exist_ok=True)
        inst_path = str(inst_dir / str('eval_inst' + vid_name + '.txt'))
        # Grasp folder
        eval_grasp = []
        grasp_dir = save_dir / 'grasp'
        grasp_dir.mkdir(parents=True, exist_ok=True)
        grasp_path = str(grasp_dir / str('eval_grasp' + vid_name + '.txt'))
        # NPC folder
        NPC = 0
        last_pred = 'None'
        npc_dir = save_dir / 'npc'
        npc_dir.mkdir(parents=True, exist_ok=True)
        npc_path = str(npc_dir / str('npc' + vid_name + '.txt'))
        # =============================== information logs ========================================== #
        # stream_log：记录视频流每一帧累积信息的
        # class_score_lod: 80×n维的列表，表示80个类别的得分记录
        stream_log = []
        class_score_log = np.zeros((80, 1))
        new_frame = np.zeros(80)
        frame_idx = 0
        trigger_flag = [False, "None"]
        for path, im, im0s, vid_cap in dataset:
            # 分数记录
            if frame_idx >= 1:
                class_score_log = np.column_stack((class_score_log, new_frame))
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if im.ndimension() == 3:
                im = im.unsqueeze(0)
    
            # Inference
            t1 = time_synchronized()
            pred = model(im, augment=opt.augment)[0]
    
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()
    
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, im, im0s)
            # ===================================至此，推理过程已经结束================================= #
            # 记录每张图片所有目标结果的列表
            frame_log = []
            # 记录图片里每个目标得分的列表
            score_list = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
    
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                im1 = im0
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
    
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化 转为list再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        coffset = get_centeroffset(xyxy, gn, normalize=True)  # 获得每个目标的中心偏移量coffset
                        # coffset = get_centeroffset_2version(xywh, normalize=True)
                        thres = Box_thres[int(cls)]
                        box_rate = get_box_thres_rate(xywh, thres)  # 获取阈值比
                        box_size = get_box_size((xywh))  # 只获得框大小
                        score = count_score(box_rate, coffset)  # 计分score
                        # 记录当前这个种类的特征
                        frame_log.append(
                            {"cls": names[int(cls)], "cls_num": int(cls), "conf": conf, "xyxy": xyxy, "xywh": xywh,
                             "coffset": coffset,
                             "box_rate": box_rate, "box_size": box_size, "score": score})
                        score_list.append(score)  # score_list每帧都更新
                        # 每次直接对应int(cls)的那个class_score_log进行append操作
                        if score >= class_score_log[int(cls), :][frame_idx]:
                            class_score_log[int(cls), :][frame_idx] = score

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)


                    # =====================================单个object检测结束================================= #
                    # ==================================TargetChoosing===================================== #
                    # Not voting
                    # target_idx = score_list.index(max(score_list))  # 这一步可以修改成voting之类的方式
                    # Voting
                    if frame_idx < step:
                        target_idx = score_list.index(max(score_list))  # 注意这里是因为之前保持了score_list和frame_log的目标索引是一样的
                    else:
                        target_idx = vote_score(frame_log, class_score_log, step=step)  # 连续step帧累积投票

                        # 归一法计算概率
                    prob_list = norm_prob(score_list)
                    prob = prob_list[target_idx]
                    target = frame_log[target_idx]
                    # 这里是判断是否预测对了target
                    eval_seq = save_eval_seq(eval_seq, target["cls"], ground_truth, prob)
                    if last_pred != target["cls"]:  # Checking number of prediction changes
                        NPC += 1
                    last_pred = target["cls"]
                    target_xyxy = target["xyxy"]
                    im1 = info_on_img(im0, gn, zoom=[0.48, 0.7], label="Box_x_loc: " + str(round(target["xywh"][0], 3)))
                    im1 = info_on_img(im1, gn, zoom=[0.48, 0.75], label="Box_y_loc: " + str(round(target["xywh"][1], 3)))
                    im1 = info_on_img(im1, gn, zoom=[0.48, 0.8], label="Box_size: " + str(round(target["box_size"], 3)))
                    im1 = info_on_img(im1, gn, zoom=[0.48, 0.85], label="Box_rate: " + str(round(target["box_rate"], 3)))
                    im1 = info_on_img(im1, gn, zoom=[0.48, 0.9],
                                      label="Score: " + str(round(target["score"].item(), 3)))
                    im1 = plot_target_box(target_xyxy, im1, line_thickness=2)
                    trigger_flag = check_trigger(target["box_rate"], target["xywh"], target["cls"], trigger_flag)
                    if trigger_flag[0]:
                        # 判断是否在grasping
                        im1 = text_on_img(im1, gn, zoom=[0.02, 0.95], label="Grasping " + trigger_flag[1])
                        if not_trigger:
                            eval_inst = save_eval_instance(eval_inst, target["cls"], ground_truth)
                            not_trigger = 0
                    else:
                        im1 = text_on_img(im1, gn, zoom=[0.02, 0.95], label="Targeting: " + target["cls"])
                    stream_log.append(frame_log)

                else:
                    # 如果没有预测出目标
                    eval_seq = save_eval_seq(eval_seq, "None", ground_truth, float(0))
                    trigger_flag = check_trigger_null(trigger_flag)
                    if trigger_flag[0]:
                        im1 = text_on_img(im1, gn, zoom=[0.02, 0.95], label="Grasping " + trigger_flag[1])
                    else:
                        im1 = text_on_img(im1, gn, zoom=[0.02, 0.95], label="No Target")
                    stream_log.append(["None"])

                im1 = text_on_img(im1, gn, zoom=[0.02, 0.1], color=[0, 0, 255], label="Frame " + str(frame_idx))
                # 记录当前帧Trigger_flag的状态
                im1 = text_on_img(im1, gn, zoom=[0.02, 0.2], color=[0, 0, 255],
                                  label="Flag on" if trigger_flag[0] else "Flag off")

                if not not_trigger:
                    eval_grasp = save_eval_grasp(eval_grasp, trigger_flag, ground_truth)
                    eval_grasp = check_gp(eval_grasp)
    
                # Stream results
                if view_img:
                    cv2.imshow(str(p), im1)
                    cv2.waitKey(1)  # 1 millisecond
    
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im1)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im1.shape[1], im1.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im1)
            frame_idx += 1
            # 至此结束当前帧
    
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #print(f"Results saved to {save_dir}{s}")
        # 打印预测的总时间
        print(frame_idx)
        print(f'Done. ({time.time() - t0:.3f}s)')
        # ============================ saving evaluation results for a single source time============================= #
        # 保存seq评估，并保持长度一致
        equal_eval_seq = equal_len(eval_seq, seq_length)
        sum_seq = list_sum(sum_seq, equal_eval_seq)
        save_file_continue(seq_path, equal_eval_seq)
        # Saving accuracy of the sequence
        accuracy = seq_accuracy(equal_eval_seq)
        sum_seq_acc.append(accuracy)
        save_file_discrete(seq_acc_path, accuracy)
        # Saving instance evaluation
        sum_inst.append(eval_inst)
        save_file_discrete(inst_path, eval_inst)
        # Saving grasping evaluation
        if not len(eval_grasp):
            eval_grasp.append(0)
        sum_grasp.append(eval_grasp[-1])
        save_file_continue(grasp_path, eval_grasp)
        # Saving NPC
        save_file_discrete(npc_path, NPC)
        sum_NPC.append(NPC)
        # 把所有class都保存到file
        # save_score_to_file(save_dir, class_score_log)

        instance += 1
    # ============================================ saving evaluation results========================================== #
    # Calculating SP
    mean_seq = list_mean(sum_seq, len(source_list))
    save_file_continue(save_dir / 'SP.txt', mean_seq)
    # Calculating SP accuracy
    mean_acc = sum(sum_seq_acc) / len(source_list)
    sum_seq_acc.append(mean_acc)
    save_file_continue(save_dir / 'SP_acc.txt', sum_seq_acc)
    # Calculating IP mean
    mean_inst = sum(sum_inst) / len(sum_inst)
    sum_inst.append(mean_inst)
    save_file_continue(save_dir / 'IP.txt', sum_inst)
    # Calculating GP mean
    mean_grasp = sum(sum_grasp) / len(sum_grasp)
    sum_grasp.append(mean_grasp)
    save_file_continue(save_dir / 'GP.txt', sum_grasp)
    # Calculating NPC mean
    mean_NPC = sum(sum_NPC) / len(sum_NPC)
    sum_NPC.append(mean_NPC)
    save_file_continue(save_dir / 'NPC.txt', sum_NPC)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov7x.pt', help='model.pt path(s)')
    parser.add_argument('--object', type=str, default='apple', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'evals', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--step', type=int, default=1, help='vote step')
    parser.add_argument('--dist', type=int, help='distance and thres realtionship')
    parser.add_argument('--thres', type=float, help='regular threshold')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
