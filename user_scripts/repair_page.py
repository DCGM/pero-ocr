# -*- coding: utf-8 -*-
import shutil
import time
import os
import argparse
import re

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import shapely.geometry
import tkinter as tk

from pero_ocr.document_ocr import layout
from pero_ocr.document_ocr import crop_engine as cropper
from pero_ocr.document_ocr import repair_engine as repair
from pero_ocr.ocr_engine import line_ocr_engine as ocr

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-img', required=True, help='Input image to repair')
    parser.add_argument('-p', '--input-page', required=True, help='Input page xml (should contain detected lines and text)')
    parser.add_argument('-r', '--repair-json', help='Path to repair engine json', default='/mnt/matylda1/hradis/PERO/layout_engines/repair_LN/repair_engine.json')
    parser.add_argument('-o', '--ocr-json', help='Path to OCR engine json', default='/mnt/matylda1/hradis/PERO/ocr_engines/CZECH_LN_2019-11-11/ocr_engine.json')
    return parser.parse_args()

class LayoutClicker(object):
    def __init__(self, page_layout):
        self.down_pos = None
        self.chosen_line = None
        self.layout = page_layout
        self.drawing = False
        self.points = []

        self.double_click = False

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.double_click:
                if not self.drawing and self.chosen_line:
                    self.points = []
                    self.points.append((y, x))
                    self.points.append((y, x))
                    if shapely.geometry.Polygon(self.chosen_line.polygon).contains(shapely.geometry.Point(self.points[-1])):
                        self.drawing = True
                    else:
                        self.points = []
                        self.drawing = False
                else:
                    self.drawing = False
            else:
                self.double_click = False
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and self.points:
            self.points[-1] = (y, x)
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.double_click = True
            self.drawing = False
            self.points = []
            self.down_pos = shapely.geometry.Point(y, x)
            for line in self.layout.lines_iterator():
                poly = shapely.geometry.Polygon(line.polygon)
                if poly.contains(self.down_pos):
                    self.chosen_line = line
                    break

class TextInputRepair(object):
    def __init__(self, transcription):
        self.transcription = transcription
        self.return_flag = None

    def repair_callback(self):
        self.transcription = self.entry_field.get()
        self.return_flag = 'repair'
        self.master.destroy()

    def revert_callback(self):
        self.return_flag = 'revert'
        self.master.destroy()

    def cancel_callback(self):
        self.master.destroy()

    def run(self):
        self.master = tk.Tk()
        self.master.title('Transcription editor')
        tk.Label(self.master, text="Text transcription:").grid(row=0, column=0)
        self.entry_field = tk.Entry(self.master, width=50)
        self.entry_field.insert(tk.END, self.transcription)
        self.entry_field.grid(row=0, column=1)

        tk.Button(self.master,
                  text='Repair',
                  command=self.repair_callback).grid(row=0, column=2)
        tk.Button(self.master,
                  text='Revert',
                  command=self.revert_callback).grid(row=1, column=2)
        tk.Button(self.master,
                  text='Cancel',
                  command=self.cancel_callback).grid(row=2, column=2)

        self.master.mainloop()

        return self.return_flag, self.transcription

class TextInputInpaint(object):
    def __init__(self, transcription_left, transcription_right):
        self.transcription_left = transcription_left
        self.transcription_right = transcription_right
        self.transcription = None
        self.return_flag = None

    def inpaint_callback(self):
        self.transcription = '{} {} {}'.format(self.transcription_left, self.entry_field.get(), self.transcription_right)
        self.return_flag = 'inpaint'
        self.master.destroy()

    def cancel_callback(self):
        self.master.destroy()

    def run(self):
        self.master = tk.Tk()
        self.master.title('Transcription editor')
        tk.Label(self.master, text="Text transcription: {}".format(self.transcription_left)).grid(row=0, column=0)
        self.entry_field = tk.Entry(self.master, width=15)
        self.entry_field.grid(row=0, column=1)
        tk.Label(self.master, text=self.transcription_right).grid(row=0, column=2)

        tk.Button(self.master,
                  text='Inpaint',
                  command=self.inpaint_callback).grid(row=0, column=3)
        tk.Button(self.master,
                  text='Cancel',
                  command=self.cancel_callback).grid(row=1, column=3)

        self.master.mainloop()

        return self.return_flag, self.transcription

def main():

    args = parseargs()

    page_img = cv2.imread(args.input_img)
    page_layout = layout.PageLayout(file=args.input_page)

    page_img_orig = page_img.copy()
    page_img_rendered = page_img.copy()

    print('\nLoading engines...')
    ocr_engine = ocr.EngineLineOCR(args.ocr_json, gpu_id=0)
    repair_engine = repair.EngineRepairCNN(args.repair_json)
    crop_engine = cropper.EngineLineCropper(line_height=repair_engine.height, poly=2, scale=1)

    cv2.namedWindow("Page Editor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Page Editor', 1024, 1024)
    layout_clicker = LayoutClicker(page_layout)
    cv2.setMouseCallback("Page Editor", layout_clicker.callback)

    while True:
        page_img_rendered = page_img.copy()
        if layout_clicker.chosen_line:
            page_img_rendered = layout.draw_lines(page_img_rendered, [layout_clicker.chosen_line.polygon], color=(0,255,0), close=True)
        if layout_clicker.points:
            page_img_rendered = layout.draw_lines(page_img_rendered, [layout_clicker.points], color=(0,0,255))

        cv2.imshow('Page Editor', page_img_rendered)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        elif key == ord('r'):
            text_input = TextInputRepair(layout_clicker.chosen_line.transcription)
            action, new_transcription = text_input.run()
            layout_clicker.chosen_line.transcription = new_transcription

            if action == 'repair':
                line_crop, line_mapping, offset = crop_engine.crop(
                            page_img,
                            layout_clicker.chosen_line.baseline,
                            layout_clicker.chosen_line.heights,
                            return_mapping=True)
                line_crop = repair_engine.repair_line(line_crop, layout_clicker.chosen_line.transcription)
                page_img = crop_engine.blend_in(page_img, line_crop, line_mapping, offset)
                page_img_rendered = page_img.copy()

            elif action == 'revert':
                line_crop, line_mapping, offset = crop_engine.crop(
                            page_img_orig,
                            layout_clicker.chosen_line.baseline,
                            layout_clicker.chosen_line.heights,
                            return_mapping=True)
                page_img = crop_engine.blend_in(page_img, line_crop, line_mapping, offset)
                page_img_rendered = page_img.copy()

        elif key == ord('e') and len(layout_clicker.points)==2:
            line_crop, line_mapping, offset = crop_engine.crop(
                        page_img,
                        layout_clicker.chosen_line.baseline,
                        layout_clicker.chosen_line.heights,
                        return_mapping=True)

            y1 = np.round(line_mapping[line_mapping.shape[0]//2, layout_clicker.points[0][1]-offset[1], 1]).astype(np.uint16)
            y2 = np.round(line_mapping[line_mapping.shape[0]//2, np.clip(layout_clicker.points[1][1]-offset[1], 0, line_mapping.shape[1]-2), 1]).astype(np.uint16)
            if layout_clicker.points[1][1]-offset[1] > line_mapping.shape[1]-10: # dirty fix noisy values at the end of coord map
                y2 = np.amax(line_mapping[:,:,1].astype(np.uint16))
            print('{}/{}'.format(y2, line_crop.shape[1]))
            transcriptions, _ = ocr_engine.process_lines([line_crop[:, :np.minimum(y1, y2), :],
                                                          line_crop[:, np.maximum(y1, y2):, :]])
            line_crop[:, np.minimum(y1, y2):np.maximum(y1, y2), :] = 0
            text_input = TextInputInpaint(transcriptions[0], transcriptions[1])
            action, new_transcription = text_input.run()
            if action == 'inpaint':
                layout_clicker.chosen_line.transcription = new_transcription

                line_crop = repair_engine.inpaint_line(line_crop, layout_clicker.chosen_line.transcription)
                page_img = crop_engine.blend_in(page_img, line_crop, line_mapping, offset)
                line_crop, line_mapping, offset = crop_engine.crop(
                            page_img,
                            layout_clicker.chosen_line.baseline,
                            layout_clicker.chosen_line.heights,
                            return_mapping=True)
                layout_clicker.points = []

if __name__=='__main__':
    main()
