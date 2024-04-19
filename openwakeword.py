#!/usr/bin/python3

import os
import logging
import time
from threading import Thread
from kalliope import Utils
from kalliope.core.NeuronModule import MissingParameterException
import openwakeword.utils
import openwakeword.model
import pyaudio
import np


logging.basicConfig()
logger = logging.getLogger("kalliope")


class Openwakeword(Thread):
	def __init__(self, **kwargs):
		super().__init__()
		logger.debug("[trigger:openwakeword] __init__()")
		self.input_device_index = kwargs.get('input_device_index', None)
		self.callback = kwargs.get('callback', None)
		if self.callback is None:
			raise MissingParameterException("Trigger callback method is missing (keyword argument 'callback')")

		self.config = dict()
		for k, v in {'model_filename': None, 'featuremodels_directory': 'resources/data/openwakeword', 'inference_framework': None, 'chunk_size': 1280}.items():
			self.config[k] = kwargs.get(k, v)
		if self.config['model_filename'] is None:
			raise MissingParameterException("'model_filename' must be configured")
		if os.path.isfile(self.config['model_filename']) is False:
			raise MissingParameterException(f"'model_filename' points to non-existing filename ({self.config['model_filename']})")
		if self.config['inference_framework'] is None:
			if self.config['model_filename'].endswith('.tflite'):
				self.config['inference_framework'] = 'tflite'
			elif self.config['model_filename'].endswith('.onnx'):
				self.config['inference_framework'] = 'onnx'
			else:
				raise MissingParameterException("inference_framework must be configured (because model_filename doesn't end with .tflite or .onnx)")
		self.openwakeword = openwakeword.model.Model(wakeword_models=[self.config['model_filename']], inference_framework=self.config['inference_framework'])
		self.audio_stream = None


	def run(self):
		logger.debug("[trigger:openwakeword] run()")
		self.audio_stream = pyaudio.PyAudio().open(rate=16000, channels=1, format=pyaudio.paInt16, input=True,
		                                           frames_per_buffer=self.config['chunk_size'],
		                                           input_device_index=self.input_device_index)
		while True:
			if self.audio_stream is not None:
				buffer = self.audio_stream.read(self.config['chunk_size'])
				audio = np.frombuffer(buffer, dtype=np.int16)
				predictions = self.openwakeword.predict(audio)
				model_detected = None
				for model, score in predictions.items():
					if score > 0.6:
						model_detected = model
				if model_detected is not None:
						logger.info(f"[trigger:openwakeword] keyword from model '{model_detected}' detected")
						self.pause()
						self.callback()
			time.sleep(0.001)


	def pause(self):
		logger.debug("[trigger:openwakeword] pause()")
		if self.audio_stream is not None:
			self.audio_stream.close()
			self.audio_stream = None


	def unpause(self):
		logger.debug("[trigger:openwakeword] unpause()")
		if self.audio_stream is None:
			self.audio_stream = pyaudio.PyAudio().open(rate=16000, channels=1, format=pyaudio.paInt16, input=True,
			                                           frames_per_buffer=self.config['chunk_size'],
			                                           input_device_index=self.input_device_index)

