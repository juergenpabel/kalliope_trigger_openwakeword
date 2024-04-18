#!/usr/bin/python3

import logging
import time
from os.path import basename, expanduser
from threading import Thread
from kalliope import Utils
from kalliope.core.NeuronModule import MissingParameterException
from openwakeword.model import Model
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
		for key, default in {'model_path': None, 'model_paths': None, 'inference_framework': 'tflite', 'chunk_size': 1280}.items():
			self.config[key] = kwargs.get(key, default)
		if self.config['model_paths'] is None:
			if self.config['model_path'] is None:
				raise MissingParameterException("model_path or model_paths must be configured")
			self.config['model_paths'] = [self.config['model_path']]
		self.config['model_paths'] = [model_path.strip() for model_path in self.config['model_paths'].split(',')]
		self.config['model_paths'] = [Utils.get_real_file_path(model_path) for model_path in self.config['model_paths']]
		self.openwakeword = None
		self.audio_stream = None


	def run(self):
		logger.debug("[trigger:openwakeword] run()")
		self.openwakeword = Model(wakeword_models=self.config['model_paths'], inference_framework=self.config['inference_framework'])
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
		if self.audio_stream is not None:
			self.audio_stream.close()
		self.audio_stream = pyaudio.PyAudio().open(rate=16000, channels=1, format=pyaudio.paInt16, input=True,
		                                      frames_per_buffer=self.config['chunk_size'],
		                                      input_device_index=self.input_device_index)

