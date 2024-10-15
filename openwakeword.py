#!/usr/bin/env python3

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
	STATUS_RUNNING = 'running'
	STATUS_PAUSED  = 'paused'

	PARAMETERS = { 'model_filename': None,
	               'model_sensitivity': 0.75,
	               'inference_framework': None,
	               'chunk_size': 2560
	             }
	def __init__(self, **kwargs):
		super().__init__()
		self.input_device_index = kwargs.get('input_device_index', None)
		self.callback = kwargs.get('callback', None)
		if self.callback is None:
			raise MissingParameterException("[trigger:openwakeword] Trigger callback method is missing (keyword argument 'callback')")

		self.config = {}
		for key, default_value in Openwakeword.PARAMETERS.items():
			self.config[key] = kwargs.get(key, default_value)
		if self.config['model_filename'] is None:
			raise MissingParameterException("[trigger:openwakeword] 'model_filename' must be configured")
		if os.path.isfile(self.config['model_filename']) is False:
			raise MissingParameterException(f"[trigger:openwakeword] 'model_filename' points to non-existing filename ({self.config['model_filename']})")
		if self.config['inference_framework'] is None:
			if self.config['model_filename'].endswith('.tflite'):
				self.config['inference_framework'] = 'tflite'
			elif self.config['model_filename'].endswith('.onnx'):
				self.config['inference_framework'] = 'onnx'
			else:
				raise MissingParameterException("[trigger:openwakeword] 'inference_framework' can't be derived, must be explicitly configured")
		logger.debug(f"[trigger:openwakeword] configuration: model_filename={self.config['model_filename']}, "
		                                                  f"model_sensitivity={self.config['model_sensitivity']}, "
		                                                  f"chunk_size={self.config['chunk_size']}")
		self.openwakeword = openwakeword.model.Model(wakeword_models=[self.config['model_filename']], inference_framework=self.config['inference_framework'])
		self.audio_stream = None
		self.status = Openwakeword.STATUS_PAUSED


	def run(self):
		logger.debug("[trigger:openwakeword] run()")
		while True:
			try:
				while True:
					audio_stream = self.audio_stream
					if self.status == Openwakeword.STATUS_RUNNING and audio_stream is not None and audio_stream.is_active() is True:
						buffer = audio_stream.read(self.config['chunk_size'])
						audio = np.frombuffer(buffer, dtype=np.int16)
						predictions = self.openwakeword.predict(audio)
						for model, score in predictions.items():
							if score >= self.config['model_sensitivity']:
								logger.info(f"[trigger:openwakeword] keyword from model '{model}' detected (score={score:.2f})")
								self.callback()
					else:
						time.sleep(0.1)
			except OSError:
				logger.warn(f"[trigger:openwakeword] caught 'OSError' exception (probably pyaudio), restarting trigger...")
				self.status = Openwakeword.STATUS_PAUSED
				if self.audio_stream is not None:
					self.audio_stream.close()
					self.audio_stream = None
				if self.status == Openwakeword.STATUS_RUNNING:
					self.unpause()


	def pause(self):
		if self.status == Openwakeword.STATUS_RUNNING:
			logger.info("[trigger:openwakeword] pause()")
			self.status = Openwakeword.STATUS_PAUSED
			if self.audio_stream is not None:
				audio_stream = self.audio_stream
				self.audio_stream = None
				time.sleep(0.01)
				audio_stream.close()


	def unpause(self):
		if self.status == Openwakeword.STATUS_PAUSED:
			logger.info("[trigger:openwakeword] unpause()")
			self.openwakeword.reset()
			while self.audio_stream is None:
				try:
					self.audio_stream = pyaudio.PyAudio().open(rate=16000, channels=1, format=pyaudio.paInt16, input=True,
					                                           frames_per_buffer=self.config['chunk_size'],
					                                           input_device_index=self.input_device_index)
					self.status = Openwakeword.STATUS_RUNNING
				except OSError:
					logger.warn(f"[trigger:openwakeword] caught 'OSError' exception (probably pyaudio) in pyaudio.PyAudio().open(), retrying...")
					time.sleep(0.1)

