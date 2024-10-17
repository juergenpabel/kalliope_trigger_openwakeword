#!/usr/bin/env python3

from enum import StrEnum as enum_StrEnum
from os.path import isfile as os_path_isfile
from logging import basicConfig as logging_basicConfig, \
                    getLogger as logging_getLogger
from time import sleep as time_sleep
from threading import Thread as threading_Thread
from pyaudio import PyAudio as pyaudio_PyAudio, \
                    paInt16 as pyaudio_paInt16
from np import frombuffer as np_frombuffer, \
               int16 as np_int16

from openwakeword.model import Model as openwakeword_model_Model

from kalliope.core.NeuronModule import MissingParameterException


class STATUS(enum_StrEnum):
	RUNNING = 'running'
	PAUSED  = 'paused'


PARAMETERS = { 'model_filename': None,
               'model_sensitivity': 0.75,
               'inference_framework': None,
               'chunk_size': 2560
}


class Openwakeword(threading_Thread):


	def __init__(self, **kwargs):
		super().__init__()
		self.callback = kwargs.get('callback', None)
		if self.callback is None:
			raise MissingParameterException("[trigger:openwakeword] Trigger callback method is missing (keyword argument 'callback')")
		self.config = {}
		self.config['input_device_index'] = kwargs.get('input_device_index', None)
		for key, default_value in PARAMETERS.items():
			self.config[key] = kwargs.get(key, default_value)
		if self.config['model_filename'] is None:
			raise MissingParameterException("[trigger:openwakeword] 'model_filename' must be configured")
		if os_path_isfile(self.config['model_filename']) is False:
			raise MissingParameterException(f"[trigger:openwakeword] 'model_filename' points to non-existing filename ({self.config['model_filename']})")
		if self.config['inference_framework'] is None:
			if self.config['model_filename'].endswith('.tflite'):
				self.config['inference_framework'] = 'tflite'
			elif self.config['model_filename'].endswith('.onnx'):
				self.config['inference_framework'] = 'onnx'
			else:
				raise MissingParameterException("[trigger:openwakeword] 'inference_framework' can't be derived, must be explicitly configured")
		logging_basicConfig()
		self.logger = logging_getLogger("kalliope")
		self.logger.debug(f"[trigger:openwakeword] configuration: model_filename={self.config['model_filename']}, "
		                  f"model_sensitivity={self.config['model_sensitivity']}, chunk_size={self.config['chunk_size']}")
		self.openwakeword = openwakeword_model_Model(wakeword_models=[self.config['model_filename']], inference_framework=self.config['inference_framework'])
		self.status = STATUS.PAUSED
		self.audio = None


	def run(self):
		self.logger.debug("[trigger:openwakeword] run()")
		while True:
			try:
				while True:
					match self.status:
						case STATUS.RUNNING:
							if self.audio is None:
								self.logger.debug(f"[trigger:openwakeword] opening PyAudio stream as trigger is now unpaused")
								self.audio = pyaudio_PyAudio().open(rate=16000, channels=1, format=pyaudio_paInt16, input=True,
								                                    input_device_index=self.config['input_device_index'],
								                                    frames_per_buffer=self.config['chunk_size'])
								self.openwakeword.reset()
								time_sleep(0.01)
							if self.audio.is_active() is True:
								buffer = self.audio.read(self.config['chunk_size'])
								samples = np_frombuffer(buffer, dtype=np_int16)
								predictions = self.openwakeword.predict(samples)
								for model, score in predictions.items():
									if score >= self.config['model_sensitivity']:
										self.logger.info(f"[trigger:openwakeword] detected keyword from model "
										                 f"'{model}' (score={score:.2f})")
										self.callback()
						case STATUS.PAUSED:
							if self.audio is not None:
								self.logger.debug(f"[trigger:openwakeword] closing PyAudio stream as trigger is now paused")
								self.audio.close()
								self.audio = None
							time_sleep(0.01)
			except OSError:
				self.logger.warn(f"[trigger:openwakeword] caught 'OSError' exception, restarting trigger...")
				if self.audio is not None:
					self.audio.close()
					self.audio = None
				time_sleep(0.1)


	def pause(self):
		if self.status == STATUS.RUNNING:
			self.logger.info("[trigger:openwakeword] pause()")
			self.status = STATUS.PAUSED


	def unpause(self):
		if self.status == STATUS.PAUSED:
			self.logger.info("[trigger:openwakeword] unpause()")
			self.status = STATUS.RUNNING

