import unittest
from app import app, classify_weather, load_model, weather_classes
import numpy as np

class TestUnit(unittest.TestCase):
	def setUp(self):
		app.testing = True
		self.client = app.test_client()
	
	# Complete this function to test proper handling of missing input field in the input
	def test_post_missing_field(self):
		form_data = {
			'temperature': '270.277',
			'pressure': '1006',
			'humidity': '84',
			# 'wind_speed' is missing
			'wind_deg': '274',
			'rain_1h': '0',
			'rain_3h': '0',
			'snow': '0',
			'clouds': '9'
		}
		response = self.client.post('/', data=form_data)
		# Tests that missing input returns HTTP 400
		self.assertEqual(response.status_code, 400, 'Missing input field should return HTTP 400, check app.py > home() > except block')
		# Tests that error message is displayed with missing input field
		self.assertIn(b'Error processing input', response.data, 'Error message not displayed with missing input field, check app.py > home() > try/except block')

		# Tests that result page is not rendered with missing input field
		self.assertNotIn(b'Prediction Result', response.data, 'Result page was rendered with missing input field, check app.py > home() > try/except block')

	# Complete this function to test that the model can be loaded correctly
	# Tests that the correct model is loaded correctly
	def test_model_can_be_loaded(self):
		model = load_model()

		# tests that a model is assigned to model variable
		self.assertIsNotNone(model, 'load_model() returned None, check that model/model.pkl exists and loads')

		# tests that the model has predict method
		self.assertTrue(hasattr(model, 'predict'), 'Loaded model does not have predict method, check that model/model.pkl is the expected model')

		# tests that model has expected number of input features
		self.assertEqual(model.n_features_in_, 9, f'Loaded model expects {model.n_features_in_} input features but the app sends 9, check that model/model.pkl is the expected model')
		

	# Sanity check on the list the class tests look their labels up in
	def test_weather_classes_count(self):
		self.assertEqual(len(weather_classes), 9, f'weather_classes should have 9 labels, one per model class, but has {len(weather_classes)}')

	# Test model classification is within the 9 classes, one test per class, each with an input chosen to return that class
	
	# tests that the model can return a 'clear' prediction
	def test_clear_classification_output(self):
		test_input = np.array([269.686,1002,78,0,23,0,0,0,0]).reshape(1,-1)
		class_result, _ = classify_weather(test_input) 
		# Look the label up by class index (0 = clear) rather than hard-coding the string
		expected = weather_classes[0]
		self.assertEqual(class_result, expected, f'Clear weather input should be classified as "{expected}" (weather_classes[0]), check that app.py returns the model\'s prediction')

	# tests that the model can return a 'rainy' prediction	
	def test_rainy_classification_output(self):
		test_input = np.array([279.626,998,99,1,314,0.3,0,0,88]).reshape(1,-1)
		class_result, _ = classify_weather(test_input) 
		# Look the label up by class index (6 = rainy) rather than hard-coding the string, so the
		# test still passes if the label is renamed (e.g. 'rainy' -> 'rain')
		expected = weather_classes[6]
		self.assertEqual(class_result, expected, f'Rainy weather input should be classified as "{expected}" (weather_classes[6]), check that app.py returns the model\'s prediction')

	# tests that the model can return a 'foggy' prediction	
	def test_foggy_classification_output(self):
		test_input = np.array([242.32, 1048, 100, 4, 65, 0, 0, 0, 77]).reshape(1,-1)
		class_result, _ = classify_weather(test_input) 
		# Look the label up by class index (3 = foggy) rather than hard-coding the string
		expected = weather_classes[3]
		self.assertEqual(class_result, expected, f'Foggy weather input should be classified as "{expected}" (weather_classes[3]), check that app.py returns the model\'s prediction')

	# tests that the model can return a 'cloudy' prediction
	def test_cloudy_classification_output(self):
		test_input = np.array((301,1037,66,9,303,0,9.4,0,77)).reshape(1,-1)
		class_result, _ = classify_weather(test_input)
		# Look the label up by class index (1 = cloudy) rather than hard-coding the string
		expected = weather_classes[1]
		self.assertEqual(class_result, expected, f'Cloudy weather input should be classified as "{expected}" (weather_classes[1]), check that app.py returns the model\'s prediction')	

	# tests that the model can return a 'thunderstorm' prediction
	def test_thunderstorm_classification_output(self):
		test_input = np.array((257,953,56,22,257,12.6,37,0,100)).reshape(1,-1)
		class_result, _ = classify_weather(test_input)
		# Look the label up by class index (8 = thunderstorm) rather than hard-coding the string
		expected = weather_classes[8]
		self.assertEqual(class_result, expected, f'Thunderstorm weather input should be classified as "{expected}" (weather_classes[8]), check that app.py returns the model\'s prediction')

	# tests that the model can return a 'misty' prediction
	def test_misty_classification_output(self):
		# extreme values that are highly unlikely
		test_input = np.array((242,979,99,3,294,0,0,0,18)).reshape(1,-1)
		class_result, _ = classify_weather(test_input)
		# Look the label up by class index (5 = misty) rather than hard-coding the string
		expected = weather_classes[5]
		self.assertEqual(class_result, expected, f'Misty weather input should be classified as "{expected}" (weather_classes[5]), check that app.py returns the model\'s prediction')

	# tests that the model can return a 'drizzly' prediction
	def test_drizzly_classification_output(self):
		# extreme values that are highly unlikely
		test_input = np.array((315,980,100,0,360,0.3,0,0,100)).reshape(1,-1)
		class_result, _ = classify_weather(test_input)
		# Look the label up by class index (2 = drizzly) rather than hard-coding the string
		expected = weather_classes[2]
		self.assertEqual(class_result, expected, f'Drizzly weather input should be classified as "{expected}" (weather_classes[2]), check that app.py returns the model\'s prediction')	

	# tests that the model can return a 'smokey' prediction
	def test_smokey_classification_output(self):
		# extreme values that are highly unlikely
		test_input = np.array((320,900,0,0,360,0,0,4,3)).reshape(1,-1)
		class_result, _ = classify_weather(test_input)
		# Look the label up by class index (7 = smokey) rather than hard-coding the string
		expected = weather_classes[7]
		self.assertEqual(class_result, expected, f'Smokey weather input should be classified as "{expected}" (weather_classes[7]), check that app.py returns the model\'s prediction')

	# tests that the model can return a 'hazey' prediction	
	def test_hazey_classification_output(self):
		# extreme values that are impossible
		test_input = np.array((1000,0,7,400,180,0,180,480,100)).reshape(1,-1)
		class_result, _ = classify_weather(test_input)
		# Look the label up by class index (4 = hazey) rather than hard-coding the string
		expected = weather_classes[4]
		self.assertEqual(class_result, expected, f'Hazey weather input should be classified as "{expected}" (weather_classes[4]), check that app.py returns the model\'s prediction')

if __name__ == '__main__':
	unittest.main()
