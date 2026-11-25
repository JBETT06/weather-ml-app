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
		

	# Test model classification is within the 9 classes, one test per class, each with an input chosen to return that class
	
	# tests that the model can return a 'clear' prediction
	def test_clear_classification_output(self):
		test_input = np.array([269.686,1002,78,0,23,0,0,0,0]).reshape(1,-1)
		class_result, _ = classify_weather(test_input) 
		# Ensure that 'clear' class is returned
		self.assertEqual(class_result, 'clear', 'Clear weather input should be classified as "clear", check that app.py returns the model\'s prediction')

	# tests that the model can return a 'rainy' prediction	
	def test_rainy_classification_output(self):
		test_input = np.array([279.626,998,99,1,314,0.3,0,0,88]).reshape(1,-1)
		class_result, _ = classify_weather(test_input) 
		# The model predicts class index 6 for this input. Look the label up by index so the
		# test still passes if the label in weather_classes is renamed (e.g. 'rainy' -> 'rain')
		expected = weather_classes[6]
		self.assertEqual(class_result, expected, f'Rainy weather input should be classified as "{expected}" (weather_classes[6]), check that app.py returns the model\'s prediction')

	# tests that the model can return a 'foggy' prediction	
	def test_foggy_classification_output(self):
		test_input = np.array([242.32, 1048, 100, 4, 65, 0, 0, 0, 77]).reshape(1,-1)
		class_result, _ = classify_weather(test_input) 
		# Ensure that 'foggy' class is returned
		self.assertEqual(class_result, 'foggy', 'Foggy weather input should be classified as "foggy", check that app.py returns the model\'s prediction')

	# tests that the model can return a 'cloudy' prediction
	def test_cloudy_classification_output(self):
		test_input = np.array((301,1037,66,9,303,0,9.4,0,77)).reshape(1,-1)
		class_result, _ = classify_weather(test_input)
		# Ensure the 'cloudy' class is returned
		self.assertEqual(class_result, 'cloudy', 'Cloudy weather input should be classified as "cloudy", check that app.py returns the model\'s prediction')	

	# tests that the model can return a 'thunderstorm' prediction
	def test_thunderstorm_classification_output(self):
		test_input = np.array((257,953,56,22,257,12.6,37,0,100)).reshape(1,-1)
		class_result, _ = classify_weather(test_input)
		# Ensure the 'thunderstorm' class is returned
		self.assertEqual(class_result, 'thunderstorm', 'Thunderstorm weather input should be classified as "thunderstorm", check that app.py returns the model\'s prediction')

	# tests that the model can return a 'misty' prediction
	def test_misty_classification_output(self):
		# extreme values that are highly unlikely
		test_input = np.array((242,979,99,3,294,0,0,0,18)).reshape(1,-1)
		class_result, _ = classify_weather(test_input)
		# Ensure the 'misty' class is returned
		self.assertEqual(class_result, 'misty', 'Misty weather input should be classified as "misty", check that app.py returns the model\'s prediction')

	# tests that the model can return a 'drizzly' prediction
	def test_drizzly_classification_output(self):
		# extreme values that are highly unlikely
		test_input = np.array((315,980,100,0,360,0.3,0,0,100)).reshape(1,-1)
		class_result, _ = classify_weather(test_input)
		# Ensure the 'drizzly' class is returned
		self.assertEqual(class_result, 'drizzly', 'Drizzly weather input should be classified as "drizzly", check that app.py returns the model\'s prediction')	

	# tests that the model can return a 'smokey' prediction
	def test_smokey_classification_output(self):
		# extreme values that are highly unlikely
		test_input = np.array((320,900,0,0,360,0,0,4,3)).reshape(1,-1)
		class_result, _ = classify_weather(test_input)
		# Ensure the 'smokey' class is returned
		self.assertEqual(class_result, 'smokey', 'Smokey weather input should be classified as "smokey", check that app.py returns the model\'s prediction')

	# tests that the model can return a 'hazey' prediction	
	def test_hazey_classification_output(self):
		# extreme values that are impossible
		test_input = np.array((1000,0,7,400,180,0,180,480,100)).reshape(1,-1)
		class_result, _ = classify_weather(test_input)
		# Ensure the 'hazey' class is returned
		self.assertEqual(class_result, 'hazey', 'Hazey weather input should be classified as "hazey", check that app.py returns the model\'s prediction')

if __name__ == '__main__':
	unittest.main()
