import unittest
from app import app, classify_weather, load_model
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
		

	# Test model classification is within the 9 classes, each time for a different class with three different inputs
	def test_clear_classification_output(self):
		test_input = np.array([269.686,1002,78,0,23,0,0,0,0]).reshape(1,-1)
		class_result, _ = classify_weather(test_input) 
		# Ensure that 'clear' class is returned
		self.assertEqual(class_result, 'clear', 'Clear weather input should be classified as "clear", check that app.py returns the model\'s prediction')
		
	def test_rainy_classification_output(self):
		test_input = np.array([279.626,998,99,1,314,0.3,0,0,88]).reshape(1,-1)
		class_result, _ = classify_weather(test_input) 
		# Ensure that 'rainy' class is returned
		self.assertEqual(class_result, 'rainy', 'Rainy weather input should be classified as "rainy", check that app.py returns the model\'s prediction')
		

	def test_foggy_classification_output(self):
		test_input = np.array([289.47,1015,88,2,300,0,0,0,20]).reshape(1,-1)
		class_result, _ = classify_weather(test_input) 
		# Ensure that 'foggy' class is returned
		self.assertEqual(class_result, 'foggy', 'Foggy weather input should be classified as "foggy", check that app.py returns the model\'s prediction')
		
if __name__ == '__main__':
	unittest.main()
